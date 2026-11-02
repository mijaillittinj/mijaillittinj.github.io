// ANNAbel stage-1 network in Float32, by batches. models.MLP: Linear → tanh → … → Linear with
// sizes 3 → 64 ×4 → 3, Xavier-normal weights and zero biases, output out_scale · softplus(y)
// (torch threshold 20). The stage-1 loss of stage1.step_fn with both halves and per-channel
// normalisation, and its analytic gradient by backpropagation. Parameter layout is torch's
// state_dict order: W0 (64×3), b0, W1 (64×64), b1, …, W4 (3×64), b4, W row-major [out][in].
// Accumulators are JS doubles, rounded when stored in the Float32 buffers.
import type { Stage1Problem } from './problem.ts';

export const SIZES: readonly number[] = [3, 64, 64, 64, 64, 3];

export function layout(sizes: readonly number[] = SIZES) {
  const wOff: number[] = [], bOff: number[] = [];
  let o = 0;
  for (let l = 0; l < sizes.length - 1; l++) {
    wOff.push(o); o += sizes[l] * sizes[l + 1];
    bOff.push(o); o += sizes[l + 1];
  }
  return { wOff, bOff, nParams: o };
}

/** mulberry32 and Box–Muller, deterministic for a seed. Not torch's generator, so the initial
 *  weights differ from the app's for the same seed (§11.2 does not ask for that). */
export function rng(seed: number) {
  let s = seed >>> 0;
  const u = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare: number | null = null;
  const normal = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let a = 0;
    while (a === 0) a = u();
    const rad = Math.sqrt(-2 * Math.log(a)), th = 2 * Math.PI * u();
    spare = rad * Math.sin(th);
    return rad * Math.cos(th);
  };
  return { u, normal };
}

/** torch.nn.init.xavier_normal_ (std = √(2 / (fan_in + fan_out))) and zero biases. */
export function initXavier(seed: number, sizes: readonly number[] = SIZES): Float32Array {
  const { wOff, nParams } = layout(sizes);
  const g = rng(seed);
  const p = new Float32Array(nParams);
  for (let l = 0; l < sizes.length - 1; l++) {
    const std = Math.sqrt(2 / (sizes[l] + sizes[l + 1]));
    for (let q = 0; q < sizes[l] * sizes[l + 1]; q++) p[wOff[l] + q] = std * g.normal();
  }
  return p;
}

const softplus = (y: number) => (y > 20 ? y : Math.log1p(Math.exp(y)));

export interface LossSums { sR: number; sL: number }

/** Work buffers for up to maxN samples. One instance per thread. */
export class Net {
  sizes: readonly number[];
  L: number;
  wOff: number[];
  bOff: number[];
  nParams: number;
  H: Float32Array[];   // H[0] inputs, H[1..L−1] activations after tanh
  Y: Float32Array;     // pre-softplus output (n, 3)
  Jn: Float32Array;    // out_scale · softplus(Y), normalised emission
  dP: Float32Array;    // ∂loss/∂P
  dA: Float32Array;
  dB: Float32Array;
  maxN: number;

  constructor(maxN: number, sizes: readonly number[] = SIZES) {
    if (sizes[0] !== 3 || sizes[sizes.length - 1] !== 3) throw new Error('stage 1 maps (r̃², z̃, t̃) to (J_R, J_G, J_B)');
    this.sizes = sizes; this.L = sizes.length - 1;
    const lo = layout(sizes);
    this.wOff = lo.wOff; this.bOff = lo.bOff; this.nParams = lo.nParams;
    this.maxN = Math.ceil(maxN / 4) * 4;
    const n4 = this.maxN;
    this.H = [];
    for (let l = 0; l < this.L; l++) this.H.push(new Float32Array(n4 * sizes[l]));
    this.Y = new Float32Array(n4 * 3); this.Jn = new Float32Array(n4 * 3); this.dP = new Float32Array(n4 * 3);
    const wmax = Math.max(...sizes);
    this.dA = new Float32Array(n4 * wmax); this.dB = new Float32Array(n4 * wmax);
  }

  /** Forward pass for samples X[n0 .. n0+n) (n ≤ maxN). Leaves Y and Jn (normalised, per mm
   *  divided by b_scale) for those samples. */
  forward(p: ArrayLike<number>, X: ArrayLike<number>, n0: number, n: number, outScale: number) {
    if (n > this.maxN) throw new Error('batch larger than the work buffers');
    const { sizes, L, wOff, bOff, H, Y, Jn } = this;
    const n4 = Math.ceil(n / 4) * 4;
    const H0 = H[0];
    for (let q = 0; q < n * 3; q++) H0[q] = X[n0 * 3 + q];
    for (let q = n * 3; q < n4 * 3; q++) H0[q] = 0;
    for (let l = 0; l < L; l++) {
      const K = sizes[l], Mo = sizes[l + 1], Wo = wOff[l], Bo = bOff[l];
      const inp = H[l];
      const out = l === L - 1 ? Y : H[l + 1];
      const act = l < L - 1;
      // 4 samples per pass share each weight load
      for (let s = 0; s < n4; s += 4) {
        const x0 = s * K, x1 = x0 + K, x2 = x1 + K, x3 = x2 + K, oi = s * Mo;
        for (let m = 0; m < Mo; m++) {
          const bb = p[Bo + m];
          let a0 = bb, a1 = bb, a2 = bb, a3 = bb;
          const wi = Wo + m * K;
          for (let k = 0; k < K; k++) {
            const wk = p[wi + k];
            a0 += inp[x0 + k] * wk; a1 += inp[x1 + k] * wk;
            a2 += inp[x2 + k] * wk; a3 += inp[x3 + k] * wk;
          }
          if (act) { a0 = Math.tanh(a0); a1 = Math.tanh(a1); a2 = Math.tanh(a2); a3 = Math.tanh(a3); }
          out[oi + m] = a0; out[oi + Mo + m] = a1; out[oi + 2 * Mo + m] = a2; out[oi + 3 * Mo + m] = a3;
        }
      }
    }
    for (let q = 0; q < n * 3; q++) Jn[q] = outScale * softplus(Y[q]);
  }

  /** Rows [row0, row1) of the problem: adds the gradient of the GLOBAL loss
   *  0.5 (l_R + l_L), l_h = mean over all N·3 terms of ((P − b_h)/nrm_c)², into grad (the
   *  caller zeroes it) and returns the unnormalised sums of ((P − b_h)/nrm_c)² over these rows. */
  lossGrad(p: ArrayLike<number>, prob: Stage1Problem, row0: number, row1: number, grad: Float32Array | Float64Array | null): LossSums {
    const { nr } = prob;
    const n0 = row0 * nr, n = (row1 - row0) * nr;
    const n4 = Math.ceil(n / 4) * 4;
    this.forward(p, prob.X, n0, n, prob.outScale);
    const { sizes, L, wOff, bOff, H, Y, Jn, dP } = this;
    const A = prob.A, bR = prob.bR, bL = prob.bL, w = prob.invNrm2, invM = 1 / prob.M;
    let sR = 0, sL = 0;
    for (let zz = 0; zz < row1 - row0; zz++) {
      const base = zz * nr * 3, gbase = (n0 + zz * nr) * 3;
      for (let i = 0; i < nr; i++) {
        let p0 = 0, p1 = 0, p2 = 0;
        const ai = i * nr;
        for (let j = i; j < nr; j++) {
          const a = A[ai + j], q = base + j * 3;
          p0 += a * Jn[q]; p1 += a * Jn[q + 1]; p2 += a * Jn[q + 2];
        }
        const q = base + i * 3, g = gbase + i * 3;
        const r0 = p0 - bR[g], r1 = p1 - bR[g + 1], r2 = p2 - bR[g + 2];
        const l0 = p0 - bL[g], l1 = p1 - bL[g + 1], l2 = p2 - bL[g + 2];
        sR += r0 * r0 * w[0] + r1 * r1 * w[1] + r2 * r2 * w[2];
        sL += l0 * l0 * w[0] + l1 * l1 * w[1] + l2 * l2 * w[2];
        dP[q] = (r0 + l0) * w[0] * invM;
        dP[q + 1] = (r1 + l1) * w[1] * invM;
        dP[q + 2] = (r2 + l2) * w[2] * invM;
      }
    }
    if (!grad) return { sR, sL };
    // dJn = Aᵀ dP, dY = dJn · out_scale · sigmoid(Y)
    const os = prob.outScale;
    let dY = this.dA;
    for (let zz = 0; zz < row1 - row0; zz++) {
      const base = zz * nr * 3;
      for (let j = 0; j < nr; j++) {
        let d0 = 0, d1 = 0, d2 = 0;
        for (let i = 0; i <= j; i++) {
          const a = A[i * nr + j], q = base + i * 3;
          d0 += a * dP[q]; d1 += a * dP[q + 1]; d2 += a * dP[q + 2];
        }
        const q = base + j * 3;
        dY[q] = d0 * os / (1 + Math.exp(-Y[q]));
        dY[q + 1] = d1 * os / (1 + Math.exp(-Y[q + 1]));
        dY[q + 2] = d2 * os / (1 + Math.exp(-Y[q + 2]));
      }
    }
    for (let q = n * 3; q < n4 * 3; q++) dY[q] = 0;   // padded samples carry no gradient
    let dX = this.dB;
    for (let l = L - 1; l >= 0; l--) {
      const K = sizes[l], Mo = sizes[l + 1], Wo = wOff[l], Bo = bOff[l];
      const inp = H[l];
      for (let s = 0; s < n4; s += 4) {
        const x0 = s * K, x1 = x0 + K, x2 = x1 + K, x3 = x2 + K, di = s * Mo;
        for (let m = 0; m < Mo; m++) {
          const d0 = dY[di + m], d1 = dY[di + Mo + m], d2 = dY[di + 2 * Mo + m], d3 = dY[di + 3 * Mo + m];
          grad[Bo + m] += d0 + d1 + d2 + d3;
          const wi = Wo + m * K;
          for (let k = 0; k < K; k++)
            grad[wi + k] += d0 * inp[x0 + k] + d1 * inp[x1 + k] + d2 * inp[x2 + k] + d3 * inp[x3 + k];
        }
      }
      if (l === 0) break;
      // dX[s,k] = (Σ_m dY[s,m] W[m,k]) (1 − h²), hidden widths are multiples of 4
      for (let s = 0; s < n4; s++) {
        const xi = s * K, di = s * Mo;
        for (let k = 0; k < K; k += 4) {
          let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
          for (let m = 0; m < Mo; m++) {
            const d = dY[di + m], wi = Wo + m * K + k;
            a0 += d * p[wi]; a1 += d * p[wi + 1]; a2 += d * p[wi + 2]; a3 += d * p[wi + 3];
          }
          const h0 = inp[xi + k], h1 = inp[xi + k + 1], h2 = inp[xi + k + 2], h3 = inp[xi + k + 3];
          dX[xi + k] = a0 * (1 - h0 * h0); dX[xi + k + 1] = a1 * (1 - h1 * h1);
          dX[xi + k + 2] = a2 * (1 - h2 * h2); dX[xi + k + 3] = a3 * (1 - h3 * h3);
        }
      }
      const t = dY; dY = dX; dX = t;
    }
    return { sR, sL };
  }
}

/** Normalised emission Jn for all samples (N, 3), by blocks of rows. Multiply by b_scale for
 *  the app's J in data units per mm. */
export function predictNorm(p: ArrayLike<number>, prob: Stage1Problem, rowsPerBlock = 16): Float32Array {
  const net = new Net(rowsPerBlock * prob.nr);
  const out = new Float32Array(prob.N * 3);
  for (let r0 = 0; r0 < prob.rows; r0 += rowsPerBlock) {
    const r1 = Math.min(prob.rows, r0 + rowsPerBlock), n = (r1 - r0) * prob.nr;
    net.forward(p, prob.X, r0 * prob.nr, n, prob.outScale);
    out.set(net.Jn.subarray(0, n * 3), r0 * prob.nr * 3);
  }
  return out;
}
