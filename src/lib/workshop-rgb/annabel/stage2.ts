// RGB-ANNAbel stage 2 (Magaña et al., Case Stud. Therm. Eng. 76 (2025) 107396), ported from
// fuentes/rgb-annabel/annabel_app/core/stage2.py with config.Stage2Config, models.py and
// calibration.KernelTable. A second network maps (r̃², z̃, t̃) to (T, f_s) through sigmoid bounds,
// T = T_lo + (T_hi − T_lo) σ(o₀) and f_s = f_s,ref σ(o₁). The loss is the direct fit of the
// stage-1 emission, mean over the grid of Σ_c w_c ((J_c − f_s K_c(T)) / nrm_c)² / Σ w, plus
// λ_ratio Σ_pairs Σ_mask ((J_a/J_b − K_a(T)/K_b(T)) / rn)² / n_mask on the signal mask, with
// K_c(T) interpolated linearly in the kernel table. Only channels with chan_w > 0 enter either
// term. Adam with the cosine schedule and the best-before-step checkpoint (train.ts), all
// samples, no validation. The closed-form per-pixel fit (solve_T_fs) is the cross-check
// (_xcheck). The stage-3 ensemble is not ported.
//
// Parameter layout is torch's state_dict order, as in net.ts. Accumulators are JS doubles, the
// stored buffers Float32, so the CPU path follows PyTorch's float32 to about 1e-6.
import { layout, initXavier } from './net.ts';
import type { CoordNorm } from './problem.ts';
import { coordNormFromGrid, coordFeatures } from './problem.ts';
import type { GradEngine, TrainOptions, TrainResult } from './train.ts';
import { trainWithEngine } from './train.ts';
import { solveTfs } from '../radiometry.ts';

/** calibration.KernelTable, the part stage 2 reads. K is (3, NT) row-major in float32. */
export interface KernelTable {
  Tgrid: ArrayLike<number>;
  K: Float32Array;
  chanW: [number, number, number];
  /** stage-1 J → kernel units, 1e3 for J per mm with K in W m⁻³ sr⁻¹; 1 when J is already SI */
  jToSi: number;
  /** sigmoid upper bound on f_s, 2e-5 (20 ppm) in absolute mode */
  fsRef: number;
}

/** config.Stage2Config defaults (NetConfig 4 × 64 tanh without Fourier features, TrainConfig 3000 iterations). */
export interface Stage2Config {
  hiddenLayers: number; width: number;
  ff: { r: number[]; z: number[]; t: number[] };
  iterations: number; lr: number; lrMinFrac: number; seed: number;
  zSub: number; signalThreshold: number; lambdaRatio: number;
}
export const STAGE2_DEFAULTS: Readonly<Stage2Config> = Object.freeze({
  hiddenLayers: 4, width: 64, ff: { r: [], z: [], t: [] },
  iterations: 3000, lr: 1e-3, lrMinFrac: 0.1, seed: 42,
  zSub: 2, signalThreshold: 0.02, lambdaRatio: 1e-2,
});

export function stage2Sizes(cfg: Pick<Stage2Config, 'hiddenLayers' | 'width' | 'ff'> = STAGE2_DEFAULTS): number[] {
  const nIn = 3 + 2 * (cfg.ff.r.length + cfg.ff.z.length + cfg.ff.t.length);
  return [nIn, ...Array.from({ length: cfg.hiddenLayers }, () => cfg.width), 2];
}

/** models.FourierEmbed: [u, sin(π k u_d), cos(π k u_d) for d, k], float32. Identity without frequencies. */
export function fourierEmbed(X: Float32Array, ff: Stage2Config['ff']): Float32Array {
  const lists = [ff.r, ff.z, ff.t];
  const extra = 2 * (ff.r.length + ff.z.length + ff.t.length);
  if (!extra) return X;
  const n = X.length / 3, d = 3 + extra, out = new Float32Array(n * d);
  const f = Math.fround;
  for (let s = 0; s < n; s++) {
    let o = s * d;
    out[o++] = X[3 * s]; out[o++] = X[3 * s + 1]; out[o++] = X[3 * s + 2];
    for (let c = 0; c < 3; c++) for (const k of lists[c]) {
      const ang = f(f(Math.PI * k) * X[3 * s + c]);
      out[o++] = Math.sin(ang); out[o++] = Math.cos(ang);
    }
  }
  return out;
}

/** Stage2Problem.__init__, everything the loss needs. rows = Nt · Nz2 rows of Nr samples, sample
 *  n = row · Nr + i in (t, z, r) order, z subsampled by z_sub. */
export interface Stage2Problem {
  nt: number; nz: number; nr: number; nz2: number;
  rows: number; N: number; nIn: number; sizes: number[];
  X: Float32Array;            // (N, nIn) network inputs on the training grid
  Y: Float32Array;            // (N, 3) J · j_to_si, float32
  nrm2: Float32Array;         // (3) max of each channel, at least 1e-30
  w: [number, number, number];
  wSum: number;
  useCh: number[];
  pairs: [number, number][];  // (numerator, denominator), [(i, j) for k, i in enumerate(uc) for j in uc[:k]]
  ratios: Float32Array[];     // per pair (N), J_a / J_b on the mask, 0 elsewhere
  ratioNrm: number[];         // per pair, max |ratio|, at least 1e-12
  mask: Uint8Array;           // (N) signal mask, every used channel above threshold × its maximum
  nTr: number;
  K: Float32Array;            // (3, NT) kernel table
  Tlo: number; Thi: number; nT: number;
  fsRef: number; lam: number;
  Tcf: Float32Array; fsCf: Float32Array;  // closed form on the training grid, NaN outside the mask
  xmask: Uint8Array;          // strong-signal pixels of the cross-check
  norm: CoordNorm;
  zIdx: number[];             // indices of the full z grid used for training
  cfg: Stage2Config;
}

export function buildStage2Problem(J: ArrayLike<number>, grid: { nt: number; nz: number; nr: number; rMm: ArrayLike<number>; zMm: ArrayLike<number>; times: ArrayLike<number> },
                                   kern: KernelTable, cfgIn: Partial<Stage2Config> = {}): Stage2Problem {
  const cfg: Stage2Config = { ...STAGE2_DEFAULTS, ...cfgIn, ff: cfgIn.ff ?? STAGE2_DEFAULTS.ff };
  const { nt, nz, nr } = grid;
  if (J.length !== nt * nz * nr * 3) throw new Error(`J has ${J.length} values, expected ${nt * nz * nr * 3}`);
  const zIdx: number[] = [];
  for (let k = 0; k < nz; k += cfg.zSub) zIdx.push(k);
  const nz2 = zIdx.length, rows = nt * nz2, N = rows * nr;
  const f = Math.fround;
  const Y = new Float32Array(N * 3);
  for (let t = 0; t < nt; t++) for (let kk = 0; kk < nz2; kk++) for (let i = 0; i < nr; i++)
    for (let c = 0; c < 3; c++) Y[((t * nz2 + kk) * nr + i) * 3 + c] = J[((t * nz + zIdx[kk]) * nr + i) * 3 + c] * kern.jToSi;
  const w = kern.chanW.map(Number) as [number, number, number];
  const useCh = [0, 1, 2].filter((c) => w[c] > 0);
  if (useCh.length < 2) throw new Error('stage2_two_channels');
  const ref = useCh.map((c) => { let m = -Infinity; for (let n = 0; n < N; n++) m = Math.max(m, Y[n * 3 + c]); return m; });
  const nrm2 = new Float32Array(3);
  for (let c = 0; c < 3; c++) { let m = -Infinity; for (let n = 0; n < N; n++) m = Math.max(m, Y[n * 3 + c]); nrm2[c] = Math.max(m, 1e-30); }
  const th = f(cfg.signalThreshold);
  const mask = new Uint8Array(N);
  let nTr = 0;
  for (let n = 0; n < N; n++) {
    let ok = 1;
    useCh.forEach((c, u) => { if (!(Y[n * 3 + c] > f(th * ref[u]))) ok = 0; });
    mask[n] = ok; nTr += ok;
  }
  if (nTr === 0) throw new Error('stage2_no_signal');
  const pairs: [number, number][] = [];
  useCh.forEach((i, k) => { for (const j of useCh.slice(0, k)) pairs.push([i, j]); });
  const ratios: Float32Array[] = [], ratioNrm: number[] = [];
  for (const [a, b] of pairs) {
    const rd = new Float32Array(N);
    let m = 0;
    for (let n = 0; n < N; n++) if (mask[n]) { rd[n] = Y[n * 3 + a] / Y[n * 3 + b]; m = Math.max(m, Math.abs(rd[n])); }
    ratios.push(rd); ratioNrm.push(Math.max(m, 1e-12));
  }
  const norm = coordNormFromGrid(grid.rMm, grid.zMm, grid.times);
  const zSubMm = zIdx.map((k) => grid.zMm[k]);
  const X = fourierEmbed(coordFeatures(norm, grid.rMm, zSubMm, grid.times), cfg.ff);
  const nT = kern.Tgrid.length;
  if (kern.K.length !== 3 * nT) throw new Error('kernel table is not (3, NT)');
  const Tlo = Number(kern.Tgrid[0]), Thi = Number(kern.Tgrid[nT - 1]);
  // closed form of the same model on the same emission (rc.solve_T_fs in float64)
  const G = kernelRgb(kern);
  const cf = solveTfs(Float64Array.from(Y), G, kern.Tgrid, w, mask);
  const Tcf = Float32Array.from(cf.T), fsCf = Float32Array.from(cf.fs);
  const xmask = new Uint8Array(N);
  for (let n = 0; n < N; n++) xmask[n] = mask[n] && cf.T[n] > Tlo + 5 && cf.T[n] < Thi - 5 && cf.fs[n] > 0 ? 1 : 0;
  return {
    nt, nz, nr, nz2, rows, N, nIn: X.length / N, sizes: stage2Sizes(cfg), X, Y, nrm2, w, wSum: w[0] + w[1] + w[2], useCh, pairs, ratios, ratioNrm,
    mask, nTr, K: kern.K, Tlo, Thi, nT, fsRef: kern.fsRef, lam: cfg.lambdaRatio, Tcf, fsCf, xmask, norm, zIdx, cfg,
  };
}

function kernelRgb(kern: KernelTable) {
  const nT = kern.Tgrid.length;
  const row = (c: number) => Float64Array.from(kern.K.subarray(c * nT, (c + 1) * nT));
  return { R: row(0), G: row(1), B: row(2) };
}

/** Generic MLP (Linear → tanh → … → Linear) in Float32 by blocks of 4 samples. */
export class Mlp {
  sizes: number[]; L: number; wOff: number[]; bOff: number[]; nParams: number; maxN: number;
  H: Float32Array[]; Y: Float32Array; dA: Float32Array; dB: Float32Array;
  constructor(maxN: number, sizes: number[]) {
    for (let l = 1; l < sizes.length - 1; l++) if (sizes[l] % 4) throw new Error('hidden widths must be multiples of 4');
    this.sizes = sizes; this.L = sizes.length - 1;
    const lo = layout(sizes); this.wOff = lo.wOff; this.bOff = lo.bOff; this.nParams = lo.nParams;
    this.maxN = Math.ceil(maxN / 4) * 4;
    this.H = sizes.slice(0, -1).map((s) => new Float32Array(this.maxN * s));
    this.Y = new Float32Array(this.maxN * sizes[this.L]);
    const wmax = Math.max(...sizes);
    this.dA = new Float32Array(this.maxN * wmax); this.dB = new Float32Array(this.maxN * wmax);
  }
  forward(p: ArrayLike<number>, X: ArrayLike<number>, n0: number, n: number) {
    if (n > this.maxN) throw new Error('batch larger than the work buffers');
    const { sizes, L, wOff, bOff, H, Y } = this;
    const n4 = Math.ceil(n / 4) * 4, d0 = sizes[0];
    const H0 = H[0];
    for (let q = 0; q < n * d0; q++) H0[q] = X[n0 * d0 + q];
    for (let q = n * d0; q < n4 * d0; q++) H0[q] = 0;
    for (let l = 0; l < L; l++) {
      const K = sizes[l], Mo = sizes[l + 1], Wo = wOff[l], Bo = bOff[l];
      const inp = H[l], out = l === L - 1 ? Y : H[l + 1], act = l < L - 1;
      for (let s = 0; s < n4; s += 4) {
        const x0 = s * K, x1 = x0 + K, x2 = x1 + K, x3 = x2 + K, oi = s * Mo;
        for (let m = 0; m < Mo; m++) {
          const bb = p[Bo + m];
          let a0 = bb, a1 = bb, a2 = bb, a3 = bb;
          const wi = Wo + m * K;
          for (let k = 0; k < K; k++) {
            const wk = p[wi + k];
            a0 += inp[x0 + k] * wk; a1 += inp[x1 + k] * wk; a2 += inp[x2 + k] * wk; a3 += inp[x3 + k] * wk;
          }
          if (act) { a0 = Math.tanh(a0); a1 = Math.tanh(a1); a2 = Math.tanh(a2); a3 = Math.tanh(a3); }
          out[oi + m] = a0; out[oi + Mo + m] = a1; out[oi + 2 * Mo + m] = a2; out[oi + 3 * Mo + m] = a3;
        }
      }
    }
  }
  /** Adds into grad the backpropagation of dOut (n, out) through the last forward pass. dOut is
   *  overwritten (it must be this.dA). */
  backward(p: ArrayLike<number>, n: number, grad: Float32Array | Float64Array) {
    const { sizes, L, wOff, bOff, H } = this;
    const n4 = Math.ceil(n / 4) * 4;
    let dY = this.dA, dX = this.dB;
    for (let q = n * sizes[L]; q < n4 * sizes[L]; q++) dY[q] = 0;
    for (let l = L - 1; l >= 0; l--) {
      const K = sizes[l], Mo = sizes[l + 1], Wo = wOff[l], Bo = bOff[l];
      const inp = H[l];
      for (let s = 0; s < n4; s += 4) {
        const x0 = s * K, x1 = x0 + K, x2 = x1 + K, x3 = x2 + K, di = s * Mo;
        for (let m = 0; m < Mo; m++) {
          const d0 = dY[di + m], d1 = dY[di + Mo + m], d2 = dY[di + 2 * Mo + m], d3 = dY[di + 3 * Mo + m];
          grad[Bo + m] += d0 + d1 + d2 + d3;
          const wi = Wo + m * K;
          for (let k = 0; k < K; k++) grad[wi + k] += d0 * inp[x0 + k] + d1 * inp[x1 + k] + d2 * inp[x2 + k] + d3 * inp[x3 + k];
        }
      }
      if (l === 0) break;
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
  }
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Stage2Problem.K_of_T and its derivative, for the three channels. */
function kOfT(prob: Stage2Problem, T: number, K: Float64Array, dK: Float64Array) {
  const { nT, Tlo, Thi } = prob, Kt = prob.K;
  const s = (nT - 1) / (Thi - Tlo);
  const u = (T - Tlo) * s;
  const i0 = Math.min(Math.max(Math.floor(u), 0), nT - 2);
  const w = Math.min(Math.max(u - i0, 0), 1);
  for (let c = 0; c < 3; c++) {
    const k0 = Kt[c * nT + i0], k1 = Kt[c * nT + i0 + 1];
    K[c] = k0 * (1 - w) + k1 * w; dK[c] = (k1 - k0) * s;
  }
}

export interface Stage2Sums { sD: number; sQ: number }

/** Rows [row0, row1): adds the gradient of the GLOBAL loss into grad (caller zeroes it) and
 *  returns the unnormalised sums, sD = Σ_n Σ_c w_c e_c² / Σ w and sQ = Σ_pairs Σ_mask e². The loss
 *  is sD / N + λ sQ / n_mask. */
export function stage2LossGrad(net: Mlp, p: ArrayLike<number>, prob: Stage2Problem, row0: number, row1: number, grad: Float32Array | Float64Array | null): Stage2Sums {
  const nr = prob.nr, n0 = row0 * nr, n = (row1 - row0) * nr;
  net.forward(p, prob.X, n0, n);
  const O = net.Y, dO = net.dA;
  const { w, wSum, nrm2, Y, mask, pairs, ratios, ratioNrm, lam, fsRef, Tlo, Thi } = prob;
  const dTr = Thi - Tlo;
  const gD = 2 / (wSum * prob.N), gQ = 2 * lam / prob.nTr;
  const K = new Float64Array(3), dK = new Float64Array(3), gK = new Float64Array(3);
  let sD = 0, sQ = 0;
  for (let s = 0; s < n; s++) {
    const g = n0 + s;
    const s0 = sigmoid(O[2 * s]), s1 = sigmoid(O[2 * s + 1]);
    const T = Tlo + dTr * s0, fs = fsRef * s1;
    kOfT(prob, T, K, dK);
    let ld = 0, gfs = 0;
    for (let c = 0; c < 3; c++) {
      gK[c] = 0;
      if (!w[c]) continue;
      const e = (Y[g * 3 + c] - fs * K[c]) / nrm2[c];
      ld += w[c] * e * e;
      const d = -gD * w[c] * e / nrm2[c];
      gfs += d * K[c]; gK[c] = d * fs;
    }
    sD += ld / wSum;
    if (mask[g]) for (let q = 0; q < pairs.length; q++) {
      const [a, b] = pairs[q], rn = ratioNrm[q];
      const pr = K[a] / K[b];
      const e = (ratios[q][g] - pr) / rn;
      sQ += e * e;
      const d = -gQ * e / rn;
      gK[a] += d / K[b]; gK[b] -= d * K[a] / (K[b] * K[b]);
    }
    const dT = gK[0] * dK[0] + gK[1] * dK[1] + gK[2] * dK[2];
    dO[2 * s] = dT * dTr * s0 * (1 - s0);
    dO[2 * s + 1] = gfs * fsRef * s1 * (1 - s1);
  }
  if (grad) net.backward(p, n, grad);
  return { sD, sQ };
}

export const stage2Loss = (prob: Stage2Problem, s: Stage2Sums) => {
  const direct = s.sD / prob.N, ratio = s.sQ / prob.nTr;
  return { loss: direct + prob.lam * ratio, direct, ratio };
};

/** Same-thread engine (Node tests, a helper Worker, or the last fallback). */
export class Stage2LocalEngine implements GradEngine {
  readonly nParams: number;
  readonly path = 'cpu' as const;
  prob: Stage2Problem; net: Mlp; rowsPerBlock: number;
  constructor(prob: Stage2Problem, rowsPerBlock = 8) {
    this.prob = prob; this.rowsPerBlock = Math.max(1, Math.min(rowsPerBlock, prob.rows));
    this.net = new Mlp(this.rowsPerBlock * prob.nr, prob.sizes);
    this.nParams = this.net.nParams;
  }
  sums(p: ArrayLike<number>, row0: number, row1: number, grad: Float64Array | null): Stage2Sums {
    let sD = 0, sQ = 0;
    for (let a = row0; a < row1; a += this.rowsPerBlock) {
      const s = stage2LossGrad(this.net, p, this.prob, a, Math.min(row1, a + this.rowsPerBlock), grad);
      sD += s.sD; sQ += s.sQ;
    }
    return { sD, sQ };
  }
  async evaluate(p: Float32Array, withGrad: boolean) {
    const grad = withGrad ? new Float64Array(this.nParams) : null;
    const l = stage2Loss(this.prob, this.sums(p, 0, this.prob.rows, grad));
    return { loss: l.loss, lR: l.direct, lL: l.ratio, grad };
  }
  dispose() { /* nothing outside the heap */ }
}

/** Initial weights: Xavier normal and zero biases, as build_net (mulberry32, not torch's generator). */
export function initStage2(seed = STAGE2_DEFAULTS.seed, sizes = stage2Sizes()): Float32Array { return initXavier(seed, sizes); }

/** Network output (T, f_s) for inputs X (n, nIn). */
export function predictTfs(p: ArrayLike<number>, prob: Pick<Stage2Problem, 'sizes' | 'Tlo' | 'Thi' | 'fsRef'>, X: Float32Array, block = 512): { T: Float32Array; fs: Float32Array } {
  const d = prob.sizes[0], n = X.length / d;
  const net = new Mlp(Math.min(block, n), prob.sizes);
  const T = new Float32Array(n), fs = new Float32Array(n);
  for (let s0 = 0; s0 < n; s0 += block) {
    const m = Math.min(block, n - s0);
    net.forward(p, X, s0, m);
    for (let s = 0; s < m; s++) {
      T[s0 + s] = prob.Tlo + (prob.Thi - prob.Tlo) * sigmoid(net.Y[2 * s]);
      fs[s0 + s] = prob.fsRef * sigmoid(net.Y[2 * s + 1]);
    }
  }
  return { T, fs };
}

// ---------------------------------------------------------------------------- diagnostics
export const XCHECK_WARN_K = 20, XCHECK_WARN_P90_K = 60, XCHECK_WARN_FS_LOG = 0.2;

/** np.percentile with linear interpolation (q in 0..100). */
export function percentile(v: ArrayLike<number>, q: number): number {
  const a = Float64Array.from(v).sort();
  if (!a.length) return NaN;
  const h = (a.length - 1) * q / 100, lo = Math.floor(h), hi = Math.min(a.length - 1, lo + 1);
  return a[lo] + (a[hi] - a[lo]) * (h - lo);
}

export interface XCheck {
  nPixels: number; TMedianDiffK: number; TMedianAbsDiffK: number; TP90AbsDiffK: number;
  fsMedianRatio: number; fsMedianAbsLogRatio: number; agrees: boolean;
  /** what failed, codes 'T_median', 'T_p90', 'fs' */
  bad: string[];
}

/** stage2._xcheck: network against the closed form on the training grid, strong-signal pixels. */
export function xcheck(prob: Stage2Problem, p: ArrayLike<number>): XCheck | null {
  const idx: number[] = [];
  for (let n = 0; n < prob.N; n++) if (prob.xmask[n]) idx.push(n);
  if (!idx.length) return null;
  const { T, fs } = predictTfs(p, prob, prob.X);
  const dT = idx.map((n) => Math.fround(T[n] - prob.Tcf[n]));
  const ratio = idx.map((n) => Math.fround(fs[n] / prob.fsCf[n]));
  const lf = ratio.map((r) => Math.abs(Math.log(r)));
  const r: XCheck = {
    nPixels: idx.length, TMedianDiffK: percentile(dT, 50), TMedianAbsDiffK: percentile(dT.map(Math.abs), 50),
    TP90AbsDiffK: percentile(dT.map(Math.abs), 90), fsMedianRatio: percentile(ratio, 50), fsMedianAbsLogRatio: percentile(lf, 50),
    agrees: true, bad: [],
  };
  if (r.TMedianAbsDiffK > XCHECK_WARN_K) r.bad.push('T_median');
  if (r.TP90AbsDiffK > XCHECK_WARN_P90_K) r.bad.push('T_p90');
  if (r.fsMedianAbsLogRatio > XCHECK_WARN_FS_LOG) r.bad.push('fs');
  r.agrees = !r.bad.length;
  return r;
}

/** stage2.display_mask: every fitted channel above threshold × its maximum, on the full grid. */
export function displayMask(J: ArrayLike<number>, chanW: [number, number, number], threshold: number): Uint8Array {
  const n = J.length / 3, uc = [0, 1, 2].filter((c) => chanW[c] > 0);
  const th = Math.fround(threshold);
  const mx = uc.map((c) => { let m = -Infinity; for (let q = 0; q < n; q++) m = Math.max(m, J[q * 3 + c]); return Math.fround(th * Math.fround(m)); });
  const out = new Uint8Array(n);
  for (let q = 0; q < n; q++) out[q] = uc.every((c, u) => Math.fround(J[q * 3 + c]) > mx[u]) ? 1 : 0;
  return out;
}

export interface BoundReport { fracTAtLower: number; fracTAtUpper: number; fracFsAtUpper: number; TMedian: number; TP02: number; TP98: number; warnT: boolean; warnFs: boolean }

/** stage2.bound_report on the display mask. */
export function boundReport(T: ArrayLike<number>, fs: ArrayLike<number>, mask: ArrayLike<number>, kern: KernelTable): BoundReport | null {
  const Tm: number[] = [], fm: number[] = [];
  for (let q = 0; q < T.length; q++) if (mask[q]) { Tm.push(T[q]); fm.push(fs[q]); }
  if (!Tm.length) return null;
  const T0 = Number(kern.Tgrid[0]), T1 = Number(kern.Tgrid[kern.Tgrid.length - 1]), span = T1 - T0;
  const mean = (a: boolean[]) => a.filter(Boolean).length / a.length;
  const r = { fracTAtLower: mean(Tm.map((t) => t < T0 + 0.01 * span)), fracTAtUpper: mean(Tm.map((t) => t > T1 - 0.01 * span)),
    fracFsAtUpper: mean(fm.map((f) => f > 0.95 * kern.fsRef)), TMedian: percentile(Tm, 50), TP02: percentile(Tm, 2), TP98: percentile(Tm, 98), warnT: false, warnFs: false };
  r.warnT = r.fracTAtLower > 0.02 || r.fracTAtUpper > 0.02;
  r.warnFs = r.fracFsAtUpper > 0.02;
  return r;
}

export interface Stage2Result {
  nt: number; nz: number; nr: number;
  /** (Nt, Nz, Nr) on the full grid, K and volume fraction */
  T: Float32Array; fs: Float32Array;
  /** display mask on the full grid */
  mask: Uint8Array;
  /** closed form on the full grid inside the display mask, NaN elsewhere and where it sits on a bound of the T grid */
  Tcf: Float32Array; fsCf: Float32Array;
  xcheck: XCheck | null;
  bounds: BoundReport | null;
  train: TrainResult;
  meta: { sizes: number[]; Tlo: number; Thi: number; fsRef: number; zSub: number; lambdaRatio: number; signalThreshold: number; nTrain: number; nMask: number; output: string };
}

/** _train_one after training: the full grid, the display mask, the closed form, the reports. */
export function finishStage2(prob: Stage2Problem, J: ArrayLike<number>, grid: { rMm: ArrayLike<number>; zMm: ArrayLike<number>; times: ArrayLike<number> }, kern: KernelTable, train: TrainResult): Stage2Result {
  const p = train.params;
  const Xf = fourierEmbed(coordFeatures(prob.norm, grid.rMm, grid.zMm, grid.times), prob.cfg.ff);
  const { T, fs } = predictTfs(p, prob, Xf);
  const mask = displayMask(J, kern.chanW, prob.cfg.signalThreshold);
  const jsi = Float64Array.from(J, (v) => Math.fround(v) * kern.jToSi);
  const cf = solveTfs(jsi, kernelRgb(kern), kern.Tgrid, kern.chanW, mask);
  const T0 = Number(kern.Tgrid[0]), T1 = Number(kern.Tgrid[kern.Tgrid.length - 1]);
  const Tcf = new Float32Array(T.length), fsCf = new Float32Array(T.length);
  for (let q = 0; q < T.length; q++) {
    const pinned = cf.T[q] <= T0 + 0.5 || cf.T[q] >= T1 - 0.5;
    Tcf[q] = pinned ? NaN : cf.T[q]; fsCf[q] = pinned ? NaN : cf.fs[q];
  }
  return {
    nt: prob.nt, nz: prob.nz, nr: prob.nr, T, fs, mask, Tcf, fsCf, xcheck: xcheck(prob, p), bounds: boundReport(T, fs, mask, kern), train,
    meta: { sizes: prob.sizes, Tlo: prob.Tlo, Thi: prob.Thi, fsRef: prob.fsRef, zSub: prob.cfg.zSub, lambdaRatio: prob.lam, signalThreshold: prob.cfg.signalThreshold,
            nTrain: prob.N, nMask: prob.nTr, output: 'T = T_lo + (T_hi - T_lo) sigmoid(o0); fs = fs_ref sigmoid(o1)' },
  };
}

export type Stage2Path = 'webgpu' | 'workers' | 'cpu';

/** CPU training on any engine (one thread or the Worker pool). */
export function trainStage2(engine: GradEngine, p0: Float32Array, prob: Stage2Problem, o: Omit<TrainOptions, 'iterations'> & { iterations?: number }): Promise<TrainResult> {
  return trainWithEngine(engine, p0, { lr: prob.cfg.lr, lrMinFrac: prob.cfg.lrMinFrac, ...o, iterations: o.iterations ?? prob.cfg.iterations });
}
