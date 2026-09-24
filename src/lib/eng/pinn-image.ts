/**
 * Physics-informed neural field for image deblurring (computing tab).
 *
 * The sharp image is a coordinate network built from sinusoidal filters, in the
 * multiplicative form of multiplicative filter networks (Fathony et al., ICLR 2021):
 *     f_θ(x, y) = σ( c₀ + Σ_{j,l} A_jl · sin(ω_j x + φ_j) · sin(ν_l y + ψ_l) ),   x, y ∈ [−1, 1],
 * with σ the logistic function (grey values in [0, 1]). Everything is trainable: the
 * frequencies and phases of the filters (random-Fourier-feature initialisation) and the
 * mixing weights A. The network is trained only through the known blur, the physics of
 * the camera:
 *     L(θ) = (1/N²) ‖h * f_θ − g‖² + μ ‖A‖².
 * It never sees the sharp image; its band-limited representation acts as the prior.
 *
 * Because the filters are separable, a forward/backward pass costs a few (N×K)(K×K) and
 * (K×N)(N×N) products; the convolution and its adjoint use the FFT helpers of deblur.ts.
 */
import { rng } from '../linalg.ts';
import { convolveSpec, correlateSpec, type Spec } from './deblur.ts';

export interface ImageNet {
  K: number;
  wx: Float64Array; px: Float64Array; wy: Float64Array; py: Float64Array; // K each
  A: Float64Array; // K × K, row = y filter l, column = x filter j
  c0: number;
}

/** Frequencies ~ N(0, (π·bandwidth)²): bandwidth ≈ cycles across half the image. */
export function initImageNet(K: number, seed: number, bandwidth = 8): ImageNet {
  const R = rng(seed);
  const mk = (f: () => number) => Float64Array.from({ length: K }, f);
  return {
    K,
    wx: mk(() => Math.PI * bandwidth * R.normal()), px: mk(() => 2 * Math.PI * R.uni()),
    wy: mk(() => Math.PI * bandwidth * R.normal()), py: mk(() => 2 * Math.PI * R.uni()),
    A: Float64Array.from({ length: K * K }, () => (0.3 / K) * R.normal()),
    c0: 1,
  };
}

const axis = (n: number) => Float64Array.from({ length: n }, (_, i) => -1 + (2 * i + 1) / n);

function filters(w: Float64Array, p: Float64Array, u: Float64Array) {
  const K = w.length, n = u.length, S = new Float64Array(K * n), C = new Float64Array(K * n);
  for (let j = 0; j < K; j++) for (let i = 0; i < n; i++) { const a = w[j] * u[i] + p[j]; S[j * n + i] = Math.sin(a); C[j * n + i] = Math.cos(a); }
  return { S, C };
}

/** M = Syᵀ A  (n × K): M[y][j] = Σ_l Sy[l][y] A[l][j]. */
function mixY(net: ImageNet, Sy: Float64Array, n: number) {
  const K = net.K, M = new Float64Array(n * K);
  for (let l = 0; l < K; l++) for (let y = 0; y < n; y++) {
    const s = Sy[l * n + y];
    if (s === 0) continue;
    const row = y * K, al = l * K;
    for (let j = 0; j < K; j++) M[row + j] += s * net.A[al + j];
  }
  return M;
}

/** Pre-activation z and image f = σ(z), n × n, row-major (row = y). */
export function evalImage(net: ImageNet, n: number) {
  const u = axis(n), K = net.K;
  const fx = filters(net.wx, net.px, u), fy = filters(net.wy, net.py, u);
  const M = mixY(net, fy.S, n);
  const z = new Float64Array(n * n).fill(net.c0);
  for (let y = 0; y < n; y++) {
    const row = y * n, mr = y * K;
    for (let j = 0; j < K; j++) {
      const m = M[mr + j], o = j * n;
      for (let x = 0; x < n; x++) z[row + x] += m * fx.S[o + x];
    }
  }
  const f = Float64Array.from(z, (v) => 1 / (1 + Math.exp(-v)));
  return { z, f, u, fx, fy, M };
}

/** Loss and gradient packed as [wx, px, wy, py, A, c0]. */
export function imageLossGrad(net: ImageNet, Hs: Spec, g: ArrayLike<number>, mu = 1e-6) {
  const n = Hs.n, NN = n * n, K = net.K;
  const { f, u, fx, fy, M } = evalImage(net, n);
  const hf = convolveSpec(Hs, f);
  const r = new Float64Array(NN);
  let data = 0;
  for (let k = 0; k < NN; k++) { r[k] = hf[k] - g[k]; data += r[k] * r[k]; }
  data /= NN;
  const dF = correlateSpec(Hs, r);
  const Q = new Float64Array(NN);
  let gc0 = 0;
  for (let k = 0; k < NN; k++) { Q[k] = (2 / NN) * dF[k] * f[k] * (1 - f[k]); gc0 += Q[k]; }

  // P = Q Sxᵀ  (n × K): P[y][j] = Σ_x Q[y][x] Sx[j][x]
  const P = new Float64Array(n * K);
  for (let y = 0; y < n; y++) {
    const row = y * n;
    for (let j = 0; j < K; j++) {
      let s = 0; const o = j * n;
      for (let x = 0; x < n; x++) s += Q[row + x] * fx.S[o + x];
      P[y * K + j] = s;
    }
  }
  const grad = new Float64Array(4 * K + K * K + 1);
  const oA = 4 * K;
  // ∂A[l][j] = Σ_y Sy[l][y] P[y][j]
  for (let l = 0; l < K; l++) for (let y = 0; y < n; y++) {
    const s = fy.S[l * n + y], pr = y * K, al = oA + l * K;
    for (let j = 0; j < K; j++) grad[al + j] += s * P[pr + j];
  }
  for (let q = 0; q < K * K; q++) grad[oA + q] += 2 * mu * net.A[q];
  // x filters: ∂L/∂Sx[j][x] = Σ_y M[y][j] Q[y][x]
  for (let j = 0; j < K; j++) {
    let gw = 0, gp = 0; const o = j * n;
    for (let x = 0; x < n; x++) {
      let d = 0;
      for (let y = 0; y < n; y++) d += M[y * K + j] * Q[y * n + x];
      const c = d * fx.C[o + x];
      gw += c * u[x]; gp += c;
    }
    grad[j] = gw; grad[K + j] = gp;
  }
  // y filters: ∂L/∂Sy[l][y] = Σ_j A[l][j] P[y][j]
  for (let l = 0; l < K; l++) {
    let gw = 0, gp = 0; const o = l * n, al = l * K;
    for (let y = 0; y < n; y++) {
      let d = 0; const pr = y * K;
      for (let j = 0; j < K; j++) d += net.A[al + j] * P[pr + j];
      const c = d * fy.C[o + y];
      gw += c * u[y]; gp += c;
    }
    grad[2 * K + l] = gw; grad[3 * K + l] = gp;
  }
  grad[oA + K * K] = gc0;
  let reg = 0;
  for (const a of net.A) reg += a * a;
  return { loss: data + mu * reg, data, grad, f };
}

export function getImageParams(net: ImageNet): Float64Array {
  const K = net.K, p = new Float64Array(4 * K + K * K + 1);
  p.set(net.wx, 0); p.set(net.px, K); p.set(net.wy, 2 * K); p.set(net.py, 3 * K); p.set(net.A, 4 * K); p[4 * K + K * K] = net.c0;
  return p;
}
export function setImageParams(net: ImageNet, p: ArrayLike<number>) {
  const K = net.K;
  for (let j = 0; j < K; j++) { net.wx[j] = p[j]; net.px[j] = p[K + j]; net.wy[j] = p[2 * K + j]; net.py[j] = p[3 * K + j]; }
  for (let q = 0; q < K * K; q++) net.A[q] = p[4 * K + q];
  net.c0 = p[4 * K + K * K];
}

/** Adam with separate rates for the filters (lrW) and the mixing weights (lrA). */
export function makeImageTrainer(net: ImageNet, Hs: Spec, g: ArrayLike<number>, lrA = 0.02, lrW = 0.02, mu = 1e-6) {
  const K = net.K, P = 4 * K + K * K + 1, m = new Float64Array(P), s = new Float64Array(P), lr = new Float64Array(P);
  for (let k = 0; k < P; k++) lr[k] = k < 4 * K ? lrW : lrA;
  let t = 0;
  return {
    get iter() { return t; },
    step() {
      const r = imageLossGrad(net, Hs, g, mu);
      t++;
      const p = getImageParams(net), c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      for (let k = 0; k < P; k++) {
        m[k] = 0.9 * m[k] + 0.1 * r.grad[k];
        s[k] = 0.999 * s[k] + 0.001 * r.grad[k] * r.grad[k];
        p[k] -= (lr[k] * (m[k] / c1)) / (Math.sqrt(s[k] / c2) + 1e-12);
      }
      setImageParams(net, p);
      return r;
    },
  };
}
