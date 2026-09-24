/**
 * PINN for flame emission tomography (mechanical and energy tab).
 *
 * A coordinate network J_θ(r) ≥ 0 is trained so that its line-of-sight projection
 * through the physical forward operator matches the measured section:
 *     L(θ) = ‖T_κ J_θ − S‖² / ‖S‖²,
 * where T_κ integrates the emission along each line of sight with self-absorption
 * (`trappingMatrix` in ../sat.ts). Putting the Abel operator inside the loss of a neural
 * network, including signal trapping, is the ANNAbel approach (Escudero et al.,
 * Proc. Combust. Inst. 40 (2024) 105493); this is a compact browser version of it.
 *
 * Network: J_θ(r) = softplus(c + Σ_j a_j tanh(w_j u + b_j)) with u = 2r² − 1. Using r²
 * as input makes J even in r, so the field is automatically smooth on the axis
 * (J'(0) = 0), and softplus keeps it non-negative. Gradients are analytic; Adam.
 */
import { rng, type Mat } from '../linalg.ts';

export interface FlameNet {
  a: Float64Array; w: Float64Array; b: Float64Array; c: Float64Array;
  m: Float64Array; v: Float64Array; t: number;
}

export function initFlameNet(hidden: number, seed: number): FlameNet {
  const R = rng(seed), P = 3 * hidden + 1;
  const net: FlameNet = {
    a: new Float64Array(hidden), w: new Float64Array(hidden), b: new Float64Array(hidden), c: new Float64Array([-1]),
    m: new Float64Array(P), v: new Float64Array(P), t: 0,
  };
  for (let j = 0; j < hidden; j++) {
    net.w[j] = R.normal() * 4;
    net.b[j] = R.normal() * 2;
    net.a[j] = R.normal() * 0.1;
  }
  return net;
}

const softplus = (z: number) => (z > 20 ? z : Math.log1p(Math.exp(z)));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function evalFlameNet(net: FlameNet, r: ArrayLike<number>): Float64Array {
  const out = new Float64Array(r.length);
  for (let i = 0; i < r.length; i++) {
    const u = 2 * r[i] * r[i] - 1;
    let z = net.c[0];
    for (let j = 0; j < net.a.length; j++) z += net.a[j] * Math.tanh(net.w[j] * u + net.b[j]);
    out[i] = softplus(z);
  }
  return out;
}

/** Relative data misfit and its gradient with respect to [a, w, b, c]. */
export function flameLossGrad(net: FlameNet, A: Mat, r: ArrayLike<number>, g: ArrayLike<number>) {
  const n = r.length, H = net.a.length;
  const S = new Float64Array(n * H), z = new Float64Array(n), f = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = 2 * r[i] * r[i] - 1;
    let zi = net.c[0];
    for (let j = 0; j < H; j++) { const s = Math.tanh(net.w[j] * u + net.b[j]); S[i * H + j] = s; zi += net.a[j] * s; }
    z[i] = zi; f[i] = softplus(zi);
  }
  let gg = 0;
  for (let i = 0; i < n; i++) gg += g[i] * g[i];
  gg = gg || 1;
  const res = new Float64Array(n);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    let s = -g[i];
    for (let k = 0; k < n; k++) s += A.a[i * n + k] * f[k];
    res[i] = s; loss += s * s;
  }
  loss /= gg;
  const q = new Float64Array(n); // ∂L/∂z = (2/‖g‖²)(Aᵀ res) ⊙ σ(z)
  for (let k = 0; k < n; k++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += A.a[i * n + k] * res[i];
    q[k] = (2 / gg) * s * sigmoid(z[k]);
  }
  const grad = new Float64Array(3 * H + 1);
  for (let i = 0; i < n; i++) {
    const u = 2 * r[i] * r[i] - 1, qi = q[i];
    for (let j = 0; j < H; j++) {
      const s = S[i * H + j], d = qi * net.a[j] * (1 - s * s);
      grad[j] += qi * s;
      grad[H + j] += d * u;
      grad[2 * H + j] += d;
    }
    grad[3 * H] += qi;
  }
  return { loss, grad, f };
}

export function flameAdam(net: FlameNet, grad: Float64Array, lr = 0.02, b1 = 0.9, b2 = 0.999) {
  net.t++;
  const c1 = 1 - b1 ** net.t, c2 = 1 - b2 ** net.t;
  let p = 0;
  for (const arr of [net.a, net.w, net.b, net.c]) for (let j = 0; j < arr.length; j++, p++) {
    net.m[p] = b1 * net.m[p] + (1 - b1) * grad[p];
    net.v[p] = b2 * net.v[p] + (1 - b2) * grad[p] * grad[p];
    arr[j] -= (lr * (net.m[p] / c1)) / (Math.sqrt(net.v[p] / c2) + 1e-8);
  }
}
