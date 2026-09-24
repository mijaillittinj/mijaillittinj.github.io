/**
 * A small multilayer perceptron, 1 input → L hidden tanh layers of width W → 1 linear output,
 * with hand-written backpropagation (no autodiff) and Adam. Loss: mean squared error
 * plus an optional L2 weight decay ½κ‖w‖² on the weights (not the biases).
 */
import { rng } from './linalg.ts';

export interface Layer { W: Float64Array; b: Float64Array; nin: number; nout: number }
export interface MLP { layers: Layer[]; m: Float64Array[]; v: Float64Array[]; t: number }

export function initMLP(depth: number, width: number, seed: number): MLP {
  const R = rng(seed), sizes = [1, ...Array(depth).fill(width), 1];
  const layers: Layer[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const nin = sizes[l], nout = sizes[l + 1], sc = Math.sqrt(1 / nin) * (l === 0 ? 3 : 1);
    const W = Float64Array.from({ length: nin * nout }, () => R.normal() * sc);
    const b = Float64Array.from({ length: nout }, () => (l === 0 ? R.normal() : 0));
    layers.push({ W, b, nin, nout });
  }
  const params = paramsOf({ layers } as MLP);
  return { layers, m: params.map((p) => new Float64Array(p.length)), v: params.map((p) => new Float64Array(p.length)), t: 0 };
}

const paramsOf = (net: Pick<MLP, 'layers'>) => net.layers.flatMap((L) => [L.W, L.b]);
export const nParams = (net: MLP) => paramsOf(net).reduce((s, p) => s + p.length, 0);

export function predict(net: MLP, x: number): number {
  let h = [x];
  net.layers.forEach((L, l) => {
    const o = new Array(L.nout).fill(0);
    for (let j = 0; j < L.nout; j++) {
      let z = L.b[j];
      for (let i = 0; i < L.nin; i++) z += L.W[j * L.nin + i] * h[i];
      o[j] = l < net.layers.length - 1 ? Math.tanh(z) : z;
    }
    h = o;
  });
  return h[0];
}

export function mse(net: MLP, xs: ArrayLike<number>, ys: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (predict(net, xs[i]) - ys[i]) ** 2;
  return s / xs.length;
}

/** Loss (MSE + weight decay) and gradients, same layout as paramsOf. */
export function lossGrad(net: MLP, xs: ArrayLike<number>, ys: ArrayLike<number>, decay = 0) {
  const grads = paramsOf(net).map((p) => new Float64Array(p.length));
  const N = xs.length, nl = net.layers.length;
  let loss = 0;
  for (let n = 0; n < N; n++) {
    // forward, keeping activations
    const acts: number[][] = [[xs[n]]];
    for (let l = 0; l < nl; l++) {
      const L = net.layers[l], h = acts[l], o: number[] = new Array(L.nout);
      for (let j = 0; j < L.nout; j++) {
        let z = L.b[j];
        for (let i = 0; i < L.nin; i++) z += L.W[j * L.nin + i] * h[i];
        o[j] = l < nl - 1 ? Math.tanh(z) : z;
      }
      acts.push(o);
    }
    const r = acts[nl][0] - ys[n];
    loss += r * r;
    // backward: delta = ∂(r²/N)/∂z
    let delta = [(2 * r) / N];
    for (let l = nl - 1; l >= 0; l--) {
      const L = net.layers[l], h = acts[l], gW = grads[2 * l], gb = grads[2 * l + 1];
      const prev = new Array(L.nin).fill(0);
      for (let j = 0; j < L.nout; j++) {
        gb[j] += delta[j];
        for (let i = 0; i < L.nin; i++) { gW[j * L.nin + i] += delta[j] * h[i]; prev[i] += L.W[j * L.nin + i] * delta[j]; }
      }
      if (l > 0) delta = prev.map((g, i) => g * (1 - h[i] * h[i]));
    }
  }
  loss /= N;
  if (decay > 0) net.layers.forEach((L, l) => {
    for (let k = 0; k < L.W.length; k++) { loss += 0.5 * decay * L.W[k] ** 2; grads[2 * l][k] += decay * L.W[k]; }
  });
  return { loss, grads };
}

export function adam(net: MLP, grads: Float64Array[], lr = 0.01, b1 = 0.9, b2 = 0.999) {
  net.t++;
  const c1 = 1 - b1 ** net.t, c2 = 1 - b2 ** net.t;
  paramsOf(net).forEach((p, k) => {
    const g = grads[k], m = net.m[k], v = net.v[k];
    for (let i = 0; i < p.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
    }
  });
}

export function cloneParams(net: MLP): Float64Array[] { return paramsOf(net).map((p) => Float64Array.from(p)); }
export function setParams(net: MLP, ps: Float64Array[]) { paramsOf(net).forEach((p, k) => p.set(ps[k])); }

/** Target function and a noisy train/validation split on [0, 1]. */
export const target = (x: number) => Math.sin(2 * Math.PI * x) + 0.5 * Math.sin(5 * Math.PI * x) * x;

export function makeSplit(nTrain: number, nVal: number, noise: number, seed: number) {
  const R = rng(seed);
  // stratified positions (one jittered point per cell) so that no large gap is left uncovered
  const draw = (n: number) => {
    const x = Array.from({ length: n }, (_, i) => (i + 0.5 + 0.8 * (R.uni() - 0.5)) / n);
    return { x, y: x.map((v) => target(v) + noise * R.normal()) };
  };
  return { train: draw(nTrain), val: draw(nVal) };
}
