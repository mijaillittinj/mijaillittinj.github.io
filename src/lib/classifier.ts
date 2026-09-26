/**
 * A small neural-network classifier for 2D points (two classes), trained in the browser.
 *
 * Inputs are chosen features of (x, y): any of x, y, x², y², xy. The network has 0, 1 or 2
 * hidden layers (tanh or ReLU) and one output logit z; the probability of class B is the
 * logistic function σ(z) (the two-class case of softmax). Loss: mean cross-entropy plus an
 * optional L2 weight decay ½κ‖W‖². Gradients by hand-written backpropagation; Adam optimiser.
 */
import { rng } from './linalg.ts';

export type Feature = 'x' | 'y' | 'x2' | 'y2' | 'xy';
export const ALL_FEATURES: Feature[] = ['x', 'y', 'x2', 'y2', 'xy'];
export type Activation = 'tanh' | 'relu';
export type DatasetId = 'blobs' | 'moons' | 'circles' | 'xor' | 'spiral';

export interface Point { x: number; y: number; c: 0 | 1 }

export function features(p: { x: number; y: number }, fs: Feature[]): number[] {
  return fs.map((f) => (f === 'x' ? p.x : f === 'y' ? p.y : f === 'x2' ? p.x * p.x : f === 'y2' ? p.y * p.y : p.x * p.y));
}

/* ------------------------------------------------------------------ data */

/** Points in roughly [−1, 1]², class 0 = A, class 1 = B, with Gaussian jitter `noise`. */
export function makeDataset(id: DatasetId, n: number, noise: number, seed: number): Point[] {
  const R = rng(seed), out: Point[] = [];
  for (let k = 0; k < n; k++) {
    const c = (k % 2) as 0 | 1;
    let x = 0, y = 0;
    switch (id) {
      case 'blobs': x = c ? 0.45 : -0.45; y = c ? 0.35 : -0.35; x += 0.25 * R.normal(); y += 0.25 * R.normal(); break;
      case 'moons': {
        const t = Math.PI * R.uni();
        if (c === 0) { x = Math.cos(t) - 0.5; y = Math.sin(t) - 0.25; } else { x = 0.5 - Math.cos(t); y = 0.25 - Math.sin(t); }
        x *= 0.75; y *= 0.75; break;
      }
      case 'circles': { const t = 2 * Math.PI * R.uni(), r = c ? 0.8 : 0.35; x = r * Math.cos(t); y = r * Math.sin(t); break; }
      case 'xor': { x = (R.uni() * 2 - 1) * 0.9; y = (R.uni() * 2 - 1) * 0.9; const cc = (x * y > 0 ? 0 : 1) as 0 | 1; out.push({ x: x + noise * R.normal(), y: y + noise * R.normal(), c: cc }); continue; }
      case 'spiral': {
        const t = R.uni(), r = 0.1 + 0.85 * t, a = 2.6 * Math.PI * t + (c ? Math.PI : 0);
        x = r * Math.cos(a); y = r * Math.sin(a); break;
      }
    }
    out.push({ x: x + noise * R.normal(), y: y + noise * R.normal(), c });
  }
  return out;
}

/* ------------------------------------------------------------------ network */

export interface Layer { W: Float64Array; b: Float64Array; nin: number; nout: number }
export interface Net { layers: Layer[]; act: Activation; fs: Feature[]; m: Float64Array[]; v: Float64Array[]; t: number }

export function initNet(fs: Feature[], hidden: number[], act: Activation, seed: number): Net {
  const R = rng(seed), sizes = [fs.length, ...hidden, 1];
  const layers: Layer[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const nin = sizes[l], nout = sizes[l + 1];
    const sc = act === 'relu' && l < sizes.length - 2 ? Math.sqrt(2 / nin) : Math.sqrt(1 / nin);
    layers.push({ W: Float64Array.from({ length: nin * nout }, () => R.normal() * sc * 1.5), b: Float64Array.from({ length: nout }, () => 0.1 * R.normal()), nin, nout });
  }
  const ps = layers.flatMap((L) => [L.W, L.b]);
  return { layers, act, fs, m: ps.map((p) => new Float64Array(p.length)), v: ps.map((p) => new Float64Array(p.length)), t: 0 };
}

export const nParams = (net: Net) => net.layers.reduce((s, L) => s + L.W.length + L.b.length, 0);
const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
const actf = (a: Activation, z: number) => (a === 'tanh' ? Math.tanh(z) : z > 0 ? z : 0);
const dactf = (a: Activation, z: number, h: number) => (a === 'tanh' ? 1 - h * h : z > 0 ? 1 : 0);

/** Probability of class B at a point. */
export function prob(net: Net, p: { x: number; y: number }): number {
  let h = features(p, net.fs);
  const nl = net.layers.length;
  for (let l = 0; l < nl; l++) {
    const L = net.layers[l], o = new Array(L.nout);
    for (let j = 0; j < L.nout; j++) {
      let z = L.b[j];
      for (let i = 0; i < L.nin; i++) z += L.W[j * L.nin + i] * h[i];
      o[j] = l < nl - 1 ? actf(net.act, z) : z;
    }
    h = o;
  }
  return sigmoid(h[0]);
}

/** Mean cross-entropy (+ weight decay) and gradients in layer order [W0, b0, W1, b1, …]. */
export function lossGrad(net: Net, pts: Point[], decay = 0) {
  const nl = net.layers.length;
  const grads = net.layers.flatMap((L) => [new Float64Array(L.W.length), new Float64Array(L.b.length)]);
  let loss = 0;
  const N = pts.length || 1;
  for (const p of pts) {
    const hs: number[][] = [features(p, net.fs)], zs: number[][] = [];
    for (let l = 0; l < nl; l++) {
      const L = net.layers[l], h = hs[l], z = new Array(L.nout), o = new Array(L.nout);
      for (let j = 0; j < L.nout; j++) {
        let s = L.b[j];
        for (let i = 0; i < L.nin; i++) s += L.W[j * L.nin + i] * h[i];
        z[j] = s; o[j] = l < nl - 1 ? actf(net.act, s) : s;
      }
      zs.push(z); hs.push(o);
    }
    const zout = hs[nl][0], q = sigmoid(zout);
    // cross-entropy in a numerically stable form: log(1 + e^z) − c z
    loss += (zout > 0 ? zout + Math.log1p(Math.exp(-zout)) : Math.log1p(Math.exp(zout))) - p.c * zout;
    let delta = [(q - p.c) / N];
    for (let l = nl - 1; l >= 0; l--) {
      const L = net.layers[l], h = hs[l], gW = grads[2 * l], gb = grads[2 * l + 1];
      for (let j = 0; j < L.nout; j++) {
        gb[j] += delta[j];
        for (let i = 0; i < L.nin; i++) gW[j * L.nin + i] += delta[j] * h[i];
      }
      if (l > 0) {
        const prev = new Array(L.nin).fill(0);
        for (let i = 0; i < L.nin; i++) {
          let s = 0;
          for (let j = 0; j < L.nout; j++) s += L.W[j * L.nin + i] * delta[j];
          prev[i] = s * dactf(net.act, zs[l - 1][i], hs[l][i]);
        }
        delta = prev;
      }
    }
  }
  loss /= N;
  if (decay) net.layers.forEach((L, l) => {
    for (let k = 0; k < L.W.length; k++) { loss += 0.5 * decay * L.W[k] ** 2; grads[2 * l][k] += decay * L.W[k]; }
  });
  return { loss, grads };
}

export function adam(net: Net, grads: Float64Array[], lr = 0.02, b1 = 0.9, b2 = 0.999) {
  net.t++;
  const c1 = 1 - b1 ** net.t, c2 = 1 - b2 ** net.t;
  const ps = net.layers.flatMap((L) => [L.W, L.b]);
  ps.forEach((p, k) => {
    const g = grads[k], m = net.m[k], v = net.v[k];
    for (let i = 0; i < p.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
    }
  });
}

export function accuracy(net: Net, pts: Point[]): number {
  if (!pts.length) return NaN;
  let ok = 0;
  for (const p of pts) if ((prob(net, p) > 0.5 ? 1 : 0) === p.c) ok++;
  return ok / pts.length;
}

export function meanLoss(net: Net, pts: Point[]): number {
  if (!pts.length) return NaN;
  let s = 0;
  for (const p of pts) { const q = Math.min(1 - 1e-12, Math.max(1e-12, prob(net, p))); s -= p.c ? Math.log(q) : Math.log(1 - q); }
  return s / pts.length;
}
