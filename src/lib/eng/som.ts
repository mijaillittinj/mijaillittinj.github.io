/**
 * Neural counterpart for delivery routing: a self-organising map (Kohonen ring, in the
 * spirit of the Durbin–Willshaw elastic net). A closed ring of neurons is pulled towards
 * randomly chosen customers; the winner and its ring neighbours move, with a neighbourhood
 * and a learning rate that shrink over time. When training ends, each customer is assigned
 * to its nearest neuron and the ring order gives the route.
 * This is a neural network but not a physics-informed one: routing has no governing
 * equation to place in the loss.
 */
import { rng } from '../linalg.ts';
import type { Pt } from './tsp.ts';

export interface Som {
  /** Neuron positions in km, interleaved [x0, y0, x1, y1, ...]. */
  ring: Float64Array;
  m: number; iter: number; total: number; sigma: number; eta: number; done: boolean;
  /** Run k training steps; returns true when training has finished. */
  run(k: number): boolean;
  /** Current route: customers ordered along the ring, starting at the depot (index 0). */
  tour(): number[];
}

export function makeSom(pts: Pt[], seed = 1, neuronsPerCity = 2.5, total = Math.max(8000, 400 * pts.length)): Som {
  const R = rng(seed);
  const n = pts.length;
  const m = Math.max(8, Math.round(neuronsPerCity * n));
  let cx = 0, cy = 0, span = 1;
  for (const p of pts) { cx += p.x / n; cy += p.y / n; }
  for (const p of pts) span = Math.max(span, Math.abs(p.x - cx), Math.abs(p.y - cy));
  const ring = new Float64Array(2 * m);
  for (let j = 0; j < m; j++) {
    const a = (2 * Math.PI * j) / m;
    ring[2 * j] = cx + 0.15 * span * Math.cos(a);
    ring[2 * j + 1] = cy + 0.15 * span * Math.sin(a);
  }
  const sigma0 = m / 8, sigma1 = 0.6, eta0 = 0.8, eta1 = 0.02;

  const winner = (x: number, y: number) => {
    let best = 0, bd = Infinity;
    for (let j = 0; j < m; j++) {
      const d = (ring[2 * j] - x) ** 2 + (ring[2 * j + 1] - y) ** 2;
      if (d < bd) { bd = d; best = j; }
    }
    return best;
  };

  const S: Som = {
    ring, m, iter: 0, total, sigma: sigma0, eta: eta0, done: n < 3,
    run(k) {
      if (S.done) return true;
      for (let q = 0; q < k && S.iter < S.total; q++, S.iter++) {
        const t = S.iter / S.total;
        S.sigma = sigma0 * (sigma1 / sigma0) ** t;
        S.eta = eta0 * (eta1 / eta0) ** t;
        const p = pts[Math.floor(R.uni() * n)];
        const w = winner(p.x, p.y);
        const reach = Math.ceil(3 * S.sigma);
        for (let o = -reach; o <= reach; o++) {
          if (2 * Math.abs(o) >= m) continue;
          const j = (((w + o) % m) + m) % m;
          const h = S.eta * Math.exp(-(o * o) / (2 * S.sigma * S.sigma));
          ring[2 * j] += h * (p.x - ring[2 * j]);
          ring[2 * j + 1] += h * (p.y - ring[2 * j + 1]);
        }
      }
      if (S.iter >= S.total) S.done = true;
      return S.done;
    },
    tour() {
      const idx = pts.map((p, i) => ({ i, w: winner(p.x, p.y), a: Math.atan2(p.y - cy, p.x - cx) }));
      idx.sort((u, v) => u.w - v.w || u.a - v.a);
      const order = idx.map((u) => u.i);
      const s = order.indexOf(0);
      return s > 0 ? [...order.slice(s), ...order.slice(0, s)] : order;
    },
  };
  return S;
}
