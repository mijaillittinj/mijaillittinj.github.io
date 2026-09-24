/**
 * Delivery routing as a travelling salesman problem (TSP), solved heuristically:
 * a nearest-neighbour route, then simulated annealing with 2-opt moves, then a
 * greedy 2-opt polish. City 0 is the depot. Tours are closed permutations.
 */
import { rng } from '../linalg.ts';

export interface Pt { x: number; y: number }

const d = (p: Pt, q: Pt) => Math.hypot(p.x - q.x, p.y - q.y);

export function tourLength(pts: Pt[], tour: ArrayLike<number>): number {
  let s = 0;
  for (let k = 0; k < tour.length; k++) s += d(pts[tour[k]], pts[tour[(k + 1) % tour.length]]);
  return s;
}

export function nearestNeighbour(pts: Pt[], start = 0): number[] {
  const n = pts.length;
  if (n === 0) return [];
  const used = new Uint8Array(n);
  const tour = [start];
  used[start] = 1;
  for (let k = 1; k < n; k++) {
    const last = pts[tour[tour.length - 1]];
    let bi = -1, bd = Infinity;
    for (let i = 0; i < n; i++) if (!used[i]) { const dd = d(last, pts[i]); if (dd < bd) { bd = dd; bi = i; } }
    tour.push(bi); used[bi] = 1;
  }
  return tour;
}

/** Length change when reversing tour[i..j] (1 ≤ i < j ≤ n − 1). */
export function twoOptDelta(pts: Pt[], t: ArrayLike<number>, i: number, j: number): number {
  const n = t.length;
  const a = pts[t[i - 1]], b = pts[t[i]], c = pts[t[j]], e = pts[t[(j + 1) % n]];
  return d(a, c) + d(b, e) - d(a, b) - d(c, e);
}

export function reverse(t: number[], i: number, j: number) {
  while (i < j) { const tmp = t[i]; t[i] = t[j]; t[j] = tmp; i++; j--; }
}

/** Greedy 2-opt until no improving move exists. Returns the number of moves applied. */
export function twoOptPolish(pts: Pt[], t: number[], maxPasses = 50): number {
  const n = t.length;
  let moves = 0;
  for (let pass = 0; pass < maxPasses; pass++) {
    let improved = false;
    for (let i = 1; i < n - 1; i++) for (let j = i + 1; j < n; j++) {
      if (twoOptDelta(pts, t, i, j) < -1e-10) { reverse(t, i, j); improved = true; moves++; }
    }
    if (!improved) break;
  }
  return moves;
}

export interface Annealer {
  tour: number[]; length: number; best: number[]; bestLength: number;
  T: number; T0: number; iter: number; total: number; done: boolean;
  /** Run k proposals; returns true when the schedule has finished (after the final polish). */
  run(k: number): boolean;
}

/**
 * Simulated annealing on 2-opt moves with geometric cooling from T0 to T0·1e-3
 * over `total` proposals. T0 is set to 0.2 × the mean |Δ| of random moves (warm, but not a random walk).
 */
export function makeAnnealer(pts: Pt[], start: number[], seed = 1, total = Math.max(20000, 2500 * pts.length)): Annealer {
  const R = rng(seed);
  const n = start.length;
  const tour = start.slice();
  let length = tourLength(pts, tour);
  let s = 0, c = 0;
  for (let k = 0; k < 200 && n > 3; k++) {
    const i = 1 + Math.floor(R.uni() * (n - 2));
    const j = i + 1 + Math.floor(R.uni() * (n - 1 - i));
    s += Math.abs(twoOptDelta(pts, tour, i, j)); c++;
  }
  const T0 = c ? (0.2 * s) / c : 1;
  const alpha = Math.pow(1e-3, 1 / total);
  const A: Annealer = {
    tour, length, best: tour.slice(), bestLength: length, T: T0, T0, iter: 0, total, done: n < 4,
    run(k) {
      if (A.done) return true;
      for (let q = 0; q < k && A.iter < A.total; q++, A.iter++) {
        const i = 1 + Math.floor(R.uni() * (n - 2));
        const j = i + 1 + Math.floor(R.uni() * (n - 1 - i));
        const dl = twoOptDelta(pts, A.tour, i, j);
        if (dl < 0 || R.uni() < Math.exp(-dl / A.T)) {
          reverse(A.tour, i, j);
          A.length += dl;
          if (A.length < A.bestLength - 1e-9) { A.bestLength = A.length; A.best = A.tour.slice(); }
        }
        A.T *= alpha;
      }
      if (A.iter >= A.total) {
        A.tour = A.best.slice();
        twoOptPolish(pts, A.tour);
        A.length = A.bestLength = tourLength(pts, A.tour);
        A.best = A.tour.slice();
        A.done = true;
      }
      return A.done;
    },
  };
  return A;
}

export function randomCustomers(k: number, seed: number, L = 100): Pt[] {
  const R = rng(seed);
  // a few demand clusters plus scattered customers, like a real service area
  const centres = Array.from({ length: 4 }, () => ({ x: 0.15 * L + 0.7 * L * R.uni(), y: 0.15 * L + 0.7 * L * R.uni() }));
  const out: Pt[] = [];
  for (let i = 0; i < k; i++) {
    if (R.uni() < 0.65) {
      const c = centres[Math.floor(R.uni() * centres.length)];
      out.push({ x: clamp(c.x + 0.09 * L * R.normal(), L), y: clamp(c.y + 0.09 * L * R.normal(), L) });
    } else out.push({ x: 0.04 * L + 0.92 * L * R.uni(), y: 0.04 * L + 0.92 * L * R.uni() });
  }
  return out;
}
const clamp = (v: number, L: number) => Math.max(0.03 * L, Math.min(0.97 * L, v));
