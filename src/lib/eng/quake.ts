/**
 * Locating an earthquake from P-wave arrival times (illustrative, homogeneous medium).
 *
 * Forward model: t_i = t0 + ‖x_i − x_s‖ / v.
 * Inverse problem: find (x_s, t0) from noisy t_i. For a trial epicentre the best
 * origin time is the mean residual, so the misfit depends on (x, y) only and can be
 * mapped over the whole region.
 */
import { rng } from '../linalg.ts';

export interface Pt { x: number; y: number }

export const V_P = 6; // km/s, typical crustal P-wave speed

export function arrivalTimes(stations: Pt[], src: Pt, t0 = 0, v = V_P): number[] {
  return stations.map((s) => t0 + Math.hypot(s.x - src.x, s.y - src.y) / v);
}

/** Misfit Σ (r_i − r̄)² with r_i = t_i − d_i / v, and the origin time r̄ that minimises it. */
export function misfitAt(stations: Pt[], times: number[], p: Pt, v = V_P): { misfit: number; t0: number } {
  const n = stations.length;
  const r = new Float64Array(n);
  let mean = 0;
  for (let i = 0; i < n; i++) { r[i] = times[i] - Math.hypot(stations[i].x - p.x, stations[i].y - p.y) / v; mean += r[i]; }
  mean /= n || 1;
  let s = 0;
  for (let i = 0; i < n; i++) s += (r[i] - mean) ** 2;
  return { misfit: s, t0: mean };
}

export interface MisfitMap {
  n: number; L: number;
  /** row-major, row 0 at y = 0 (bottom) */
  data: Float32Array;
  best: Pt; t0: number; min: number;
}

/** Misfit on an n × n grid of cell centres over [0, L]², then refined locally around the minimum. */
export function misfitMap(stations: Pt[], times: number[], n = 100, L = 200, v = V_P): MisfitMap {
  const data = new Float32Array(n * n);
  const h = L / n;
  let min = Infinity, bi = 0, bj = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const m = misfitAt(stations, times, { x: (i + 0.5) * h, y: (j + 0.5) * h }, v).misfit;
    data[j * n + i] = m;
    if (m < min) { min = m; bi = i; bj = j; }
  }
  // local refinement: progressively finer patterns around the grid minimum
  let best = { x: (bi + 0.5) * h, y: (bj + 0.5) * h };
  for (let step = h; step > 0.01; step /= 2) {
    let improved = true;
    while (improved) {
      improved = false;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const p = { x: best.x + dx * step, y: best.y + dy * step };
        const m = misfitAt(stations, times, p, v).misfit;
        if (m < min - 1e-15) { min = m; best = p; improved = true; }
      }
    }
  }
  return { n, L, data, best, t0: misfitAt(stations, times, best, v).t0, min };
}

/**
 * Misfit level bounding an approximate 95 % confidence region for the epicentre:
 * min + Δχ² σ² with Δχ² = 5.99 (two parameters). σ has a floor so the region stays visible.
 */
export function confidenceLevel(min: number, sigma: number): number {
  const s = Math.max(sigma, 0.03);
  return min + 5.99 * s * s;
}

/** Area (km²) of the grid cells below a misfit level. */
export function regionArea(map: MisfitMap, level: number): number {
  const h = map.L / map.n;
  let c = 0;
  for (const v of map.data) if (v <= level) c++;
  return c * h * h;
}

/** Hidden earthquake with noisy arrivals. Noise draws are fixed per station so moving a station keeps its draw. */
export function hiddenQuake(seed: number, L = 200) {
  const R = rng(seed);
  const src = { x: 0.15 * L + 0.7 * L * R.uni(), y: 0.15 * L + 0.7 * L * R.uni() };
  const t0 = 2 + 4 * R.uni();
  const draws = Array.from({ length: 16 }, () => R.normal());
  return { src, t0, draws };
}

/**
 * Isochrone segments: marching squares on a scalar grid (row 0 at the bottom),
 * returning line segments in grid-cell-centre coordinates for the level set f = level.
 */
export function contourSegments(data: ArrayLike<number>, n: number, level: number): [number, number, number, number][] {
  const segs: [number, number, number, number][] = [];
  const v = (i: number, j: number) => data[j * n + i] - level;
  const lerp = (a: number, b: number) => (a === b ? 0.5 : a / (a - b));
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const a = v(i, j), b = v(i + 1, j), c = v(i + 1, j + 1), d = v(i, j + 1);
    const k = (a < 0 ? 1 : 0) | (b < 0 ? 2 : 0) | (c < 0 ? 4 : 0) | (d < 0 ? 8 : 0);
    if (k === 0 || k === 15) continue;
    // edge points: bottom (a-b), right (b-c), top (d-c), left (a-d)
    const B: [number, number] = [i + lerp(a, b), j];
    const Rr: [number, number] = [i + 1, j + lerp(b, c)];
    const T: [number, number] = [i + lerp(d, c), j + 1];
    const Lf: [number, number] = [i, j + lerp(a, d)];
    const add = (p: [number, number], q: [number, number]) => segs.push([p[0], p[1], q[0], q[1]]);
    switch (k) {
      case 1: case 14: add(Lf, B); break;
      case 2: case 13: add(B, Rr); break;
      case 3: case 12: add(Lf, Rr); break;
      case 4: case 11: add(Rr, T); break;
      case 6: case 9: add(B, T); break;
      case 7: case 8: add(Lf, T); break;
      case 5: add(Lf, B); add(Rr, T); break;
      case 10: add(B, Rr); add(Lf, T); break;
    }
  }
  return segs;
}
