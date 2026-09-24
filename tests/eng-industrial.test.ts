// Checks for the delivery-routing (TSP) tab: run with `node tests/eng-industrial.test.ts`
import { tourLength, nearestNeighbour, makeAnnealer, twoOptDelta, reverse, randomCustomers, type Pt } from '../src/lib/eng/tsp.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const isPerm = (t: number[], n: number) => t.length === n && new Set(t).size === n && t.every((v) => v >= 0 && v < n);

// 1. 2-opt delta equals the actual change in length
{
  const pts = randomCustomers(25, 3);
  const t = nearestNeighbour(pts);
  let maxErr = 0;
  for (const [i, j] of [[1, 5], [3, 20], [2, 24], [10, 11]]) {
    const before = tourLength(pts, t), dl = twoOptDelta(pts, t, i, j);
    const u = t.slice(); reverse(u, i, j);
    maxErr = Math.max(maxErr, Math.abs(tourLength(pts, u) - before - dl));
  }
  check('2-opt delta matches the tour length change', maxErr < 1e-9, `(max err ${maxErr.toExponential(1)})`);
}

// 2. Points on a circle: annealing finds the perimeter (the optimum)
{
  const n = 40, pts: Pt[] = Array.from({ length: n }, (_, k) => ({ x: 50 + 40 * Math.cos((2 * Math.PI * k * 17) / n), y: 50 + 40 * Math.sin((2 * Math.PI * k * 17) / n) }));
  const opt = 2 * n * 40 * Math.sin(Math.PI / n);
  const A = makeAnnealer(pts, Array.from({ length: n }, (_, i) => i), 2);
  while (!A.run(5000));
  check('annealing reaches the optimum on a circle', A.length < opt * 1.001 && isPerm(A.tour, n), `(${A.length.toFixed(2)} vs ${opt.toFixed(2)} km)`);
}

// 3. Random customers: annealing beats nearest neighbour and keeps a valid tour
{
  const pts = [{ x: 50, y: 50 }, ...randomCustomers(50, 7)];
  const nn = nearestNeighbour(pts), lnn = tourLength(pts, nn);
  const A = makeAnnealer(pts, nn, 1);
  while (!A.run(5000));
  check('annealing improves on nearest neighbour', A.length < 0.97 * lnn && isPerm(A.tour, pts.length), `(${A.length.toFixed(1)} vs ${lnn.toFixed(1)} km)`);
  check('tracked length equals recomputed length', Math.abs(A.length - tourLength(pts, A.tour)) < 1e-6);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
