// Checks for the self-organising map route: run with `node tests/som.test.ts`
import { makeSom } from '../src/lib/eng/som.ts';
import { tourLength, nearestNeighbour, makeAnnealer, twoOptPolish, randomCustomers, type Pt } from '../src/lib/eng/tsp.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const isPerm = (t: number[], n: number) => t.length === n && new Set(t).size === n && t.every((v) => v >= 0 && v < n);

for (const seed of [11, 12, 13]) {
  const pts: Pt[] = [{ x: 50, y: 50 }, ...randomCustomers(40, seed)];
  const S = makeSom(pts, seed);
  while (!S.run(2000));
  const t = S.tour();
  const A = makeAnnealer(pts, nearestNeighbour(pts), seed);
  while (!A.run(5000));
  const som = tourLength(pts, t), sa = A.length;
  const pol = t.slice(); twoOptPolish(pts, pol);
  check(`SOM tour is a valid permutation starting at the depot (seed ${seed})`, isPerm(t, pts.length) && t[0] === 0);
  check(`SOM within 15 % of annealing (seed ${seed})`, som < 1.15 * sa, `(SOM ${som.toFixed(0)} km, +2-opt ${tourLength(pts, pol).toFixed(0)} km, annealing ${sa.toFixed(0)} km)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
