// Checks for the SAT flame tomography tab: run with `node tests/sat.test.ts`
import { trappingMatrix, splineBasis, satSolve, sootEmission } from '../src/lib/sat.ts';
import { chordMatrix } from '../src/lib/abel.ts';
import { matvec, rng } from '../src/lib/linalg.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const n = 80;
const r = Float64Array.from({ length: n }, (_, i) => (i + 0.5) / n);
// off-axis soot ring (as low in a coflow flame) and its temperature
const fs = Float64Array.from(r, (x) => Math.exp(-(((x - 0.55) / 0.12) ** 2)) + 0.3 * Math.exp(-((x / 0.3) ** 2)));
const T = Float64Array.from(r, (x) => 1700 + 200 * Math.exp(-(((x - 0.6) / 0.25) ** 2)));
const J = Float64Array.from(fs, (f, i) => sootEmission(f, T[i]));
const Jm = Math.max(...J);
for (let i = 0; i < n; i++) J[i] /= Jm;
const err = (a: ArrayLike<number>) => { let e = 0; for (let i = 0; i < n; i++) e += (a[i] - J[i]) ** 2; return Math.sqrt(e / n) / Math.max(...J); };

// 1. without absorption the trapping operator is the Abel (chord) matrix
{
  const A = chordMatrix(n, 1), Tm = trappingMatrix(n, new Float64Array(n));
  let d = 0;
  for (let k = 0; k < A.a.length; k++) d = Math.max(d, Math.abs(A.a[k] - Tm.a[k]));
  check('trapping operator reduces to Abel when κ = 0', d < 1e-12, `(max diff ${d.toExponential(1)})`);
}

// 2. SAT recovers the emission profile from noisy, trapped data only when trapping is modelled
{
  const kappa = Float64Array.from(fs, (f) => 1.2 * f); // optically thick enough to matter
  const Tm = trappingMatrix(n, kappa);
  const g0 = matvec(Tm, J), gmax = Math.max(...g0), R = rng(3);
  const g = Float64Array.from(g0, (v) => v + 0.01 * gmax * R.normal());
  const B = splineBasis(n, 14);
  const eCorr = err(satSolve(Tm, g, B, 1e-4));
  const eIgn = err(satSolve(chordMatrix(n, 1), g, B, 1e-4));
  check('SAT with trapping correction recovers J (RMS < 4 % of peak)', eCorr < 0.04, `(${(eCorr * 100).toFixed(1)} %)`);
  check('ignoring trapping biases the result', eIgn > 2 * eCorr, `(${(eIgn * 100).toFixed(1)} %)`);
}

// 3. the spline basis is a partition of unity (sum of all basis functions = 1)
{
  const B = splineBasis(n, 10);
  let d = 0;
  for (let i = 0; i < n; i++) { let s = 0; for (let q = 0; q < B.m; q++) s += B.a[i * B.m + q]; d = Math.max(d, Math.abs(s - 1)); }
  check('spline basis sums to one', d < 1e-9, `(${d.toExponential(1)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
