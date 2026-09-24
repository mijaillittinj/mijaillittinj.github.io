// Checks for the SVD learning module: run with `node tests/learn-svd.test.ts`
import { svd, uTg, tsvd, tikhonovSVD, lcurve, blurOperator, testSignal } from '../src/lib/svd.ts';
import { matvec, rng } from '../src/lib/linalg.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const n = 48;
const A = blurOperator(n, 0.05);
const d = svd(A);

// 1. U Σ Vᵀ reproduces A
{
  let err = 0, nrm = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < n; k++) s += d.U[i * n + k] * d.s[k] * d.V[j * n + k];
    err = Math.max(err, Math.abs(s - A.a[i * n + j])); nrm = Math.max(nrm, Math.abs(A.a[i * n + j]));
  }
  check('U Σ Vᵀ = A', err < 1e-12 * nrm * n, `(max err ${err.toExponential(1)})`);
}
// 2. V orthogonal and singular values sorted, spanning many decades
{
  let e = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = 0; for (let k = 0; k < n; k++) s += d.V[k * n + i] * d.V[k * n + j];
    e = Math.max(e, Math.abs(s - (i === j ? 1 : 0)));
  }
  const sorted = d.s.every((v, i) => i === 0 || v <= d.s[i - 1]);
  check('V orthogonal, σ sorted', e < 1e-10 && sorted, `(orth err ${e.toExponential(1)})`);
  check('blur is severely ill-conditioned', d.s[0] / d.s[n - 1] > 1e10, `(cond ${(d.s[0] / d.s[n - 1]).toExponential(1)})`);
}
// 3. Full TSVD on noise-free data reproduces the signal; with noise, a truncated one beats the full one
{
  const f = testSignal(n), g = matvec(A, f);
  const k0 = d.s.findIndex((s) => s < 1e-10 * d.s[0]);
  const xk = tsvd(d, uTg(d, g), k0 < 0 ? n : k0);
  const rel = (x: ArrayLike<number>) => { let e = 0, m = 0; for (let i = 0; i < n; i++) { e += (x[i] - f[i]) ** 2; m += f[i] ** 2; } return Math.sqrt(e / m); };
  check('TSVD recovers noise-free signal (well-resolved part)', rel(xk) < 0.35, `(rel err ${rel(xk).toFixed(3)})`);
  const R = rng(3), gn = Float64Array.from(g, (v) => v + 0.01 * R.normal());
  const b = uTg(d, gn);
  const full = rel(tsvd(d, b, n)), best = Math.min(...Array.from({ length: n }, (_, k) => rel(tsvd(d, b, k + 1))));
  check('with noise, truncation beats full inversion', best < 0.1 * full, `(best ${best.toFixed(2)}, full ${full.toExponential(1)})`);
  const lams = [1e-6, 1e-4, 1e-2];
  const L = lcurve(d, b, Array.from(gn).reduce((s, v) => s + v * v, 0), lams);
  const x2 = tikhonovSVD(d, b, 1e-4);
  const ax = matvec(A, x2); let r = 0; for (let i = 0; i < n; i++) r += (ax[i] - gn[i]) ** 2;
  check('L-curve residual matches direct computation', Math.abs(Math.sqrt(r) - L.res[1]) < 1e-9, '');
  check('L-curve monotone', L.res[0] <= L.res[2] && L.sol[0] >= L.sol[2], '');
}
if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
