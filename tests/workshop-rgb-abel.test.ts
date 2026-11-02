// Abel inversion of the workshop tool: chordMatrixHalf against annabel_app op_matrix, onion peeling and
// Tikhonov against an analytic profile and prepare_data.py, PyAbel (dasch onion_peeling), the reusable
// factorisations against tikhonov() and satSolve(), the L-curve corner, SAT with κ = 0 in metres against a
// numpy mirror and pysat-main: run with `node --experimental-strip-types tests/workshop-rgb-abel.test.ts`
// Tolerances that are not parity were written into abel_ref.json by make_web_fixtures.py before running.
import { readFileSync } from 'node:fs';
import { chordMatrix, chordMatrixHalf, tikhonov, tikhonovFactor, tikhonovApply, onionPeel, project, profile } from '../src/lib/abel.ts';
import { satSolve, satFactor, satApply, splineBasis } from '../src/lib/sat.ts';
import { cholFactor, cholSolveFactored, cholSolve, gram, rng } from '../src/lib/linalg.ts';
import { invertAll, lcurveCorner, splitHalves, overRegularised, LAMBDA_REL_MIN, LAMBDA_REL_MAX } from '../src/lib/workshop-rgb/invert.ts';
import { interp1 } from '../src/lib/workshop-rgb/stats.ts';
import { PER_MM_TO_PER_M } from '../src/lib/workshop-rgb/radiometry.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const ref = JSON.parse(readFileSync(fx('abel_ref.json'), 'utf8'));
const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>, lo = 0, hi = b.length) => { let m = 0; for (let i = lo; i < hi; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const amax = (a: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };

// 1. chordMatrixHalf is op_matrix of annabel_app (rays at (i + ½) dr)
{
  const { n, dr, A } = ref.op_matrix;
  const L = chordMatrixHalf(n, dr);
  let m = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) m = Math.max(m, Math.abs(L.a[i * n + j] - A[i][j]) / amax(A.flat()));
  check('chordMatrixHalf equals op_matrix (1e-12)', m < 1e-12, `(max ${m.toExponential(1)})`);
  const L0 = chordMatrix(5, 1);
  check('chordMatrix (rays at i·dr) is unchanged', Math.abs(L0.a[0] - 0.4) < 1e-15 && Math.abs(L0.a[1] - 2 * (Math.sqrt(0.16) - Math.sqrt(0.04)) * 1) < 1e-15);
}

// 2. onion peeling against the analytic profile and against PyAbel Dasch onion peeling
{
  const r = ref.analytic;
  const L = chordMatrixHalf(r.N, r.dr);
  const f = onionPeel(L, r.P);
  const fm = amax(r.f_true);
  check('onion equals the numpy back substitution of prepare_data (1e-9)', maxAbs(f, r.f_onion) / fm < 1e-9);
  const e = maxAbs(f, r.f_true) / fm;
  check(`onion against the analytic profile within the declared ${100 * r.tol_onion_vs_true} %`, e <= r.tol_onion_vs_true, `(${(100 * e).toFixed(3)} %)`);
  const rc = Array.from({ length: r.N }, (_, i) => (i + 0.5) * r.dr);
  let m = 0;
  for (let i = r.pyabel_range[0]; i < r.pyabel_range[1]; i++) m = Math.max(m, Math.abs(interp1(i * r.dr, rc, f) - r.f_pyabel_dasch[i]));
  check(`own onion (rays at (i + ½) dr) against PyAbel ${r.pyabel.split(',')[0]} within ${100 * r.tol_pyabel} %`, m / fm <= r.tol_pyabel, `(${(100 * m / fm).toFixed(3)} %)`);
  // GB2r1-numerico-05: the tolerance must catch the half pixel ray convention error
  const fw = onionPeel(chordMatrix(r.N, r.N * r.dr), r.P);
  const ew = maxAbs(fw, r.f_true) / fm;
  check('negative test: chordMatrix (rays at i·dr) on these P fails the declared tolerance', ew > r.tol_onion_vs_true, `(${(100 * ew).toFixed(2)} % against ${100 * r.tol_onion_vs_true} %)`);
  let es = 0;
  for (let i = 0; i < r.N; i++) es = Math.max(es, Math.abs(f[i] - Math.exp(-((i * r.dr) ** 2)) * (1 + 2 * (i * r.dr) ** 2)));
  check('negative test: the right solution read at a radius shifted half a pixel fails it too', es / fm > r.tol_onion_vs_true, `(${(100 * es / fm).toFixed(2)} %)`);
  const P2 = project(L, r.f_true);
  check('forward projection L f against the exact P', maxAbs(P2, r.P) / amax(r.P) < 5e-3, `(${(100 * maxAbs(P2, r.P) / amax(r.P)).toFixed(3)} %)`);
}

// 3. reusable factorisations equal the existing solvers
{
  const n = 50;
  const L = chordMatrix(n);
  const f = Float64Array.from({ length: n }, (_, i) => profile('annular', (i + 0.5) / n, 0.5, 0.4));
  const g = rng(3);
  const p = Float64Array.from(project(L, f), (v) => v + 0.01 * g.normal());
  for (const lr of [1e-5, 1e-3, 1e-1]) {
    const a = tikhonov(L, p, lr), b = tikhonovApply(tikhonovFactor(L, lr), L, p);
    check(`tikhonovApply(tikhonovFactor) equals tikhonov at λ = ${lr} (1e-12)`, maxAbs(a, b) / amax(a) < 1e-12);
  }
  const S = gram(L);
  for (let i = 0; i < n; i++) S.a[i * n + i] += 1;
  const x1 = cholSolve(S, p)!, x2 = cholSolveFactored(cholFactor(S)!, n, p);
  check('cholSolveFactored(cholFactor) equals cholSolve', maxAbs(x1, x2) === 0);
  const A = chordMatrixHalf(n, 1 / n), B = splineBasis(n, 10);
  const s1 = satSolve(A, p, B, 1e-3), s2 = satApply(satFactor(A, B, 1e-3), p);
  check('satApply(satFactor) equals satSolve (1e-12)', maxAbs(s1, s2) / amax(s1) < 1e-12);
}

// 4. Tikhonov with Γ of the presentation (prepare_data.tikhonov_op) and the L-curve corner
{
  const r = ref.tikhonov;
  const L = chordMatrixHalf(r.N, r.dr);
  for (const s of r.solutions) {
    const F = tikhonovFactor(L, s.lam_rel, 'gamma2');
    const f = tikhonovApply(F, L, r.P_noisy);
    check(`Tikhonov Γ = gamma2 at λ_rel = ${s.lam_rel.toExponential(2)} equals tikhonov_op (α and f, 1e-9)`, Math.abs(F.lam / s.alpha - 1) < 1e-12 && maxAbs(f, s.f) / amax(s.f) < 1e-9);
  }
  const c = lcurveCorner(L, [Float64Array.from(r.P_noisy)], r.lambdas, 'gamma2');
  // GB2r1-numerico-12: accuracy against the true profile with 1 % noise, declared before running
  const fc = tikhonovApply(tikhonovFactor(L, r.corner, 'gamma2'), L, r.P_noisy);
  const et = maxAbs(fc, r.f_true) / amax(r.f_true);
  check(`Tikhonov at the L-curve corner, 1 % noise, against the true profile within ${(100 * r.tol_noisy_vs_true).toFixed(2)} %`, et <= r.tol_noisy_vs_true, `(${(100 * et).toFixed(2)} %)`);
  const sat = invertAll([[Float64Array.from(r.P_noisy)]], r.dr, 'sat', { segments: r.sat_noisy.segments, satLambda: r.sat_noisy.lam }).J[0][0];
  const esat = maxAbs(sat, r.f_true) / amax(r.f_true);
  check(`SAT (${r.sat_noisy.segments} segments, λ ${r.sat_noisy.lam}), 1 % noise, against the true profile within ${(100 * r.tol_noisy_vs_true).toFixed(2)} %`, esat <= r.tol_noisy_vs_true, `(${(100 * esat).toFixed(2)} %)`);
  check('L-curve corner equals lcurve_corner (same λ, ρ and η)', c.index === r.corner_index && c.lambda === r.corner && maxAbs(c.rho, r.rho) / amax(r.rho) < 1e-9 && maxAbs(c.eta, r.eta) / amax(r.eta) < 1e-9,
    `(λ_rel = ${c.lambda.toExponential(2)})`);
}

// 5. all rows and channels with one factor, reprojection and residual, mm to m
{
  const r = ref.analytic;
  const n = 60, dr = 0.05; // mm
  const L = chordMatrixHalf(n, dr);
  const f = Float64Array.from({ length: n }, (_, i) => Math.exp(-(((i + 0.5) * dr) ** 2)));
  const P = [0, 1, 2].map((c) => [1, 2, 3].map((k) => Float64Array.from(project(L, f), (v) => v * (c + 1) * k)));
  const inv = invertAll(P, dr, 'tikhonov', { lambdaRel: 1e-6 });
  check('every row and channel inverted, residual = P − L f', inv.J.length === 3 && inv.J[2].length === 3 && maxAbs(inv.resid[1][2], P[1][2].map((v, i) => v - inv.reproj[1][2][i])) === 0);
  const invM = invertAll(P, dr * 1e-3, 'tikhonov', { lambdaRel: 1e-6 });
  check('J per mm × 1e3 equals J per m (ANNAbel mm to SI)', maxAbs(inv.J[0][0].map((v) => v * PER_MM_TO_PER_M), invM.J[0][0]) / amax(invM.J[0][0]) < 1e-9);
  const on = invertAll(P, dr, 'onion');
  check('onion of every row reproduces P exactly without noise', amax(on.resid[2][2]) / amax(P[2][2]) < 1e-12);
  // GB2r1-numerico-08: limits of λ_rel. λ → 0 is onion peeling, λ → ∞ goes to 0 with Γ = gamma2 and to the
  // least squares constant with the site regulariser, so the tool keeps λ_rel on the L-curve grid
  const Pr = P[1][1];
  const f0 = tikhonovApply(tikhonovFactor(L, 1e-14, 'gamma2'), L, Pr), fo = onionPeel(L, Pr);
  check('λ_rel → 0 equals onion peeling', maxAbs(f0, fo) / amax(fo) < 1e-6, `(${(maxAbs(f0, fo) / amax(fo)).toExponential(1)})`);
  const fInf = tikhonovApply(tikhonovFactor(L, 1e14, 'gamma2'), L, Pr);
  check('λ_rel → ∞ with gamma2 drives j to zero', amax(fInf) / amax(fo) < 1e-5, `(${(amax(fInf) / amax(fo)).toExponential(1)})`);
  const fS = tikhonovApply(tikhonovFactor(L, 1e14, 'site'), L, Pr);
  const one = new Float64Array(n).fill(1), L1 = project(L, one);
  let num = 0, den = 0; for (let i = 0; i < n; i++) { num += L1[i] * Pr[i]; den += L1[i] * L1[i]; }
  check('λ_rel → ∞ with the site regulariser tends to the least squares constant', maxAbs(fS, one.map(() => num / den)) / (num / den) < 1e-3);
  let thrown = '';
  try { invertAll(P, dr, 'tikhonov', { lambdaRel: 1e6 }); } catch (e) { thrown = (e as Error).message; }
  check(`invertAll refuses λ_rel outside [${LAMBDA_REL_MIN}, ${LAMBDA_REL_MAX}]`, thrown === 'lambda_out_of_range');
  const g3 = rng(11);
  const sig = 0.01 * amax(Pr);
  const Pn = Float64Array.from(Pr, (v) => v + sig * g3.normal());
  const hi = invertAll([[Pn]], dr, 'tikhonov', { lambdaRel: LAMBDA_REL_MAX }), lo = invertAll([[Pn]], dr, 'tikhonov', { lambdaRel: 1e-3 });
  const fx8 = tikhonovApply(tikhonovFactor(L, 1e8, 'gamma2'), L, Pn), r8 = project(L, fx8).map((v, i) => Pn[i] - v);
  check('over_regularised warns at λ_rel = 1e8 (j pulled to zero) and not at 1e-3 or at the top of the slider (1e3)', overRegularised(r8, sig) && !overRegularised(lo.resid[0][0], sig) && !overRegularised(hi.resid[0][0], sig));
  check('Tikhonov records its regulariser (gamma2) for run.json', inv.reg === 'gamma2' && on.reg === null);
  // GB2r1-numerico-07: a saturated ray leaves its row (all channels) and nothing else changes
  const Ps = P.map((ch) => ch.map((p) => Float64Array.from(p)));
  for (const ch of Ps) ch[1][40] *= 0.8;
  const ref0 = invertAll(P, dr, 'onion'), ex = invertAll(Ps, dr, 'onion', { excludeRows: [1] });
  check('excludeRows leaves that height as NaN in every channel and the others identical', ex.J.every((ch) => ch[1].every(Number.isNaN)) && ex.J.every((ch, c) => maxAbs(ch[0], ref0.J[c][0]) === 0 && maxAbs(ch[2], ref0.J[c][2]) === 0) && ex.excluded.join() === '1');
  const drop = new Uint8Array(n); drop[40] = 1;
  const fTrue = Float64Array.from({ length: n }, (_, i) => Math.exp(-(((i + 0.5) * dr) ** 2)) * 2 * 2);
  const satBad = invertAll(Ps, dr, 'sat', { satLambda: 1e-6 }).J[1][1], satMask = invertAll(Ps, dr, 'sat', { satLambda: 1e-6, rayMask: new Map([[1, drop]]) }).J[1][1];
  const eb = maxAbs(satBad, fTrue) / amax(fTrue), em = maxAbs(satMask, fTrue) / amax(fTrue);
  check('SAT with the saturated ray out of the least squares recovers the row', em < 0.01 && em < eb / 5, `(${(100 * em).toFixed(2)} % against ${(100 * eb).toFixed(1)} % with the ray in)`);
  const h = splitHalves([4, 3, 2, 1, 1, 2, 3, 5]);
  check('halves split with the axis between columns', h.right.join(',') === '1,2,3,5' && h.left.join(',') === '1,2,3,4');
  void r;
}

// 6. SAT with κ = 0 in metres: numpy mirror (parity) and pysat-main (contrast with declared tolerances)
{
  const r = ref.sat;
  const inv = invertAll([[Float64Array.from(r.P)]], r.dr_m, 'sat', { segments: r.segments, satLambda: r.lam });
  const f = inv.J[0][0];
  const fm = amax(r.f_true);
  const hi = r.range[1];
  check(`SAT equals the numpy mirror of sat.ts (${r.tol_numpy})`, maxAbs(f, r.f_numpy) / fm <= r.tol_numpy, `(${(maxAbs(f, r.f_numpy) / fm).toExponential(1)})`);
  const eT = maxAbs(f, r.f_true, 0, hi) / fm, eP = maxAbs(r.pysat.f, r.f_true, 0, hi) / fm, eTP = maxAbs(f, r.pysat.f, 0, hi) / fm;
  check(`SAT (site, metres) against the analytic profile within ${100 * r.tol_vs_true} %`, eT <= r.tol_vs_true, `(${(100 * eT).toFixed(2)} %)`);
  check(`pysat-main get_optimized_spline against the analytic profile within ${100 * r.tol_vs_true} %`, eP <= r.tol_vs_true, `(${(100 * eP).toFixed(2)} %)`);
  check(`SAT (site) against pysat-main within ${100 * r.tol_vs_pysat} %`, eTP <= r.tol_vs_pysat, `(${(100 * eTP).toFixed(2)} %)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
