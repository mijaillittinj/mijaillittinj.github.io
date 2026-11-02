// IR tail calibration of ψ_R with the oven (src/lib/workshop-rgb/tailcal.ts) against tail_calibration of
// manim/tools/calibration.py on the points of data/oven/oven_fit_points.csv (tailcal_ref.json, written by
// tools/make_web_fixtures.py --only tailcal). Tolerance 1e-6 relative on s_R, K_R/K_G and every K_j,
// 1e-4 K on the ΔT, declared in the fixture before running.
// run with `node --experimental-strip-types tests/workshop-rgb-tailcal.test.ts`
import { readFileSync } from 'node:fs';
import { buildPsi, CH } from '../src/lib/workshop-rgb/radiometry.ts';
import { DEFAULT_PRESET } from '../src/lib/workshop-rgb/presets.ts';
import { tailGrid, tailCalibration, tailPoints, scaleTailR, type TailPoint } from '../src/lib/workshop-rgb/tailcal.ts';
import type { FrameSignal } from '../src/lib/workshop-rgb/oven.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const ref = JSON.parse(readFileSync(new URL('./fixtures/workshop-rgb/tailcal_ref.json', import.meta.url), 'utf8'));
const TOL = ref.tol, TOLK = ref.tol_dT_K;
let worst = 0, worstK = 0;
const rel = (a: number, b: number) => { const e = Math.abs(a - b) / Math.max(Math.abs(b), 1e-300); if (e > worst) worst = e; return e <= TOL; };
const absK = (a: number, b: number) => { const e = Math.abs(a - b); if (e > worstK) worstK = e; return e <= TOLK; };
const C = ref.calibration;

const p = DEFAULT_PRESET.response;
const psi = buildPsi(p.lamNm, p.eta, { basis: 'photon' });
const g = tailGrid(p.lamNm, psi);
const pts: TailPoint[] = ref.points;
const t = tailCalibration(pts, g, { sigmaT: ref.params.sigmaT, Tflame: ref.params.Tflame });

check('four temperatures and the R/G pairs of calibration.py', t.ok && t.temps.join() === '800,900,1000,1050' && t.nPairs === C.n_pares && t.temps.every((T) => t.nPerT[T] === C.n_por_T[String(T)]), `(${t.nPairs} pairs)`);
check('s_R equals calibration.py at 1e-6', rel(t.s, C.s_R), `(s_R = ${t.s.toFixed(6)} against ${C.s_R.toFixed(6)})`);
check('K_R/K_G of the fit equals calibration.py at 1e-6', rel(t.KRKG, C.K_R_K_G), `(${t.KRKG.toFixed(6)})`);
check('rms of the fit equals calibration.py', rel(t.rmsPct, C.rms_pct));
check('K_j per temperature with the corrected ψ_R (band 5 to 90 %) equal calibration.py at 1e-6',
  CH.every((c) => t.temps.every((T) => rel(t.Kcorr[c].perT[T].K, C.K_por_T_banda_5_90[c][String(T)]))));
check('K_j geometric mean with the corrected ψ_R (band 50 to 70 %) and its spread equal calibration.py at 1e-6',
  CH.every((c) => rel(t.Kfit[c].K, C.K[c].K) && rel(100 * t.Kfit[c].spreadRel, C.K[c].dispersion_pct)), `(K_R ${(t.Kfit.R.K / 1e6).toFixed(4)}, K_G ${(t.Kfit.G.K / 1e6).toFixed(4)} × 10⁶)`);
check('K_R/K_G per temperature, corrected and nominal, equal calibration.py',
  t.temps.every((T) => rel(t.KRKGperT[T], C.KRKG_por_T[String(T)]) && rel(t.KRKGperTNominal[T], C.KRKG_por_T_nominal[String(T)])));
check('drift of K_R/K_G, corrected and nominal, and the spread equal calibration.py',
  rel(t.driftPct, C.deriva_pct) && rel(t.driftNominalPct, C.deriva_nominal_pct) && rel(t.spreadPct, C.desvio_max_pct) && CH.every((c) => rel(t.driftNominalByChannelPct[c], C.deriva_nominal_por_canal_pct[c])),
  `(${t.driftNominalPct.toFixed(1)} % nominal, ${t.driftPct.toFixed(2)} % corrected)`);
for (const l of t.loo) {
  const r = C.una_fuera[String(l.tC)];
  check(`${l.tC} °C left out, s, K_R/K_G, residuals and K_R/K_G of that temperature equal calibration.py`,
    rel(l.s, r.s) && rel(l.KRKG, r.K_R_K_G) && rel(l.KRKGout, r.KRKG_T_fuera) && Math.abs(l.residPct - r.resid_pct) < 1e-6 * 100 && Math.abs(l.residNominalPct - r.resid_nominal_pct) < 1e-4
      && Math.abs(l.KRKGrelPct - r.KRKG_rel_pct) < 1e-4 && Math.abs(l.KRKGrelNominalPct - r.KRKG_rel_nominal_pct) < 1e-4,
    `(${l.KRKGrelNominalPct.toFixed(1)} % nominal → ${l.KRKGrelPct.toFixed(2)} %)`);
  check(`${l.tC} °C left out, ΔT at 1800 K (fit of the others, and with K_R/K_G of that temperature) equal calibration.py at 1e-4 K`,
    absK(l.dTflameK, r.dT_llama_K) && absK(l.dToutK, r.dT_fuera_K), `(${l.dToutK.toFixed(2)} K)`);
}
for (const r of t.ratio) {
  const q = C.prueba_razon[String(r.tC)];
  check(`ratio test in the flame with K_R/K_G of ${r.tC} °C, before and after, equal calibration.py`,
    rel(r.KRKGrelTop, q.KRKG_rel_alta) && rel(r.KRKGrelTopNominal, q.KRKG_rel_alta_nominal) && absK(r.dTflameK, q.dT_llama_K) && absK(r.dTflameNominalK, q.dT_llama_nominal_K),
    `(${r.dTflameNominalK.toFixed(0)} → ${r.dTflameK.toFixed(1)} K)`);
}
const crit = Math.max(...Object.values(C.una_fuera as Record<string, { KRKG_rel_pct: number }>).map((v) => Math.abs(v.KRKG_rel_pct)));
check('acceptance criterion, worst K_R/K_G left out under 1 % (as checks.json oven_tailcal_loo_KRKG), 800 °C out is the strong test',
  t.criterion.pass && Math.abs(t.criterion.worstPct - crit) < 1e-4 && t.criterion.strongT === 800 && Math.abs(t.criterion.nominalPct - 17.86) < 0.01,
  `(${t.criterion.worstPct.toFixed(2)} %, nominal ${t.criterion.nominalPct.toFixed(1)} %, 800 °C out ${t.criterion.strongPct.toFixed(2)} %)`);
check('ratio test in the flame, worst out of sample after and nominal before', t.ratioTest.afterK < 5 && t.ratioTest.beforeK > 370, `(${t.ratioTest.beforeK.toFixed(0)} → ${t.ratioTest.afterK.toFixed(1)} K)`);
// deterministic uncertainty, own numpy code in the fixture
const D = ref.deterministic;
check('deterministic u(s_R) and its parts equal the numpy rewrite at 1e-6', rel(t.u.s, D.u_s) && rel(t.u.sStat, D.u_s_stat) && rel(t.u.sT, D.u_s_T) && rel(t.u.sCommon, D.d_s_common) && rel(t.u.lnc, D.u_lnc),
  `(u = ${t.u.s.toFixed(4)}, bootstrap of calibration.py ${ref.u_s_R_bootstrap.toFixed(4)})`);
check('deterministic u(T) at 1800 K by the parameter equals the numpy rewrite, and is within 15 % of the Monte Carlo of calibration.py',
  Math.abs(t.u.flameK - D.u_flame_K) < 1e-3 && Math.abs(t.u.dTdsK - D.dT_ds_K) < 1e-2 && Math.abs(t.u.flameK / ref.flame_u_param_K_bootstrap - 1) < 0.15, `(${t.u.flameK.toFixed(2)} K against ${ref.flame_u_param_K_bootstrap.toFixed(2)} K)`);
if (ref.calibration_py_unrounded) check('points written with one decimal give s_R within 1e-4 of calibration.py on unrounded y', Math.abs(t.s - ref.calibration_py_unrounded.s_R) < 1e-4, `(${ref.calibration_py_unrounded.s_R.toFixed(6)})`);

// three temperatures, the coldest one left out is the criterion (G5cola-adv-04)
const t3 = tailCalibration(pts.filter((q) => q.tC !== 1000), g, { sigmaT: 5 });
check('with three temperatures the criterion is the coldest one left out', t3.ok && t3.criterion.from.join() === '800' && Math.abs(t3.criterion.worstPct - Math.abs(t3.loo[0].KRKGrelPct)) < 1e-12, `(${t3.criterion.worstPct.toFixed(2)} %)`);
const t2 = tailCalibration(pts.filter((q) => q.tC >= 1000), g);
check('with two temperatures the tail is not calibrated', !t2.ok && t2.why === 'tail_few_temps');

// points from plateau signals, intercept dark as calibration.py
const sig = (tC: number, texpUs: number, S: number[], nSat = [0, 0, 0]): FrameSignal => ({ tC, texpUs, S: { R: S[0], G: S[1], B: S[2] }, p999: { R: S[0], G: S[1], B: S[2] }, nSat: { R: nSat[0], G: nSat[1], B: nSat[2] }, n: { R: 1, G: 1, B: 1 }, strip: { R: NaN, G: NaN, B: NaN }, stripSigma: { R: NaN, G: NaN, B: NaN }, nStrip: { R: 0, G: 0, B: 0 } });
const sigs = [sig(900, 1000, [500, 300, 100]), sig(900, 2000, [990, 590, 190]), sig(900, 3000, [1480, 880, 280]), sig(900, 9000, [4095, 2620, 820], [5, 0, 0])];
const tp = tailPoints(sigs);
const r1 = tp.find((q) => q.ch === 'R' && q.texpUs === 1000)!;
check('points from signals, intercept over 5 to 90 %, y = (S − D)/t_exp and the band flags', Math.abs(r1.y - (500 - 10) / 1e-3) < 1e-6 && r1.inLin && !r1.inFit && tp.find((q) => q.ch === 'R' && q.texpUs === 9000)!.saturated && !tp.find((q) => q.ch === 'B' && q.texpUs === 1000)!.inLin,
  `(D_R ${(500 - r1.y * 1e-3).toFixed(2)} DN)`);
const sc = scaleTailR(p.lamNm, psi, 0.5);
const i700 = Array.from(p.lamNm).indexOf(700);
check('scaleTailR scales ψ_R from 700 nm on and leaves G and B', sc.R[i700] === 0.5 * psi.R[i700] && sc.R[i700 - 1] === psi.R[i700 - 1] && sc.G[i700] === psi.G[i700]);

console.log(`worst relative difference ${worst.toExponential(2)}, worst ΔT difference ${worstK.toExponential(2)} K`);
console.log(fails ? `${fails} FAIL` : 'all PASS');
if (fails) process.exit(1);
