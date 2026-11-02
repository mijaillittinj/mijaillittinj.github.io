// ψ_j, B_j(T), G_j(T), ratio table and its inverse against manim/tools/prepare_data.py (Band) and
// radiometric_calibration.py (kernel_table, band_integral, load_responsivity):
// run with `node --experimental-strip-types tests/workshop-rgb-radiometry.test.ts`
import { readFileSync } from 'node:fs';
import {
  makeBand, bandX, bandG, buildPsi, ratioTable, invertRatio, isMonotonic, tailFraction, emValue, irCut, shortPass, SP700,
  kernelTableTrapz, bandIntegral, planckNmRC, parseResponseCsv, gaussLegendre, scaleTail, withEmLaw, CH,
} from '../src/lib/workshop-rgb/radiometry.ts';
import { DEFAULT_PRESET, PRESETS, presetFor, presetPsi } from '../src/lib/workshop-rgb/presets.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const ref = JSON.parse(readFileSync(fx('radiometry.json'), 'utf8'));
const maxRel = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0; for (let i = 0; i < b.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]) / Math.max(Math.abs(b[i]), 1e-300)); return m; };
const maxAbsRel = (a: ArrayLike<number>, b: ArrayLike<number>) => { let m = 0, s = 0; for (let i = 0; i < b.length; i++) { m = Math.max(m, Math.abs(a[i] - b[i])); s = Math.max(s, Math.abs(b[i])); } return m / s; };
const TOL = 1e-6; // WEB_SPEC §14.1

const imx = DEFAULT_PRESET.response;
const psi = buildPsi(imx.lamNm, imx.eta, { basis: 'photon' });
const psiE = buildPsi(imx.lamNm, imx.eta, { basis: 'energy' });
const T: number[] = ref.T;

// 0. Gauss-Legendre nodes
{
  const g = gaussLegendre(6);
  const s = g.w.reduce((a, b) => a + b, 0);
  let q = 0; for (let k = 0; k < 6; k++) q += g.w[k] * g.x[k] ** 10;
  check('6 node Gauss-Legendre integrates x¹⁰ exactly', Math.abs(s - 2) < 1e-14 && Math.abs(q - 2 / 11) < 1e-14);
}
// 1. ψ_j per photon of the default preset
{
  const probe = [0, 100, 250, 400];
  check('ψ_j = (λ/550) η_j of the IMX392 preset', CH.every((c) => maxRel(probe.map((i) => psi[c][i]), ref.psi_photon_probe[c]) < 1e-15));
  check('default preset is the a2A1920 with no extra IR cut', DEFAULT_PRESET.id === 'a2A1920-160ucPRO' && DEFAULT_PRESET.irCut === null && !DEFAULT_PRESET.allowRemoveIr);
}
// 2. B_j, G_j and dG_j/dT with Gauss-Legendre, Riemann and per energy, 50 temperatures
for (const [nm, p, mode] of [['gl', psi, 'gl'], ['riemann', psi, 'riemann'], ['gl_energy', psiE, 'gl']] as const) {
  const b = makeBand(imx.lamNm, p, mode);
  let m = 0;
  for (const c of CH) {
    m = Math.max(m, maxRel(T.map((t) => bandX(b, c, t)), ref[nm][c].X), maxRel(T.map((t) => bandG(b, c, t)), ref[nm][c].G), maxRel(T.map((t) => bandG(b, c, t, true)), ref[nm][c].dG));
  }
  check(`B_j, G_j, dG_j/dT (${nm}) equal prepare_data.Band`, m < TOL, `(max rel ${m.toExponential(1)})`);
}
const band = makeBand(imx.lamNm, psi, 'gl');
// 3. ratio table on the same 50 temperatures
{
  const rt = ratioTable(band, T);
  const m = Math.max(maxRel(rt.RG, ref.RG), maxRel(rt.BG, ref.BG), maxRel(rt.BR, ref.BR), maxRel(rt.dlnRG, ref.dlnRG));
  check('R/G, B/G, B/R and d ln(R/G)/dT equal prepare_data', m < TOL, `(max rel ${m.toExponential(1)})`);
  check('R/G decreases with T', Array.from(rt.RG).every((v, i, a) => i === 0 || v < a[i - 1]));
}
// 4. table 1000 to 2600 K every 1 K, inversion against invert_monotone, B/G not monotonic
{
  const rt = ratioTable(band);
  const got = ref.invert.r.map((r: number) => invertRatio(rt.T, rt.RG, r));
  const ok = got.every((v: number, i: number) => (ref.invert.T[i] === null ? Number.isNaN(v) : Math.abs(v - ref.invert.T[i]) < 1e-6));
  check('T from R/G equals invert_monotone (NaN outside the table)', ok, `(${got.map((v: number) => v.toFixed(2)).join(', ')})`);
  let k = 0; for (let i = 1; i < rt.BG.length; i++) if (rt.BG[i] < rt.BG[k]) k = i;
  check('B/G is not monotonic, minimum where prepare_data finds it', !isMonotonic(rt.BG) && ref.BG_monotonic === false && rt.T[k] === ref.BG_min_T, `(${rt.T[k]} K)`);
  const Tb = invertRatio(rt.T, rt.BG, rt.BG[1200], { branch: 'high' });
  check('B/G inverted on its high branch when asked (branch high)', Math.abs(Tb - rt.T[1200]) < 1e-6, `(${Tb.toFixed(3)} K)`);
  // GB2r1-numerico-04: a value both branches reach is ambiguous, NaN by default (1200 K gave 2156 K before)
  const amb = [1100, 1200, 1300, 1400, 1500].map((T) => invertRatio(rt.T, rt.BG, rt.BG[T - 1000]));
  check('B/G of the low branch (1100 to 1500 K) gives NaN, not a false high T', amb.every(Number.isNaN), `(${amb.map((v) => v.toFixed(0)).join(', ')})`);
  const Tr = invertRatio(rt.T, rt.BG, rt.BG[1200], { range: [1600, 2600] });
  check('B/G with a T range above the minimum (from R/G) inverts on that branch', Math.abs(Tr - rt.T[1200]) < 1e-6, `(${Tr.toFixed(3)} K)`);
  check('R/G is monotonic and unaffected', isMonotonic(rt.RG) && Math.abs(invertRatio(rt.T, rt.RG, rt.RG[801]) - rt.T[801]) < 1e-6);
}
// 5. IR tail above 700 nm (ir_tail warning) and the tail × 0 and × 4 bands
{
  let m = 0;
  for (const c of CH) for (const t of [1073.15, 1273.15, 1800]) {
    const r = ref.tail700[`${c}_${t === 1800 ? '1800.0' : t}`];
    m = Math.max(m, Math.abs(tailFraction(band, c, t, 700, 'X') - r.X), Math.abs(tailFraction(band, c, t, 700, 'G') - r.G));
  }
  check('fraction of B_j and G_j above 700 nm equals prepare_data', m < 1e-9, `(max abs ${m.toExponential(1)})`);
  const r = ref.tail700['R_1073.15'], g = ref.tail700['G_1073.15'], b = ref.tail700['B_1073.15'];
  console.log(`      tail of B_j over 700 nm at 1073 K  R ${(100 * r.X).toFixed(0)} %, G ${(100 * g.X).toFixed(0)} %, B ${(100 * b.X).toFixed(0)} %`);
  const lo = scaleTail(band, 0), hi = scaleTail(band, 4);
  const x0 = bandX(band, 'R', 1273.15), f = tailFraction(band, 'R', 1273.15);
  check('tail × 0 and × 4 scale only above 700 nm', Math.abs(bandX(lo, 'R', 1273.15) / x0 - (1 - f)) < 1e-12 && Math.abs(bandX(hi, 'R', 1273.15) / x0 - (1 + 3 * f)) < 1e-12);
}
// 6. E(m) laws and IR cut
{
  const lq: number[] = ref.E_group.lamNm;
  check('E(m) of the group', maxRel(lq.map((l) => emValue({ kind: 'group' }, l)), ref.E_group.E) < 1e-14);
  const c = ref.E_coeffs;
  check('E(m) with edited coefficients', maxRel(lq.map((l) => emValue({ kind: 'coeffs', a: c.a, b: c.b, c: c.c }, l)), c.E) < 1e-14);
  const bc = withEmLaw(band, { kind: 'constant', E: 0.4 });
  const ratio = bandG(bc, 'R', 1800) / bandG(band, 'R', 1800);
  check('E constant changes G_j, ratio of laws is a weighted mean of 0,4/E', ratio > 0.4 / 0.45 && ratio < 0.4 / 0.2, `(${ratio.toFixed(3)})`);
  check('sigmoid IR cut', maxRel(ref.ircut.lamNm.map((l: number) => irCut(l, ref.ircut.cut, ref.ircut.width)), ref.ircut.tau) < 1e-14);
}
// 7. kernel_table (trapezoid in nm, per-nm Planck) and band_integral of radiometric_calibration.py
{
  const G = kernelTableTrapz(imx.lamNm, psi, ref.kernel_table.T);
  const m = Math.max(...CH.map((c, k) => maxRel(G[c], ref.kernel_table.G[k])));
  check('kernel_table equals radiometric_calibration', m < 1e-9, `(max rel ${m.toExponential(1)})`);
  const L = Array.from(imx.lamNm, (l) => planckNmRC(l, 2900));
  const mb = Math.max(...CH.map((c) => Math.abs(bandIntegral(imx.lamNm, psi[c], L) / ref.band_integral_2900[c] - 1)));
  check('band_integral of a 2900 K lamp', mb < 1e-9, `(max rel ${mb.toExponential(1)})`);
}
// 8. response CSV in the digitiser pair format, as load_responsivity, and the acA2040 preset
{
  const r = parseResponseCsv(readFileSync(fx('responsivity_basler.csv'), 'utf8'));
  const L = ref.load_responsivity;
  const ok = r.format === 'pairs' && r.lamNm.length === L.n && r.lamNm[0] === L.lam0 && r.lamNm[r.lamNm.length - 1] === L.lam1;
  let m = 0;
  for (const c of CH) {
    m = Math.max(m, maxAbsRel(L.probe_idx.map((i: number) => r.eta[c][i]), L.probe[c]));
    m = Math.max(m, Math.abs(r.eta[c].reduce((a, b) => a + b, 0) / L.sum[c] - 1));
  }
  check('responsivity_basler.csv read as load_responsivity', ok && m < 1e-12, `(${r.lamNm.length} nm, max rel ${m.toExponential(1)})`);
  const aca = PRESETS[1];
  check('acA2040 preset equals the parsed CSV', CH.every((c) => maxAbsRel(aca.response.eta[c], r.eta[c]) < 1e-14) && aca.irCut?.cutNm === 630);
  const cols = parseResponseCsv('# comment\nlambda_nm,R,G,B\n500,0.1,0.5,0.3\n501,0.2,0.6,0.25\n');
  check('response CSV in columns', cols.format === 'columns' && cols.lamNm[1] === 501 && cols.eta.B[1] === 0.25);
  const a = presetPsi(aca), b = presetPsi(aca, null);
  check('IR cut removable only on the acA2040', a.R[400] < b.R[400] && presetPsi(DEFAULT_PRESET, null).R[400] === psi.R[400]);
  check('preset from the pylon name and from the sensor size', presetFor({ name: 'Basler_acA2040-55uc__23734412__20260901_155912181_0000.tiff' }).preset.id === 'acA2040-55uc'
    && presetFor({ width: 1100, height: 2048 }).why === 'size' && presetFor({}).preset === DEFAULT_PRESET);
}

// short-pass 700 nm preset (decision after G4 r3), same logistic as prepare_data.shortpass (cut 700 nm, width
// 5 nm, floor 10⁻³), multiplied into τ(λ). The signal it removes at 1800 K is sp700_loss_R/G/B of checks.json.
{
  const t = [600, 700, 750, 900].map((l) => shortPass(l, SP700));
  check('short-pass 700 nm, about 1 below, half at the edge, floor OD 3 above', Math.abs(t[0] - 1) < 1e-8 && Math.abs(t[1] - (1 + 1e-3) / 2) < 1e-15 && Math.abs(t[2] - 1e-3) < 1e-4 && Math.abs(t[3] - 1e-3) < 1e-15);
  const sp = buildPsi(imx.lamNm, imx.eta, { basis: 'photon', shortPass: SP700 });
  check('short-pass enters ψ_j as a factor of τ', CH.every((c) => maxRel(sp[c], psi[c].map((v, k) => v * shortPass(imx.lamNm[k], SP700))) < 1e-14));
  const b0 = makeBand(imx.lamNm, psi), b1 = makeBand(imx.lamNm, sp);
  const loss = CH.map((c) => 100 * (1 - bandG(b1, c, 1800) / bandG(b0, c, 1800)));
  const want = [4.637448369141128, 2.6564960635970136, 8.703773643272772]; // manim/data/checks.json sp700_loss_R/G/B, valor
  check('soot signal removed by the short-pass at 1800 K equals sp700_loss_R/G/B', want.every(Number.isFinite) && loss.every((v, k) => Math.abs(v - want[k]) < 0.05), `(${loss.map((v) => v.toFixed(2)).join(', ')} % vs ${want.map((v) => v.toFixed(2)).join(', ')} %)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
