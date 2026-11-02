// Reader of the oven images for the IR tail (src/lib/workshop-rgb/ovenread.ts) against steps 2 to 4 of
// manim/tools/calibration.py (ovenread_ref.json, written by tools/make_web_fixtures.py --only ovenread), and the
// parity of the tail figures of the tool with data/checks.json on the 51 disc signals of the real oven.
// Reduced fixture for the CI (34 kB). The full parity, reading the 51 TIFFs of manim/data/oven_raw through the
// same code as the Worker, is tests/workshop-rgb-ovenread-full.ts (run by hand, log in WEB_REPORT.md).
// Tolerances declared in the fixture before running.
// run with `node --experimental-strip-types tests/workshop-rgb-ovenread.test.ts`
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { readGeometry, readSignal, referenceIndex, commonRadius, smoothG } from '../src/lib/workshop-rgb/ovenread.ts';
import { buildPsi } from '../src/lib/workshop-rgb/radiometry.ts';
import { DEFAULT_PRESET } from '../src/lib/workshop-rgb/presets.ts';
import { tailGrid, tailCalibration, tailPoints } from '../src/lib/workshop-rgb/tailcal.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const dirUrl = new URL('./fixtures/workshop-rgb/', import.meta.url);
const ref = JSON.parse(readFileSync(new URL('ovenread_ref.json', dirUrl), 'utf8'));
const tol = ref.tol;
const u16 = (file: string) => { const b = gunzipSync(readFileSync(new URL(file, dirUrl))); return new Uint16Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

// 1 geometry on the synthetic frame (box filter with reflect, half height, largest component, holes, ring rule)
{
  const s = ref.synthetic, g16 = u16(s.file);
  const dn = new Float32Array(s.w * s.h * 3);
  for (let p = 0; p < s.w * s.h; p++) dn[3 * p] = dn[3 * p + 1] = dn[3 * p + 2] = g16[p];
  const g = readGeometry({ w: s.w, h: s.h, dn }, 1), r = s.ref;
  check('synthetic aperture centre as calibration.py', Math.abs(g.cy - r.cy) <= tol.geometry_px && Math.abs(g.cx - r.cx) <= tol.geometry_px, `(${g.cy.toFixed(4)}, ${g.cx.toFixed(4)})`);
  check('synthetic area with the hole filled and the second blob left out', g.area === r.area, `(${g.area} px)`);
  check('synthetic plateau level', Math.abs(g.level - r.level) <= tol.level * r.level, `(${g.level.toFixed(3)})`);
  check('synthetic plateau radius by the 3 % ring rule', g.plateauR === r.plateauR, `(${g.plateauR} px)`);
  const sm = smoothG({ w: s.w, h: s.h, dn }, 9);
  check('smoothing keeps the mean of a flat patch', Math.abs(sm[72 * s.w + 88] - r.level) < 50);
}
// 2 disc mean and ceiling count on a real crop (1050 °C, 3000 µs)
{
  const c = ref.crop, a = u16(c.file);
  const dn = Float32Array.from(a);
  const d = readSignal({ w: c.w, h: c.h, dn }, 1050, 3000, c.center_yx[0] - c.y0, c.center_yx[1] - c.x0, c.R, 4095);
  const want = c.disc_full.S;
  check('disc of the crop has the pixels of calibration.py', d.n === c.disc_full.n, `(${d.n})`);
  check('disc mean per channel equals calibration.py', ['R', 'G', 'B'].every((k, j) => Math.abs((d.S as Record<string, number>)[k] - want[j]) <= tol.S * want[j]), `(R ${d.S.R.toFixed(4)}, G ${d.S.G.toFixed(4)})`);
  check('ceiling count per channel', d.nSat.R === c.disc_full.nSat[0] && d.nSat.G === c.disc_full.nSat[1]);
}
// 3 reference frame and common radius
{
  const gc = ref.geometry_calibration;
  check('common radius is the smallest plateau', commonRadius(Object.values(gc).map((v: any) => ({ plateauR: v.plateau_r_px })) as any) === ref.R, `(${ref.R} px)`);
  const t = [1000, 2000, 3000, 4000], mx = [[900, 500, 100], [1800, 1000, 200], [2700, 1500, 300], [4095, 2000, 400]];
  check('reference frame, longest exposure with nothing at the ceiling', referenceIndex(t, mx, 4095) === 2);
}
// 4 tail figures of the tool on the 51 disc signals equal checks.json
{
  const T = ref.target, rel = (a: number, b: number) => Math.abs(a - b) <= tol.tail_rel * Math.abs(b), abs = (a: number, b: number) => Math.abs(a - b) <= tol.tail_pct_abs;
  const p = DEFAULT_PRESET.response;
  const g = tailGrid(p.lamNm, buildPsi(p.lamNm, p.eta, { basis: 'photon' }));
  const t = tailCalibration(tailPoints(ref.signals, { ceilingDn: 4095, darkMode: 'intercept' }), g, { sigmaT: 5 });
  check('51 images, 39 R/G pairs', ref.signals.length === 51 && t.nPairs === T.n_pairs, `(${t.nPairs})`);
  check('s_R = oven_tail_scale_R', rel(t.s, T.s_R), `(${t.s.toFixed(6)} against ${T.s_R.toFixed(6)})`);
  check('K_R/K_G = oven_tailcal_KRKG', rel(t.KRKG, T.K_R_K_G), `(${t.KRKG.toFixed(6)})`);
  check('drift = oven_tailcal_drift', abs(t.driftPct, T.drift_pct) && abs(t.driftNominalPct, T.drift_nominal_pct), `(${t.driftPct.toFixed(4)} %, nominal ${t.driftNominalPct.toFixed(3)} %)`);
  check('one T out, worst = oven_tailcal_loo_KRKG', abs(t.criterion.worstPct, T.loo_KRKG_worst_pct) && abs(t.criterion.nominalPct, T.loo_KRKG_nominal_pct), `(${t.criterion.worstPct.toFixed(4)} %, before ${t.criterion.nominalPct.toFixed(3)} %)`);
  check('one T out, every temperature', t.loo.every((l) => abs(l.KRKGrelPct, T.loo_KRKG_pct[String(l.tC)]) && rel(l.s, T.loo_s[String(l.tC)])));
  check('800 °C out (the strong test)', t.criterion.strongT === T.strong_T && abs(t.criterion.strongPct, T.strong_pct), `(${t.criterion.strongPct.toFixed(4)} %)`);
  check('criterion passes', t.criterion.pass);
}
console.log(fails ? `${fails} FAIL` : 'all PASS');
if (fails) process.exit(1);
