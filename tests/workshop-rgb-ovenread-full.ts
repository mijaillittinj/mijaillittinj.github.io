// Full parity of the IR tail calibration of the tool with tools/calibration.py on the real oven images
// (decision of the director for the closing). Reads every TIFF of <dir>/<T>C/<t>exp.tiff through the code the
// Worker runs (engine.ts loadFrame, readTailSignals of ovenread.ts, runOvenFit), and compares s_R, K_R/K_G, the
// drift and the test with one temperature out with data/checks.json (oven_tail_scale_R, oven_tailcal_*).
// It also prints what the plateau reading of 1a.3 (2 × 2 grouping, median, saturation dilated across
// channels) gives, the figures the tool showed before.
// Not part of the CI (the 51 images are 700 MB), the CI runs tests/workshop-rgb-ovenread.test.ts.
//   node --experimental-strip-types tests/workshop-rgb-ovenread-full.ts [dir] [checks.json]
// defaults ~/Desktop/Workshop_RGB_Pyrometry/manim/data/oven_raw and ../data/checks.json next to it.
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, dirname } from 'node:path';
import { loadFrame, seriesHot, ovenFrameOf, analyseSeries, frameSignal, runOvenFit, readTailSignals, bandFor, tailGrid } from '../src/components/workshop-rgb/engine.ts';
import { buildPsi } from '../src/lib/workshop-rgb/radiometry.ts';
import { DEFAULT_PRESET } from '../src/lib/workshop-rgb/presets.ts';
import type { FrameSignal } from '../src/lib/workshop-rgb/oven.ts';

const dir = process.argv[2] ?? join(homedir(), 'Desktop/Workshop_RGB_Pyrometry/manim/data/oven_raw');
const checksPath = process.argv[3] ?? join(dirname(dir), 'checks.json');
if (!existsSync(dir)) { console.log(`SKIP  no images in ${dir}`); process.exit(0); }
let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const opts = { group: true, bayer: 'RGGB' as const, centre: false };
const buf = (p: string) => { const b = readFileSync(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const temps = readdirSync(dir).filter((d) => /^\d+C$/.test(d)).map((d) => +d.slice(0, -1)).sort((a, b) => a - b);
const files: Record<number, { path: string; t: number }[]> = {};
for (const tC of temps) files[tC] = readdirSync(join(dir, `${tC}C`)).filter((f) => /^\d+exp\.tiff?$/.test(f)).map((f) => ({ path: join(dir, `${tC}C`, f), t: parseInt(f, 10) })).sort((a, b) => a.t - b.t);
const n = temps.reduce((a, t) => a + files[t].length, 0);
console.log(`${n} images, ${temps.join(', ')} °C, ${dir}`);

// as the Worker, oven.analyse then oven.fit with the proposed plateaus
const t0 = Date.now();
const signals: FrameSignal[] = [], max12: Record<string, number[]> = {};
let ceil12 = 4095;
for (const tC of temps) {
  const fr = files[tC].map((f) => loadFrame(buf(f.path), opts));
  ceil12 = fr[0].info.ceil12;
  const hot = seriesHot(fr.map((x) => x.frame), [], ceil12);
  let pick = -1;
  fr.forEach((x, i) => { const inf = x.info; if (inf.satRaw.every((v) => v === 0) && inf.p999.every((v) => v < 0.9 * inf.ceil12) && (pick < 0 || inf.meanG > fr[pick].info.meanG)) pick = i; });
  if (pick < 0) pick = 0;
  const { aperture, plateau } = analyseSeries(ovenFrameOf(tC, files[tC][pick].t, fr[pick].frame, hot));
  fr.forEach((x, i) => { max12[files[tC][i].path] = x.info.max12; signals.push(frameSignal(ovenFrameOf(tC, files[tC][i].t, x.frame, hot), plateau, aperture)); });
}
const tail = await readTailSignals(temps.map((tC) => ({ tC, texpUs: files[tC].map((f) => f.t), max12: files[tC].map((f) => max12[f.path]), load: async (i: number) => loadFrame(buf(files[tC][i].path), { ...opts, group: false }) })), ceil12, new Map(signals.map((s) => [`${s.tC}|${s.texpUs}`, s.strip])));
const p = DEFAULT_PRESET.response, psi = buildPsi(p.lamNm, p.eta, { basis: 'photon' });
const base = { signals, band: bandFor(p.lamNm, psi, { kind: 'group' }), bandEnergy: null, eps: 1, sigmaT: 5, ceil12, darkMode: 'intercept' as const, perT: 'slope' as const, calibrateAt: 'highest' as const, tailGrid: tailGrid(p.lamNm, psi) };
const after = runOvenFit({ ...base, tailSignals: tail.signals, tailRead: tail.read });
const before = runOvenFit(base);
console.log(`read in ${((Date.now() - t0) / 1000).toFixed(1)} s, disc radius ${tail.read.R} px, plateaus ${tail.read.geo.map((g) => `${g.tC} °C ${g.plateauR} px (ref ${g.refTexpUs} µs)`).join(', ')}`);
const show = (k: string, r: typeof after) => { const t = r.tail!; console.log(`${k.padEnd(28)} s_R ${t.s.toFixed(4)} ± ${t.u.s.toFixed(4)}  K_R/K_G ${t.KRKG.toFixed(4)}  drift ${t.driftNominalPct.toFixed(2)} → ${t.driftPct.toFixed(3)} %  one T out worst ${t.criterion.worstPct.toFixed(3)} % (before ${t.criterion.nominalPct.toFixed(2)} %)  ${t.criterion.strongT} °C out ${t.criterion.strongPct.toFixed(3)} %  ${t.loo.map((l) => `${l.tC}: ${l.KRKGrelPct.toFixed(3)}`).join(' ')}`); };
show('before (plateau of 1a.3)', before);
show('now (reader of calibration.py)', after);
if (!existsSync(checksPath)) { console.log(`SKIP  comparison, no ${checksPath}`); process.exit(0); }
const ck = JSON.parse(readFileSync(checksPath, 'utf8'));
const t = after.tail!, lk = ck.oven_tailcal_loo_KRKG;
const rel = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.abs(b), abs = (a: number, b: number) => Math.abs(a - b) <= 1e-5;
if (n === 51) {
  check('s_R = oven_tail_scale_R at 1e-6', rel(t.s, ck.oven_tail_scale_R.valor), `(${t.s.toFixed(7)} against ${ck.oven_tail_scale_R.valor.toFixed(7)})`);
  check('K_R/K_G = oven_tailcal_KRKG at 1e-6', rel(t.KRKG, ck.oven_tailcal_KRKG.valor), `(${t.KRKG.toFixed(7)})`);
  check('drift = oven_tailcal_drift at 1e-5 points', abs(t.driftPct, ck.oven_tailcal_drift.valor), `(${t.driftPct.toFixed(5)} %)`);
  check('one T out worst = oven_tailcal_loo_KRKG at 1e-5 points', abs(t.criterion.worstPct, lk.valor), `(${t.criterion.worstPct.toFixed(5)} %)`);
  check('each temperature out at 1e-5 points', t.loo.every((l) => abs(l.KRKGrelPct, lk.por_T[String(l.tC)])));
  check('800 °C out = prueba_fuerte_pct', abs(t.criterion.strongPct, lk.prueba_fuerte_pct), `(${t.criterion.strongPct.toFixed(5)} %)`);
  const oc = join(dirname(checksPath), 'oven', 'oven_calibration.json');
  if (existsSync(oc)) {
    const g = JSON.parse(readFileSync(oc, 'utf8'));
    check('common radius and plateaus = oven_calibration.json', tail.read.R === g.roi_radius_px && tail.read.geo.every((x) => x.plateauR === g.geometria[String(x.tC)].plateau_r_px && Math.hypot(x.cy - g.geometria[String(x.tC)].center_yx[0], x.cx - g.geometria[String(x.tC)].center_yx[1]) < 1e-9), `(${tail.read.R} px)`);
  }
} else console.log(`not the 51 images of checks.json (${n}), comparison skipped`);
console.log(fails ? `${fails} FAIL` : 'all PASS');
if (fails) process.exit(1);
