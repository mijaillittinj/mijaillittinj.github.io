// Oven calibration (plateau, dark by intercept and strip, K_j with σ, residuals, linearity, σ_T, criterion,
// ratio test, IR tail) against the numpy rewrite in manim/tools/make_web_fixtures.py on the same images:
// run with `node --experimental-strip-types tests/workshop-rgb-oven.test.ts`
// Images: 3 temperatures × 3 t_exp of fuentes/02_calibracion_horno (a2A1920), window of 720 px on the
// aperture decimated 8 × 8, 16 bit MM as pylon writes them.
import { readFileSync } from 'node:fs';
import { readTiff, detectEncoding } from '../src/lib/workshop-rgb/tiff.ts';
import { saturationMask } from '../src/lib/workshop-rgb/frames.ts';
import { bandG, makeBand, buildPsi, bandX, CH } from '../src/lib/workshop-rgb/radiometry.ts';
import { DEFAULT_PRESET } from '../src/lib/workshop-rgb/presets.ts';
import { compareHottest, findAperture, proposePlateau, frameSignal, fitOven, choosePlateauFrame, fitDarkLine, throughOrigin, type OvenFrame, type FrameSignal } from '../src/lib/workshop-rgb/oven.ts';
import { decideCalibration } from '../src/lib/workshop-rgb/calstate.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const ref = JSON.parse(readFileSync(fx('oven_ref.json'), 'utf8'));
const TOL = 1e-9; // WEB_SPEC §14.1
let worst = 0;
const same = (a: number, b: number | null) => {
  if (b === null) return Number.isNaN(a);
  const e = Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
  if (e > worst) worst = e;
  return e <= TOL;
};
const all = (a: number[], b: (number | null)[]) => a.length === b.length && a.every((v, i) => same(v, b[i]));

const P = ref.params;
const frames: OvenFrame[] = ref.files.map((f: string) => {
  const img = readTiff(readFileSync(fx(f)));
  const enc = detectEncoding(img);
  const [, tC, t] = f.match(/(\d+)C\/(\d+)exp/)!;
  const dn = Float64Array.from(img.data, (v) => v * enc.toDn12);
  return { tC: +tC, texpUs: +t, width: img.width, height: img.height, dn, flag: saturationMask({ width: img.width, height: img.height, nc: 3, data: img.data }, enc.ceiling) };
});
check('nine oven images read, 12 bits shifted', frames.length === 9 && frames.every((f) => f.width === 90 && f.height === 90));

// 1. aperture and plateau on the brightest frame with nothing saturated
const pf = choosePlateauFrame(frames);
const ap = findAperture(pf);
check('aperture (centroid, radius, background, top) equals numpy', same(ap.cy, ref.aperture.cy) && same(ap.cx, ref.aperture.cx) && same(ap.r, ref.aperture.r) && same(ap.bg, ref.aperture.bg) && same(ap.top, ref.aperture.top),
  `(r = ${ap.r.toFixed(2)} px decimated, ${(8 * ap.r).toFixed(0)} px on the sensor)`);
const pl = proposePlateau(pf, ap, { block: P.block, tolI: P.tolI, tolRatio: P.tolRatio, fitRho: P.fitRho });
const RP = ref.plateau;
check('plateau radius, block profiles and parabola curvatures equal numpy', same(pl.radius, RP.radius) && all(pl.profile.r, RP.profile.r) && pl.profile.n.join(',') === RP.profile.n.join(',')
  && CH.every((c) => all(pl.profile.I[c], RP.profile.I[c]) && same(pl.shape.kI[c], RP.shape.kI[c])) && all(pl.profile.RG, RP.profile.RG) && all(pl.profile.BG, RP.profile.BG) && same(pl.shape.kRG, RP.shape.kRG) && same(pl.shape.kBG, RP.shape.kBG),
  `(${pl.radius.toFixed(2)} px, ${(pl.radius / ap.r).toFixed(2)} radii, the oven image is a dome)`);
check('plateau flatness at its edge equals numpy', CH.every((c) => same(pl.flatness[c], RP.flatness[c])) && same(pl.flatRG, RP.flatRG) && same(pl.flatBG, RP.flatBG));

// 1b. full resolution crops (GB2r1-numerico-01): 1 px rings gave a 1 px plateau here, blocks of 8 × 8 do not
for (const F of ref.full) {
  const img = readTiff(readFileSync(fx(F.file)));
  const enc = detectEncoding(img);
  const fr: OvenFrame = { tC: F.tC, texpUs: F.texpUs, width: img.width, height: img.height, dn: Float64Array.from(img.data, (v) => v * enc.toDn12), flag: saturationMask({ width: img.width, height: img.height, nc: 3, data: img.data }, enc.ceiling) };
  const apF = { ...F.aperture, bg: 0, top: 0 };
  const p = proposePlateau(fr, apF);
  const frac = p.radius / apF.r;
  check(`${F.tC} °C full resolution, plateau at least ${ref.full_min_radius_frac} radii and equal to numpy`, frac >= ref.full_min_radius_frac && same(p.radius, F.plateau.radius) && CH.every((c) => same(p.shape.kI[c], F.plateau.shape.kI[c])),
    `(${p.radius.toFixed(1)} px, ${frac.toFixed(2)} radii of ${apF.r.toFixed(0)} px, ${p.profile.n[0]} blocks in the centre)`);
}

// 2. per frame signal in the plateau and in the strip outside 2,5 radii
const sig = frames.map((f) => frameSignal(f, pl, ap, P.stripFactor));
let okS = true;
sig.forEach((s, i) => {
  const r = ref.signals[i];
  for (const c of CH) if (!(same(s.S[c], r.S[c]) && same(s.p999[c], r.p999[c]) && s.nSat[c] === r.nSat[c] && same(s.strip[c], r.strip[c]) && same(s.stripSigma[c], r.stripSigma[c]) && s.n[c] === r.n[c])) okS = false;
});
check('plateau median, p99,9, saturated count and strip per frame equal numpy', okS);

// 3. fit
const band = makeBand(DEFAULT_PRESET.response.lamNm, buildPsi(DEFAULT_PRESET.response.lamNm, DEFAULT_PRESET.response.eta, { basis: 'photon' }));
const fit = fitOven(sig, { band, eps: P.eps, sigmaT: P.sigmaT, ceilingDn: P.ceilingDn, maxFrac: P.maxFrac, minSnr: P.minSnr, minTexpUs: P.minTexpUs, tailCutNm: P.tailCutNm, darkMode: 'intercept' });
const F = ref.fit;
check('accepted channels per image equal numpy', CH.every((c) => fit.accepted[c].join(',') === F.accepted[c].join(',')), `(R ${fit.accepted.R.filter(Boolean).length} of 9)`);
for (const c of CH) {
  const a = fit.channels[c], r = F.channels[c];
  const perT = a.perT.length === r.perT.length && a.perT.every((p, k) => {
    const q = r.perT[k];
    return p.tC === q.tC && p.n === q.n && (q.a === null ? Number.isNaN(p.a) : Math.abs(p.a - q.a) <= TOL * P.ceilingDn) && same(p.b, q.b) && same(p.sb, q.sb) && same(p.B, q.B) && same(p.K, q.K) && same(p.sK, q.sK) && same(p.linMaxPct, q.linMaxPct);
  });
  check(`${c}: slope, intercept, σ, B_j(T), K_j(T) and linearity per temperature`, perT);
  check(`${c}: K at ${a.Tcal.toFixed(0)} K = ${(a.K / 1e6).toFixed(4)}e6 ± ${(a.sK / 1e6).toFixed(4)}e6 DN s⁻¹ (W m⁻² sr⁻¹)⁻¹, residuals, σ_T, IR tail`,
    same(a.K, r.K) && same(a.Tcal, r.Tcal) && same(a.sKstat, r.sKstat) && same(a.sKT, r.sKT) && same(a.sK, r.sK) && all(a.resid, r.resid) && a.weights === r.weights && same(a.tailLo, r.tailLo) && same(a.tailHi, r.tailHi));
  check(`${c}: K of all temperatures (line through the origin) and its σ, for comparison`, same(a.Kall, r.Kall) && same(a.sKallStat, r.sKallStat), `(${(a.Kall / 1e6).toFixed(4)}e6 ± ${(a.sKallStat / 1e6).toFixed(4)}e6)`);
}
for (const [k, r] of [['RG', F.RG], ['BG', F.BG]] as const) {
  const a = fit.ratio[k];
  check(`${k}: dispersion of ln K ratio against what ±σ_T and the fit allow`, all(a.lnq, r.lnq) && same(a.dispersion, r.dispersion) && same(a.allowedT, r.allowedT) && same(a.allowedStat, r.allowedStat) && a.pass === r.pass_,
    `(${a.dispersion.toFixed(4)} against ${a.allowed.toFixed(4)}, ${a.pass ? 'passes' : 'fails'})`);
}
const rtOk = fit.ratioTest.length === F.ratioTest.length && fit.ratioTest.every((x, i) => same(x.rMeas, F.ratioTest[i].rMeas) && same(x.Tpred, F.ratioTest[i].Tpred) && Math.abs(x.dT - F.ratioTest[i].dT) < 1e-6);
check('ratio test (R/G of the blackbody, K_j leaving that temperature out) equals brentq', rtOk, `(ΔT ${fit.ratioTest.map((x) => x.dT.toFixed(0)).join(', ')} K)`);
const rtt = fit.ratioTestTop.length === F.ratioTestTop.length && fit.ratioTestTop.every((x, i) => same(x.rMeas, F.ratioTestTop[i].rMeas) && Math.abs(x.dT - F.ratioTestTop[i].dT) < 1e-6) && fit.Ttop === F.Ttop;
check(`ratio test with K_R/K_G of the highest temperature (${fit.Ttop} °C, default convention) equals brentq`, rtt, `(ΔT ${fit.ratioTestTop.map((x) => `${x.dT.toFixed(0)} K at ${x.tC} °C`).join(', ')})`);
console.log(`      worst relative difference against numpy ${worst.toExponential(1)}`);

// 4. what the real oven says with the nominal IMX392 curve (G2 finding), and the decision of stage 2
check('the real oven is not consistent with one K per channel, as found in G2', !fit.consistent && fit.calibrated.length === 0);
check('slopes follow R > G > B at every temperature', fit.order.ok && fit.order.proposal.join('') === 'RGB');
const st = decideCalibration({ source: 'oven', calibrated: fit.calibrated, consistent: fit.consistent, emissivityKnown: true, texpConfirmed: true, scaleConfirmed: true });
check('stage 2 says cal_inconsistent, T with a warning and no f_s', st.state === 'inconsistent' && st.fs === 'no' && st.T === 'warn' && st.codes[0] === 'cal_inconsistent');
check('stray light of the strip as a percentage of the plateau (stray_light over 1 %)', CH.every((c) => Number.isFinite(fit.strayPct[c])), `(R/G/B ${CH.map((c) => fit.strayPct[c].toFixed(2)).join(' / ')} %)`);

// 5. ε and the strip dark, and a synthetic consistent oven
{
  const f9 = fitOven(sig, { band, eps: 0.9, sigmaT: 5 });
  check('ε = 0,9 scales every K_j by 1/0,9 and leaves the ratios unchanged', CH.every((c) => Math.abs(f9.channels[c].K * 0.9 / fit.channels[c].K - 1) < 1e-12) && Math.abs(f9.ratio.RG.dispersion - fit.ratio.RG.dispersion) < 1e-12);
  const fs = fitOven(sig, { band, darkMode: 'strip' });
  // with the strip as dark the slope is the plateau slope minus the slope of the strip (least squares is linear)
  const okStrip = CH.every((c) => fs.channels[c].perT.every((p, k) => {
    const q = fit.channels[c].perT[k];
    const idx = sig.map((_, i) => i).filter((i) => sig[i].tC === p.tC && fit.accepted[c][i]);
    if (idx.length < 2 || idx.length !== p.n) return true;
    const t = idx.map((i) => sig[i].texpUs * 1e-6), y = idx.map((i) => sig[i].strip[c]);
    const tm = t.reduce((a, b) => a + b) / t.length, ym = y.reduce((a, b) => a + b) / y.length;
    let sxy = 0, sxx = 0; t.forEach((v, i) => { sxy += (v - tm) * (y[i] - ym); sxx += (v - tm) ** 2; });
    return Math.abs(p.b - (q.b - sxy / sxx)) <= 1e-9 * q.b;
  }));
  check('strip dark gives the plateau slope minus the strip slope', okStrip);
  // synthetic oven with K = (3, 4, 6)·10⁵ and a pedestal of 40 DN, no noise
  const K = { R: 3e5, G: 4e5, B: 6e5 };
  const syn: FrameSignal[] = [];
  for (const tC of [800, 900, 1000, 1050]) for (const t of [1000, 2000, 3000]) {
    const S = { R: 0, G: 0, B: 0 };
    for (const c of CH) S[c] = 40 + K[c] * t * 1e-6 * bandX(band, c, tC + 273.15);
    syn.push({ tC, texpUs: t, S, p999: { ...S }, nSat: { R: 0, G: 0, B: 0 }, n: { R: 50, G: 50, B: 50 }, strip: { R: 40, G: 40, B: 40 }, stripSigma: { R: 1, G: 1, B: 1 }, nStrip: { R: 9, G: 9, B: 9 } });
  }
  const fsyn = fitOven(syn, { band, ceilingDn: Infinity });
  check('synthetic oven with one K per channel is consistent and gives three channels', fsyn.consistent && fsyn.calibrated.length === 3 && CH.every((c) => Math.abs(fsyn.channels[c].K / K[c] - 1) < 1e-9));
  check('synthetic ratio test gives ΔT within 1e-6 K', fsyn.ratioTest.every((x) => Math.abs(x.dT) < 1e-6), `(${fsyn.ratioTest.map((x) => x.dT.toExponential(0)).join(', ')})`);
  check('synthetic intercept recovers the pedestal of 40 DN', CH.every((c) => fsyn.channels[c].perT.every((p) => Math.abs(p.a - 40) < 1e-6)));
  const stR = decideCalibration({ source: 'oven', calibrated: fsyn.calibrated, consistent: fsyn.consistent, emissivityKnown: false, texpConfirmed: true, scaleConfirmed: false });
  check('consistent oven with ε unknown is relative, unconfirmed scale makes f_s provisional', stR.state === 'relative' && stR.fs === 'relative' && stR.provisional && stR.T === 'yes' && stR.codes.includes('scale_missing'));
  const dl = fitDarkLine([{ texpUs: 1000, level: { R: 41, G: 40, B: 40 } }, { texpUs: 3000, level: { R: 43, G: 40, B: 41 } }]);
  check('dark line D = a + b t_exp from dark frames', Math.abs(dl.R.a - 40) < 1e-9 && Math.abs(dl.R.b - 1000) < 1e-6);
}

// 6. σ_K with relative weights (GB2r1-numerico-02): one t_exp per temperature, 1 % scatter, known K
{
  const K = { R: 3e5, G: 4e5, B: 6e5 }, scat = [1.01, 0.99, 1.005, 0.995];
  const sg: FrameSignal[] = [800, 900, 1000, 1050].map((tC, k) => {
    const S = { R: 0, G: 0, B: 0 };
    for (const c of CH) S[c] = K[c] * 2e-3 * bandX(band, c, tC + 273.15) * scat[k];
    return { tC, texpUs: 2000, S, p999: { ...S }, nSat: { R: 0, G: 0, B: 0 }, n: { R: 50, G: 50, B: 50 }, strip: { R: NaN, G: NaN, B: NaN }, stripSigma: { R: 1, G: 1, B: 1 }, nStrip: { R: 0, G: 0, B: 0 } };
  });
  const f1 = fitOven(sg, { band, ceilingDn: Infinity, calibrateAt: 'all' });
  const rel = CH.map((c) => f1.channels[c].sKallStat / f1.channels[c].Kall);
  check('relative weights, 1 % scatter gives σ_K/K under 1 % (it was 1/√n = 50 %)', rel.every((v) => v < 0.01) && f1.channels.R.weights === 'relative', `(${rel.map((v) => (100 * v).toFixed(2)).join(' / ')} %)`);
  const b = [1, 2, 3], B = [1, 2, 3];
  const t0 = throughOrigin(b, B, [NaN, NaN, NaN]);
  check('relative weights, points exactly on the line give σ_K = 0', t0.K === 1 && t0.sStat === 0);
  const t1 = throughOrigin([1.1, 1.9, 3.05], B, [0.01, 0.01, 0.01]);
  check('with σ_b known the factor stays max(1, √(χ²/(n − 1)))', t1.weights === 'sigma' && t1.sStat >= Math.sqrt(1 / (3 * 3 + 2 * 2 + 1)) * 0.01 * Math.sqrt(1 / 1));
}

// 7. parity with tools/calibration.py, the numbers of the presentation (GB2r1-numerico-03), on its own
// plateau signals (disc of 39 px, mean, data/oven/oven_fit_points.csv) with fitOven perT 'band'
{
  const C = JSON.parse(readFileSync(fx('oven_calibration_ref.json'), 'utf8'));
  const key = (p: any) => `${p.tC}/${p.texpUs}`;
  const byKey = new Map<string, FrameSignal>();
  for (const p of C.points) {
    let s = byKey.get(key(p));
    if (!s) { s = { tC: p.tC, texpUs: p.texpUs, S: { R: 0, G: 0, B: 0 }, p999: { R: 0, G: 0, B: 0 }, nSat: { R: 0, G: 0, B: 0 }, n: { R: 1, G: 1, B: 1 }, strip: { R: NaN, G: NaN, B: NaN }, stripSigma: { R: NaN, G: NaN, B: NaN }, nStrip: { R: 0, G: 0, B: 0 } }; byKey.set(key(p), s); }
    s.S[p.ch as 'R'] = p.S; s.p999[p.ch as 'R'] = p.S; s.nSat[p.ch as 'R'] = p.nSat;
  }
  const sg = [...byKey.values()];
  const fa = fitOven(sg, { band, ceilingDn: 4095, perT: 'band', calibrateAt: 'all', minTexpUs: 0 });
  let mK = 0;
  for (const c of CH) for (const p of fa.channels[c].perT) mK = Math.max(mK, Math.abs(p.K / C.K_perT[c][String(p.tC)] - 1));
  const mG = Math.max(...CH.map((c) => Math.abs(fa.channels[c].Kall / C.K_geomean[c] - 1)));
  check(`band convention: K_j(T) and geometric mean equal calibration.py within ${C.tol_K}`, mK <= C.tol_K && mG <= C.tol_K && CH.every((c) => fa.channels[c].perT.length === Object.keys(C.K_perT[c]).length), `(K_j(T) ${mK.toExponential(1)}, mean ${mG.toExponential(1)})`);
  const dL = fa.ratioTest.map((x) => Math.abs(x.dT - C.dT_loo[String(x.tC)]));
  const r1050 = fa.ratioTest.find((x) => x.tC === 1050)!;
  check(`band convention: leave one out ratio test equals calibration.py within ${C.tol_dT_K} K (oven_ratio_test_dT)`, Math.max(...dL) <= C.tol_dT_K && Math.abs(r1050.dT - C.oven_ratio_test_dT) <= C.tol_dT_K,
    `(${fa.ratioTest.map((x) => `${x.dT >= 0 ? '+' : ''}${x.dT.toFixed(0)}`).join(', ')} K, checks.json ${C.oven_ratio_test_dT.toFixed(0)} K at 1050 °C)`);
  const fh = fitOven(sg, { band, ceilingDn: 4095, perT: 'band', calibrateAt: 'highest', minTexpUs: 0 });
  const dTop = fh.ratioTestTop.map((x) => Math.abs(x.dT - C.oven_ratio_test_top_dT[String(x.tC)]));
  check(`band convention, K of the highest temperature: ratio test equals oven_ratio_test_top_dT within ${C.tol_dT_K} K`, fh.Ttop === 1050 && dTop.length === 3 && Math.max(...dTop) <= C.tol_dT_K,
    `(${fh.ratioTestTop.map((x) => `+${x.dT.toFixed(0)} K at ${x.tC} °C`).join(', ')})`);
  const fs = fitOven(sg, { band, ceilingDn: 4095, perT: 'slope', calibrateAt: 'highest', minTexpUs: 0 });
  console.log(`      slope convention on the same points, ratio test ${fs.ratioTestTop.map((x) => `${x.dT >= 0 ? '+' : ''}${x.dT.toFixed(0)} K at ${x.tC} °C`).join(', ')} (difference of the K_j(T) estimator)`);
}

// calibration with the hottest oven temperature (professor, 2026-11-03)
{
  const lam = Float64Array.from({ length: 401 }, (_, i) => 400 + i);
  const psi = { R: lam.map((l) => Math.exp(-(((l - 600) / 40) ** 2))), G: lam.map((l) => Math.exp(-(((l - 530) / 40) ** 2))), B: lam.map((l) => Math.exp(-(((l - 460) / 40) ** 2))) };
  const bd = makeBand(lam, psi);
  const row = (tC: number, K: number) => ({ tC, TK: tC + 273.15, n: 3, a: 0, b: 1, sb: 0, B: 1, K, sK: 0, linMaxPct: 0, tail: 0 });
  const mk = (rows: [number, number][]) => ({ channels: { R: { perT: rows.map(([t, q]) => row(t, q)) }, G: { perT: rows.map(([t]) => row(t, 1)) } } }) as never;
  const c = compareHottest(mk([[800, 0.7], [1000, 0.87], [1050, 0.89]]), bd);
  const r = (T: number) => bandG(bd, 'R', T) / bandG(bd, 'G', T);
  check('compareHottest uses the hottest and compares with the next', c.tUsed === 1050 && c.tNext === 1000 && c.rows.length === 3);
  check('compareHottest, the flame reads lower with the smaller K_R/K_G and keeps the measured ratio', c.shiftK! < 0 && Math.abs((0.87 * r(1800 + c.shiftK!)) / (0.89 * r(1800)) - 1) < 1e-8, `(${c.shiftK!.toFixed(1)} K)`);
  const one = compareHottest(mk([[1050, 0.89]]), bd);
  check('compareHottest with one temperature gives no comparison', one.shiftK === null && one.tNext === null && one.tUsed === 1050);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }

