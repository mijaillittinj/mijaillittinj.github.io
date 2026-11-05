// More points with SAT or a network, the mask by f_s, and what runBand really uses (the list of step 5).
// Run with `node --experimental-strip-types tests/workshop-rgb-refine.test.ts`.
import { chordMatrixHalf } from '../src/lib/abel.ts';
import { splineBasis } from '../src/lib/sat.ts';
import { matvec } from '../src/lib/linalg.ts';
import { makeBand, buildPsi, ratioTable, bandG } from '../src/lib/workshop-rgb/radiometry.ts';
import { DEFAULT_PRESET } from '../src/lib/workshop-rgb/presets.ts';
import { runAbel, runTfs, runRefine, runBand, type Prepared, type TfsOpts, type AbelRun, type EmissionNet, type TfsNet } from '../src/components/workshop-rgb/engine.ts';
import { fineGrid, finePoints, splineBasisAt, splineX, fsMaskOf, runsOf, validMaskPct, refineFactor, forNearest, basePointsOnFine, REFINE_TOL_T_K, REFINE_TOL_REL } from '../src/lib/workshop-rgb/refine.ts';
import { buildProblem, type Projections } from '../src/lib/workshop-rgb/annabel/problem.ts';
import { initXavier } from '../src/lib/workshop-rgb/annabel/net.ts';
import { LocalEngine, trainWithEngine, finishStage1 } from '../src/lib/workshop-rgb/annabel/train.ts';
import { buildStage2Problem, Stage2LocalEngine, trainStage2, finishStage2, initStage2, type KernelTable } from '../src/lib/workshop-rgb/annabel/stage2.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const e = (x: number) => x.toExponential(1);

// A smooth synthetic flame, projected with the matrix of the tool, with a little noise so the halves differ.
const imx = DEFAULT_PRESET.response;
const band = makeBand(imx.lamNm, buildPsi(imx.lamNm, imx.eta));
const table = ratioTable(band);
const nz = 20, nr = 26, drMm = 0.2, N = nz * nr;
const rMm = Float64Array.from({ length: nr }, (_, i) => (i + 0.5) * drMm);
const zMm = Float64Array.from({ length: nz }, (_, k) => 10 + k * 0.8);
const Ttrue = (r: number, z: number) => 1650 + 200 * Math.exp(-(((r - 2) / 1.4) ** 2)) + 3 * (z - 10);
const fTrue = (r: number, z: number) => 2e-6 * Math.exp(-(((r - 2.2) / 1.1) ** 2)) * (0.6 + 0.4 * Math.sin((z - 10) / 5));
const jTrue = (c: 'R' | 'G' | 'B', r: number, z: number) => fTrue(r, z) * bandG(band, c, Ttrue(r, z));
const L = chordMatrixHalf(nr, drMm * 1e-3);
let seed = 12345;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296 - 0.5; };
const right: Float64Array[][] = [], left: Float64Array[][] = [];
const CH = ['R', 'G', 'B'] as const;
for (let c = 0; c < 3; c++) {
  const R: Float64Array[] = [], Lh: Float64Array[] = [];
  for (let k = 0; k < nz; k++) {
    const f = Float64Array.from(rMm, (r) => jTrue(CH[c], r, zMm[k]));
    const P = matvec(L, f);
    let m = 0; for (const v of P) m = Math.max(m, v);
    R.push(P.map((v) => v + 1e-3 * m * rnd())); Lh.push(P.map((v) => v + 1e-3 * m * rnd()));
  }
  right.push(R); left.push(Lh);
}
const prep = (): Prepared => {
  const sigma = [0, 1, 2].map((c) => Array.from({ length: nz }, (_, k) => { let s = 0; for (let i = 0; i < nr; i++) s += (right[c][k][i] - left[c][k][i]) ** 2; return Math.sqrt(s / nr / 2); }));
  return { nz, nr, drMm, zMm, rMm, right, left, satRows: [], axis: 0, axisInt: 0, axisCodes: [], axisSnr: 0, edgeRatio: 0, halvesRms: [], sigma, oriented: { w: 0, h: 0 }, rows: [0, 0], preview: { w: 0, h: 0, step: 1, v: new Float32Array(0) }, unitAbs: true };
};
const p = prep();
const tfsO = (x: Partial<TfsOpts> = {}): TfsOpts => ({ path: 'ratio', pair: 'RG', emLaw: { kind: 'group' }, sigFrac: 0.02, islandFrac: 0.05, chemBG: true, zCutMm: null, chemThr: 0.1, fsChannel: 'RG', refine: true, weights: [1, 1, 0], ...x });
const runSat = (sym: AbelRun['sym']): AbelRun => ({ method: 'sat', sym, lambdaRel: 0.1, segments: 12, satLambda: 1e-4 });
const same = (a: ArrayLike<number>, b: ArrayLike<number>) => { for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false; return a.length === b.length; };

// 1. the grid
{
  for (const f of [2, 4]) {
    const g = fineGrid(rMm, zMm, f, f);
    let okB = true;
    const idx = basePointsOnFine(g, nz, nr);
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) { const o = idx[k * nr + i], l = Math.floor(o / g.nr), s = o - l * g.nr; if (g.rMm[s] !== rMm[i] || g.zMm[l] !== zMm[k] || g.onR[s] !== i || g.onZ[l] !== k) okB = false; }
    check(`factor ${f}, every base point is on the fine grid, exactly`, okB);
    check(`factor ${f}, nothing past the last radius or the two end heights`, g.rMm[g.nr - 1] === rMm[nr - 1] && g.zMm[0] === zMm[0] && g.zMm[g.nz - 1] === zMm[nz - 1]);
    check(`factor ${f}, the first radius is the axis`, g.rMm[0] === 0 && g.axisNode && g.rMm[g.inner] === rMm[0]);
    check(`factor ${f}, even steps`, g.rMm.every((r, s) => Math.abs(r - (s * drMm) / f) < 1e-12));
    check(`factor ${f}, count of points`, finePoints(nr, nz, f, f) === g.nr * g.nz);
  }
  const g1 = fineGrid(rMm, zMm, 1, 1);
  check('factor 1 is the base grid', same(g1.rMm, rMm) && same(g1.zMm, zMm) && !g1.axisNode);
  const gs = fineGrid(rMm, zMm, 4, 1);
  check('SAT grid keeps the base heights', same(gs.zMm, zMm));
  const seen: number[][] = [];
  forNearest(fineGrid(Float64Array.of(0.5, 1.5), Float64Array.of(0, 1), 2, 2), 2, (q, c) => { seen[q] = [...c].sort((a, b) => a - b); });
  check('a point half way depends on all the cells around it', JSON.stringify(seen[4 + 2]) === JSON.stringify([0, 1, 2, 3]));
  check('factor from a log', refineFactor(4) === 4 && refineFactor(2) === 2 && refineFactor(3) === 1 && refineFactor(undefined) === 1);
}

// 2. the spline, without touching sat.ts
{
  const B = splineBasis(nr, 12);
  const g = fineGrid(rMm, zMm, 1, 1);
  const Bf = splineBasisAt(splineX(g, nr), 12);
  check('basis at the base points equals splineBasis of sat.ts, bit by bit', same(B.a, Bf.a));
  const g4 = fineGrid(rMm, zMm, 4, 1), B4 = splineBasisAt(splineX(g4, nr), 12);
  let ok = true;
  for (let s = 0; s < g4.nr; s++) if (g4.onR[s] >= 0) for (let q = 0; q < B.m; q++) if (B4.a[s * B.m + q] !== B.a[g4.onR[s] * B.m + q]) ok = false;
  check('basis at factor 4 on the base rays equals splineBasis, bit by bit', ok);
  const B0 = splineBasisAt([0, 1e-6], 12);
  check('the fine grid reaches the axis, where the spline has zero slope', B0.a[0] === 1 && Math.abs(B0.a[0] - B0.a[B0.m]) / 1e-6 < 1e-3);
}

// 3. the same path with the masks given reproduces the base results exactly
{
  const ab = runAbel(p, { method: 'tikhonov', sym: 'average', lambdaRel: 0.1, segments: 12, satLambda: 1e-4 });
  for (const [name, o] of [['ratio R/G', tfsO()], ['ratio B/G', tfsO({ pair: 'BG', fsChannel: 'G', weights: [1, 1, 1] })], ['three channels with blue', tfsO({ fsChannel: 'RGB', weights: [1, 1, 1] })], ['fit', tfsO({ path: 'fit3', weights: [1, 1, 1] })], ['fit with chemiluminescence', tfsO({ path: 'fit3', weights: [1, 1, 1], zCutMm: 13 })]] as [string, TfsOpts][]) {
    const a = runTfs(ab.J, p, band, o, undefined, table);
    const use = Uint8Array.from(a.mask, (m) => (m & 2 || m & 8 ? 0 : 1));
    const b = runTfs(ab.J, p, band, o, undefined, table, { use, chem: a.chem });
    check(`runTfs with the base masks, ${name}`, same(a.T, b.T) && same(a.fs, b.fs) && same(a.mask, b.mask));
  }
  const a = runTfs(ab.J, p, band, tfsO(), undefined, table), c = runTfs(ab.J, p, band, tfsO(), undefined, table);
  check('runTfs without masks is repeatable', same(a.T, c.T) && same(a.fs, c.fs));
}

const usable = (J: Float64Array[][], o = tfsO()) => runTfs(J, p, band, o, undefined, table);
const baseOf = (J: Float64Array[][], t: ReturnType<typeof usable>) => ({ T: t.T, fs: t.fs, mask: t.mask, chem: t.chem, J });
const tolOk = (c: { dTK: number; dFsRel: number; dJRel: number; mismatch: number }) => !c.mismatch && c.dTK <= REFINE_TOL_T_K && c.dFsRel <= REFINE_TOL_REL && c.dJRel <= REFINE_TOL_REL;
const report = (c: { dTK: number; dFsRel: number; dJRel: number; mismatch: number; nBase: number }) => `(${c.nBase} points, ΔT ${e(c.dTK)} K, f_s rel ${e(c.dFsRel)}, J rel ${e(c.dJRel)}, ${c.mismatch} mismatches)`;

// 4. SAT, one fit per height, so only the radius
for (const sym of ['both', 'average', 'right'] as const) for (const [pn, o] of [['ratio', tfsO()], ['fit', tfsO({ path: 'fit3', weights: [1, 1, 0] })]] as [string, TfsOpts][]) {
  const ab = runAbel(p, runSat(sym));
  const t = usable(ab.J, o);
  for (const f of [2, 4]) {
    const r = runRefine(p, band, { kind: 'sat', factor: f, run: runSat(sym), tfs: o, base: baseOf(ab.J, t) });
    check(`SAT ${sym} ${pn} factor ${f}, heights unchanged, base points agree`, r.nz === nz && r.fz === 1 && r.fr === f && tolOk(r.check), report(r.check));
    const passes = (() => { let n = 0; for (const v of r.onBase) n += v; return n; })();
    check(`SAT ${sym} ${pn} factor ${f}, ${nz * nr} base points are marked`, passes === nz * nr);
    if (sym === 'both' && pn === 'ratio') {
      // between rays the function is the spline, and it stays near the true field where there is signal
      let worst = 0, between = 0;
      for (let l = 0; l < r.nz; l++) for (let s = 0; s < r.nr; s++) {
        const q = l * r.nr + s;
        if (r.onBase[q] || !Number.isFinite(r.T[q]) || r.rMm[s] < 0.8 || r.rMm[s] > 3.4) continue;
        between++; worst = Math.max(worst, Math.abs(r.T[q] - Ttrue(r.rMm[s], r.zMm[l])));
      }
      check(`SAT factor ${f} between rays stays on the smooth field`, between > 0 && worst < 60, `(${between} points, worst ${worst.toFixed(1)} K)`);
    }
  }
}
// saturated heights stay out, and base results are not written
{
  const q: Prepared = { ...p, satRows: [3, 4] };
  const ab = runAbel(q, runSat('both'));
  const t = runTfs(ab.J, q, band, tfsO(), undefined, table);
  const snapshot = [t.T.slice(), t.fs.slice(), t.mask.slice()];
  const r = runRefine(q, band, { kind: 'sat', factor: 2, run: runSat('both'), tfs: tfsO(), base: baseOf(ab.J, t) });
  check('saturated heights stay empty on the fine grid', tolOk(r.check) && Array.from({ length: r.nr }, (_, s) => r.T[3 * r.nr + s]).every(Number.isNaN), report(r.check));
  check('base results are not written', same(snapshot[0], t.T) && same(snapshot[1], t.fs) && same(snapshot[2], t.mask));
  // a fine point next to a masked cell is not shown, whichever value the function has there
  let bad = 0;
  forNearest(fineGrid(q.rMm, q.zMm, 2, 1), nr, (o, cells) => { if (Number.isFinite(r.T[o]) && cells.some((c) => !Number.isFinite(t.T[c]))) bad++; });
  check('nothing is shown outside the zone of the base grid', bad === 0);
}

// 5. the emission network, with params kept after training, on a coarse training so the test is quick
const proj = (): { proj: Projections } => {
  const b = new Float32Array(nz * 2 * nr * 3);
  for (let k = 0; k < nz; k++) for (let c = 0; c < 3; c++) for (let i = 0; i < nr; i++) { b[(k * 2 * nr + nr + i) * 3 + c] = right[c][k][i]; b[(k * 2 * nr + nr - 1 - i) * 3 + c] = left[c][k][i]; }
  return { proj: { nt: 1, nz, nr, b, rMm, zMm: Array.from(zMm), times: [0] } };
};
{
  const a = proj();
  const prob = buildProblem(a.proj);
  const train = await trainWithEngine(new LocalEngine(prob), initXavier(7), { iterations: 400 });
  const res = finishStage1(prob, a.proj.b, { rMm: a.proj.rMm, zMm: a.proj.zMm, times: a.proj.times }, train, { absolute: true });
  const src = res.jSI!;
  const J: Float64Array[][] = [[], [], []];
  for (let c = 0; c < 3; c++) for (let k = 0; k < nz; k++) J[c].push(Float64Array.from({ length: nr }, (_, i) => src[(k * nr + i) * 3 + c]));
  const t = usable(J);
  const emission: EmissionNet = { params: train.params, norm: prob.norm, outScale: prob.outScale, bScale: prob.bScale };
  for (const f of [2, 4]) {
    const r = runRefine(p, band, { kind: 'annabel', factor: f, run: { ...runSat('both'), method: 'annabel' }, tfs: tfsO(), base: baseOf(J, t), emission });
    check(`emission network factor ${f}, radius and height, base points agree`, r.fr === f && r.fz === f && r.nz === (nz - 1) * f + 1 && tolOk(r.check), report(r.check));
  }

  // 6. the network for T and f_s, on that emission, with the shape of runNet in the page
  const Jf = new Float32Array(nz * nr * 3);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) for (let c = 0; c < 3; c++) Jf[(k * nr + i) * 3 + c] = J[c][k][i];
  const nT = t.table.T.length, K = new Float32Array(3 * nT);
  K.set(t.table.G!.R, 0); K.set(t.table.G!.G, nT); K.set(t.table.G!.B, 2 * nT);
  const kern: KernelTable = { Tgrid: t.table.T, K, chanW: [1, 1, 0], jToSi: 1, fsRef: 2e-5 };
  const grid = { nt: 1, nz, nr, rMm, zMm: Array.from(zMm), times: [0] };
  const pr2 = buildStage2Problem(Jf, grid, kern, { iterations: 300, lambdaRatio: 1 });
  const train2 = await trainStage2(new Stage2LocalEngine(pr2), initStage2(11, pr2.sizes), pr2, { iterations: 300 });
  const r2 = finishStage2(pr2, Jf, grid, kern, train2);
  const T2 = new Float64Array(N).fill(NaN), F2 = new Float64Array(N).fill(NaN), m2 = new Uint8Array(N);
  for (let q = 0; q < N; q++) if (!(t.mask[q] & 10) && r2.mask[q]) { T2[q] = r2.T[q]; F2[q] = r2.fs[q]; m2[q] = 1; } else m2[q] = 2;
  const net: TfsNet = { params: train2.params, norm: pr2.norm, sizes: pr2.sizes, Tlo: pr2.Tlo, Thi: pr2.Thi, fsRef: pr2.fsRef };
  const base2 = { T: T2, fs: F2, mask: m2, chem: new Uint8Array(N), J };
  for (const f of [2, 4]) {
    const r = runRefine(p, band, { kind: 'net', factor: f, run: { ...runSat('both'), method: 'tikhonov' }, tfs: tfsO(), base: base2, net });
    check(`T and f_s network factor ${f}, radius and height, base points agree`, r.fr === f && r.fz === f && tolOk(r.check) && r.J === null, report(r.check));
    let out = 0, bad = 0;
    forNearest(fineGrid(rMm, zMm, f, f), nr, (o, cells) => { if (Number.isFinite(r.T[o])) { if (cells.some((c) => !Number.isFinite(T2[c]))) out++; if (r.T[o] < pr2.Tlo || r.T[o] > pr2.Thi) bad++; } });
    check(`T and f_s network factor ${f}, nothing outside the zone of the base grid`, out === 0 && bad === 0);
    const r3 = runRefine(p, band, { kind: 'net', factor: f, run: { ...runSat('both'), method: 'annabel' }, tfs: tfsO(), base: base2, net, emission });
    check(`T and f_s network with the emission network, emission also agrees at factor ${f}`, r3.J !== null && tolOk(r3.check), report(r3.check));
  }
  // a network is not evaluated past its training range, which saturated rows at an end can shorten
  const zShort = { ...net, norm: { ...net.norm, z_max: zMm[nz - 3] } };
  const rs = runRefine(p, band, { kind: 'net', factor: 2, run: { ...runSat('both'), method: 'tikhonov' }, tfs: tfsO(), base: base2, net: zShort });
  let beyond = 0;
  for (let l = 1; l < rs.nz; l += 2) if (rs.zMm[l] > zMm[nz - 3]) for (let s = 0; s < rs.nr; s++) if (Number.isFinite(rs.T[l * rs.nr + s])) beyond++;
  check('a network stops at the end of its training range', beyond === 0);
}

// 7. the mask
{
  const fs = Float64Array.of(4.25, 0.2, 0.3, NaN, 9, 0.2125);
  const mask = Uint8Array.of(1, 1, 1, 1, 8, 1);
  const copy = fs.slice();
  const m = fsMaskOf(fs, mask, 5)!;
  check('maximum from the zone used, not from a cell outside it', m.max === 4.25);
  check('5 % of 4,25 ppm is 0,2125 ppm', Math.abs(m.thr - 0.2125) < 1e-12);
  check('points pass strictly above the threshold', m.pass[0] === 1 && m.pass[1] === 0 && m.pass[2] === 1 && m.pass[5] === 0);
  check('count of points in the zone and of those that remain', m.nZone === 4 && m.nPass === 2);
  check('a NaN does not pass', m.pass[3] === 0);
  check('f_s is not written', same(copy, fs));
  check('no positive f_s gives no mask', fsMaskOf(Float64Array.of(-1, NaN), Uint8Array.of(1, 1), 5) === null);
  check('range of the percentage', validMaskPct(5) && validMaskPct(100) && !validMaskPct(0) && !validMaskPct(101) && !validMaskPct(NaN) && !validMaskPct(undefined));
  check('runs of a profile', JSON.stringify(runsOf(6, (i) => i !== 2 && i !== 3)) === JSON.stringify([[0, 2], [4, 6]]));
}

// 8. the band does what the list of step 5 says
{
  const run: AbelRun = { method: 'tikhonov', sym: 'average', lambdaRel: 0.1, segments: 12, satLambda: 1e-4 };
  const base = { n: 12, seed: 3, sigmaT: 5, Tcal: 1323, eps: 1, run, tfs: tfsO({ refine: false }), absolute: true, sKrel: [0.002, 0.002, 0.002] as [number, number, number], scaleRel: 0.01 };
  const b0 = runBand(p, band, base);
  check('with the emission network it computes with onion peeling with smoothing', same(runBand(p, band, { ...base, run: { ...run, method: 'annabel', sym: 'both' } }).Tlo, b0.Tlo));
  check('with SAT and both halves it keeps SAT, the same band as SAT with the mean of the halves', same(runBand(p, band, { ...base, run: { ...run, method: 'sat', sym: 'both' } }).Tlo, runBand(p, band, { ...base, run: { ...run, method: 'sat', sym: 'average' } }).Tlo));
  const a1 = runAbel(p, { ...run, method: 'sat', sym: 'both' }), a2 = runAbel(p, { ...run, method: 'sat', sym: 'average' });
  const t1 = runTfs(a1.J, p, band, tfsO({ refine: false })), t2 = runTfs(a2.J, p, band, tfsO({ refine: false }));
  let dT = 0, dF = 0; for (let q = 0; q < t1.T.length; q++) { if (Number.isFinite(t1.T[q]) && Number.isFinite(t2.T[q])) dT = Math.max(dT, Math.abs(t1.T[q] - t2.T[q])); if (t2.fs[q] > 0) dF = Math.max(dF, Math.abs(t1.fs[q] / t2.fs[q] - 1)); }
  check('SAT with both halves in the fit gives the T and f_s of SAT with the mean of the halves', dT < 1e-6 && dF < 1e-9, `${e(dT)} K, ${e(dF)}`);
  check('with SAT and the mean of the halves it does use SAT', !same(runBand(p, band, { ...base, run: { ...run, method: 'sat', sym: 'average' } }).Tlo, b0.Tlo));
  check('it uses the mean of the halves when one half was chosen', same(runBand(p, band, { ...base, run: { ...run, sym: 'right' } }).Tlo, b0.Tlo));
  const quiet: Prepared = { ...p, sigma: p.sigma.map((c) => c.map(() => 0)) };
  const spread = (lo: Float64Array, hi: Float64Array) => { let m = 0; for (let q = 0; q < lo.length; q++) if (Number.isFinite(lo[q])) m = Math.max(m, hi[q] - lo[q]); return m; };
  const none = { ...base, sigmaT: 0, sKrel: [0, 0, 0] as [number, number, number], scaleRel: 0 };
  const rel = (lo: Float64Array, hi: Float64Array) => { let m = 0; for (let q = 0; q < lo.length; q++) if (lo[q] > 0) m = Math.max(m, hi[q] / lo[q] - 1); return m; };
  const b1 = runBand(quiet, band, none);
  check('with no source the band has no width', spread(b1.Tlo, b1.Thi) < 1e-6 && rel(b1.fslo, b1.fshi) < 1e-9);
  const b2 = runBand(quiet, band, { ...none, scaleRel: 0.01 });
  check('the ruler moves f_s and not T', spread(b2.Tlo, b2.Thi) < 1e-6 && rel(b2.fslo, b2.fshi) > 1e-3);
  const b3 = runBand(quiet, band, { ...none, sigmaT: 5 });
  check('the furnace error moves T', spread(b3.Tlo, b3.Thi) > 0.1);
  const b4 = runBand(quiet, band, { ...none, sigmaT: 5, Tcal: NaN });
  check('without a temperature of calibration the furnace error does not enter', spread(b4.Tlo, b4.Thi) < 1e-6);
  const b5 = runBand(quiet, band, { ...none, sKrel: [0.01, 0, 0] });
  check('the statistical error of K moves T', spread(b5.Tlo, b5.Thi) > 0.1);
  const b6 = runBand(p, band, none);
  check('the noise of the two halves moves T', spread(b6.Tlo, b6.Thi) > 0.1);
}

console.log(fails ? `${fails} FAILED` : 'All checks passed.');
process.exit(fails ? 1 : 0);
