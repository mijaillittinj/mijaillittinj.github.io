// RGB-ANNAbel stage 2 parity with the app (PyTorch): problem setup, loss and gradient for fixed
// initial weights, a short Adam trajectory, the full-grid result and the closed-form cross-check
// (_xcheck). CPU path and the Worker code path (the same stage2LossGrad by blocks of rows), no GPU.
// Run with `node --experimental-strip-types tests/workshop-rgb-stage2.test.ts`.
// Fixture written by manim/tools/make_stage2_fixtures.py from fuentes/rgb-annabel/annabel_app/core.
import { readFileSync } from 'node:fs';
import { buildStage2Problem, Stage2LocalEngine, Mlp, stage2LossGrad, stage2Loss, trainStage2, finishStage2, stage2Sizes, STAGE2_DEFAULTS, initStage2, type KernelTable } from '../src/lib/workshop-rgb/annabel/stage2.ts';
import { layout } from '../src/lib/workshop-rgb/annabel/net.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

type Enc = { f32?: string; f64?: string };
const dec = (e: Enc): Float64Array => {
  if (e.f32) { const b = Buffer.from(e.f32, 'base64'); return Float64Array.from(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)); }
  const b = Buffer.from(e.f64 as string, 'base64');
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const dec32 = (e: Enc) => Float32Array.from(dec(e));
const rel = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  if (a.length !== b.length) return Infinity;
  let d = 0, m = 0;
  for (let i = 0; i < a.length; i++) { d = Math.max(d, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(b[i])); }
  return m > 0 ? d / m : d;
};
const relL2 = (a: ArrayLike<number>, b: ArrayLike<number>) => { let n = 0, d = 0; for (let i = 0; i < a.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; } return Math.sqrt(n / d); };
const rs = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
const e = (x: number) => x.toExponential(1);
const nanEq = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  let worst = 0, mism = 0;
  for (let i = 0; i < a.length; i++) {
    const na = !Number.isFinite(a[i]) || a[i] === -1, nb = b[i] === -1;
    if (na !== nb) { mism++; continue; }
    if (!na) worst = Math.max(worst, Math.abs(a[i] - b[i]) / Math.max(Math.abs(b[i]), 1e-300));
  }
  return { ok: mism === 0 && worst <= tol, worst, mism };
};

const fx = JSON.parse(readFileSync(new URL('./fixtures/workshop-rgb/annabel_stage2.json', import.meta.url), 'utf8'));
const T1: number = fx.tolerance.level1, T2: number = fx.tolerance.level2;
const p0ByName: Record<string, Float32Array> = {};

check('Stage2Config defaults (4 × 64 tanh, no FF, 3000 it, lr 1e-3, 0.1, z_sub 2, 0.02, 1e-2, seed 42)',
  JSON.stringify(stage2Sizes()) === '[3,64,64,64,64,2]' && STAGE2_DEFAULTS.iterations === 3000 && STAGE2_DEFAULTS.lr === 1e-3 && STAGE2_DEFAULTS.lrMinFrac === 0.1
  && STAGE2_DEFAULTS.zSub === 2 && STAGE2_DEFAULTS.signalThreshold === 0.02 && STAGE2_DEFAULTS.lambdaRatio === 1e-2 && STAGE2_DEFAULTS.seed === 42);

for (const c of fx.cases) {
  const tag = `[${c.name}: Nt ${c.nt}, Nz ${c.nz}, Nr ${c.nr}, chan_w ${c.kernels.chan_w.join(',')}]`;
  const cfgPy = c.config;
  check(`fixture config is the app default except iterations ${tag}`, cfgPy.z_sub === 2 && cfgPy.signal_threshold === 0.02 && cfgPy.lambda_ratio === 0.01
    && cfgPy.net.width === 64 && cfgPy.net.hidden_layers === 4 && cfgPy.net.activation === 'tanh' && cfgPy.net.ff_r.length === 0 && cfgPy.train.lr === 1e-3);
  const kern: KernelTable = { Tgrid: dec(c.kernels.T_grid), K: dec32(c.kernels.K), chanW: c.kernels.chan_w, jToSi: c.kernels.j_to_si, fsRef: c.kernels.fs_ref };
  const J = dec32(c.input.J);
  const grid = { nt: c.nt, nz: c.nz, nr: c.nr, rMm: dec(c.input.r), zMm: dec(c.input.z), times: dec(c.input.times) };
  const prob = buildStage2Problem(J, grid, kern, { iterations: c.config.train.iterations });
  const l1 = c.level1;
  // ---- level 1, problem setup
  const sig = Array.from(prob.mask), xm = Array.from(prob.xmask);
  check(`L1 X2 = (r̃², z̃, t̃) on z[::2] ${tag}`, rel(prob.X, dec(l1.X2)) <= T1, `(${e(rel(prob.X, dec(l1.X2)))})`);
  check(`L1 Y2 = J · j_to_si ${tag}`, rel(prob.Y, dec(l1.Y2)) <= T1, `(${e(rel(prob.Y, dec(l1.Y2)))})`);
  check(`L1 nrm2 ${tag}`, rel(prob.nrm2, dec(l1.nrm2)) <= T1);
  check(`L1 signal mask, ${l1.n_tr} pixels ${tag}`, prob.nTr === l1.n_tr && sig.every((v, i) => v === l1.sig_mask[i]));
  check(`L1 ratio pairs ${JSON.stringify(l1.ratio_pairs)} ${tag}`, JSON.stringify(prob.pairs) === JSON.stringify(l1.ratio_pairs));
  check(`L1 ratio data and their norms ${tag}`, prob.ratios.every((r, q) => rel(r, dec(l1.ratios_data[q])) <= T1 && rs(prob.ratioNrm[q], l1.ratios_nrm[q]) <= T1));
  const tcf = nanEq(prob.Tcf, Array.from(dec(l1.T_cf), (v) => (Number.isFinite(v) ? v : -1)), T1);
  check(`L1 closed form T_cf (solve_T_fs) on the training grid ${tag}`, tcf.ok, `(${e(tcf.worst)}, ${tcf.mism} NaN mismatches)`);
  const fcf = nanEq(prob.fsCf, Array.from(dec(l1.fs_cf), (v) => (Number.isFinite(v) ? v : -1)), 1e-5);
  check(`L1 closed form fs_cf ${tag}`, fcf.ok, `(${e(fcf.worst)})`);
  check(`L1 cross-check mask ${xm.filter(Boolean).length} pixels ${tag}`, xm.every((v, i) => v === l1.xmask[i]));
  check(`L1 CoordNorm ${tag}`, rel(Object.values(prob.norm), [l1.norm.r_max, l1.norm.z_min, l1.norm.z_max, l1.norm.t_max]) <= T1);
  // ---- level 2, loss and gradient for the PyTorch initial weights
  const l2 = c.level2;
  const p0 = typeof l2.p0 === 'string' ? p0ByName[l2.p0] : dec32(l2.p0);
  p0ByName[c.name] = p0;
  check(`L2 parameter count ${layout(prob.sizes).nParams} ${tag}`, p0.length === layout(prob.sizes).nParams);
  const eng = new Stage2LocalEngine(prob, 3);
  const ev = await eng.evaluate(p0, true);
  check(`L2 loss ${tag}`, rs(ev.loss, l2.loss) <= T2, `(${e(rs(ev.loss, l2.loss))}, ${ev.loss} vs ${l2.loss})`);
  check(`L2 direct term ${tag}`, rs(ev.lR, l2.direct) <= T2, `(${e(rs(ev.lR, l2.direct))})`);
  check(`L2 ratio term ${tag}`, rs(ev.lL, l2.ratio) <= T2, `(${e(rs(ev.lL, l2.ratio))})`);
  const g = dec(l2.grad);
  check(`L2 gradient, relative L2 and max ${tag}`, relL2(ev.grad!, g) <= T2 && rel(ev.grad!, g) <= T2, `(${e(relL2(ev.grad!, g))}, ${e(rel(ev.grad!, g))})`);
  // Worker code path: one helper per block of rows, sums added by the host (workers.ts)
  {
    const net = new Mlp(2 * prob.nr, prob.sizes);
    const grad = new Float64Array(p0.length);
    let sD = 0, sQ = 0;
    for (let a = 0; a < prob.rows; a += 2) { const s = stage2LossGrad(net, p0, prob, a, Math.min(prob.rows, a + 2), grad); sD += s.sD; sQ += s.sQ; }
    const l = stage2Loss(prob, { sD, sQ });
    check(`L2 blocks of 2 rows (Worker split) equal one pass ${tag}`, rs(l.loss, ev.loss) <= 1e-12 && relL2(grad, ev.grad!) <= 1e-10, `(${e(rs(l.loss, ev.loss))}, ${e(relL2(grad, ev.grad!))})`);
  }
  if (!c.train) continue;
  // ---- short training, Adam + cosine, best before the step
  const tr = c.train;
  const res = await trainStage2(new Stage2LocalEngine(prob), p0, prob, { iterations: tr.iterations });
  const hist = dec(tr.history);
  check(`T ${tr.iterations} iterations, loss history ${tag}`, rel(res.history, hist) <= T2, `(${e(rel(res.history, hist))})`);
  check(`T best iteration ${res.bestIter} = ${tr.best_iter} ${tag}`, res.bestIter === tr.best_iter && rs(res.bestLoss, tr.best_loss) <= T2);
  // parameters after N steps (a separate run that keeps the last state)
  {
    const pLast = p0.slice(), pPrev = p0.slice();
    const { Adam } = await import('../src/lib/workshop-rgb/annabel/train.ts');
    const opt = new Adam(pLast, tr.iterations, 1e-3, 0.1);
    const en = new Stage2LocalEngine(prob);
    for (let it = 0; it < tr.iterations; it++) { if (it === tr.iterations - 1) pPrev.set(pLast); const r = await en.evaluate(pLast, true); opt.step(pLast, r.grad!, r.loss); }
    const a = relL2(pLast, dec(tr.p_last)), b = relL2(pPrev, dec(tr.p_prev));
    const dl = relL2(Float64Array.from(pLast, (v, i) => v - p0[i]), Float64Array.from(dec(tr.p_last), (v, i) => v - p0[i]));
    check(`T parameters after ${tr.iterations} steps, relative L2 ${tag}`, a <= T2 && b <= T2, `(${e(a)}; of the change p − p0 ${e(dl)})`);
    const best = tr.best_iter === 1 ? p0 : tr.best_iter === tr.iterations ? pPrev : null;
    check(`T best checkpoint = parameters before step ${tr.best_iter} ${tag}`, !!best && relL2(res.params, best) <= 1e-12);
  }
  // ---- full grid, display mask, closed form, _xcheck and bound_report
  const out = finishStage2(prob, J, grid, kern, res);
  check(`R T on the full grid ${tag}`, rel(out.T, dec(tr.T)) <= T2, `(${e(rel(out.T, dec(tr.T)))})`);
  check(`R fs on the full grid ${tag}`, rel(out.fs, dec(tr.fs)) <= T2, `(${e(rel(out.fs, dec(tr.fs)))})`);
  check(`R display mask ${tag}`, Array.from(out.mask).every((v, i) => v === tr.mask[i]));
  const a1 = nanEq(out.Tcf, dec(tr.T_cf), T1), a2 = nanEq(out.fsCf, dec(tr.fs_cf), 1e-5);
  check(`R closed form on the full grid, bounds blanked ${tag}`, a1.ok && a2.ok, `(${e(a1.worst)}, ${e(a2.worst)})`);
  const x = out.xcheck!, xp = tr.xcheck;
  const xr = [rs(x.TMedianDiffK, xp.T_median_diff_K), rs(x.TMedianAbsDiffK, xp.T_median_abs_diff_K), rs(x.TP90AbsDiffK, xp.T_p90_abs_diff_K), rs(x.fsMedianRatio, xp.fs_median_ratio), rs(x.fsMedianAbsLogRatio, xp.fs_median_abs_log_ratio)];
  check(`R _xcheck medians, p90 and f_s ratio, ${x.nPixels} pixels, agrees ${x.agrees} ${tag}`, x.nPixels === xp.n_pixels && x.agrees === xp.agrees && Math.max(...xr) <= 1e-3, `(${e(Math.max(...xr))})`);
  const b = out.bounds!, bp = tr.bound_report;
  check(`R bound_report ${tag}`, b.fracTAtLower === bp.frac_T_at_lower && b.fracTAtUpper === bp.frac_T_at_upper && b.fracFsAtUpper === bp.frac_fs_at_upper
    && rs(b.TMedian, bp.T_median) <= T2 && rs(b.TP02, bp.T_p02) <= T2 && rs(b.TP98, bp.T_p98) <= T2);
}

// errors of the app
{
  const c = fx.cases[1];
  const kern: KernelTable = { Tgrid: dec(c.kernels.T_grid), K: dec32(c.kernels.K), chanW: [1, 0, 0], jToSi: 1e3, fsRef: 2e-5 };
  let msg = '';
  try { buildStage2Problem(dec32(c.input.J), { nt: c.nt, nz: c.nz, nr: c.nr, rMm: dec(c.input.r), zMm: dec(c.input.z), times: dec(c.input.times) }, kern); } catch (err) { msg = (err as Error).message; }
  check('one channel is refused (stage2_two_channels)', msg === 'stage2_two_channels');
}
check('initial weights are Xavier normal with zero biases', (() => { const p = initStage2(); const lo = layout(stage2Sizes()); return p.length === lo.nParams && p[lo.bOff[0]] === 0 && p[0] !== 0; })());

console.log(fails ? `\n${fails} FAIL` : '\nall PASS');
if (fails) process.exit(1);
