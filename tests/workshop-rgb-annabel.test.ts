// ANNAbel stage 1 parity with the app (PyTorch), levels 1 and 2 of §11.2, the short training
// trajectory, and the per-mm → W m⁻³ sr⁻¹ conversion. Run with
// `node --experimental-strip-types tests/workshop-rgb-annabel.test.ts`.
// Fixture written by manim/tools/make_annabel_fixtures.py from the real annabel_app/core.
import { readFileSync } from 'node:fs';
import { opMatrix, buildProblem, coordNormFromGrid, type Projections } from '../src/lib/workshop-rgb/annabel/problem.ts';
import { layout, predictNorm, SIZES } from '../src/lib/workshop-rgb/annabel/net.ts';
import { LocalEngine, trainWithEngine, finishStage1, cosineLr, channelPlanes, relL2 } from '../src/lib/workshop-rgb/annabel/train.ts';
import { emissionPerMmToSI, emissionSIToPerMm, PER_MM_TO_PER_M } from '../src/lib/workshop-rgb/annabel/units.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

type Enc = { f32?: string; f64?: string };
const dec = (e: Enc): Float64Array => {
  if (e.f32) { const b = Buffer.from(e.f32, 'base64'); return Float64Array.from(new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4)); }
  const b = Buffer.from(e.f64 as string, 'base64');
  return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
};
const dec32 = (e: Enc) => Float32Array.from(dec(e));
/** max |a − b| / max |b|, the relative measure of the prototype and of WEB_SPEC §9.4 */
const rel = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  if (a.length !== b.length) return Infinity;
  let d = 0, m = 0;
  for (let i = 0; i < a.length; i++) { d = Math.max(d, Math.abs(a[i] - b[i])); m = Math.max(m, Math.abs(b[i])); }
  return m > 0 ? d / m : d;
};
const relL2v = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let n = 0, d = 0;
  for (let i = 0; i < a.length; i++) { n += (a[i] - b[i]) ** 2; d += b[i] ** 2; }
  return Math.sqrt(n / d);
};
const e = (x: number) => x.toExponential(1);

const fx = JSON.parse(readFileSync(new URL('./fixtures/workshop-rgb/annabel_weights.json', import.meta.url), 'utf8'));
const T1: number = fx.tolerance.level1, T2: number = fx.tolerance.level2;

for (const c of fx.cases) {
  const proj: Projections = { nt: c.nt, nz: c.nz, nr: c.nr, b: dec32(c.input.b), rMm: dec(c.input.r), zMm: dec(c.input.z), times: dec(c.input.times) };
  const prob = buildProblem(proj);
  const l1 = c.level1;
  const tag = `[${c.name}: Nt ${c.nt}, Nz ${c.nz}, Nr ${c.nr}]`;

  // ---- level 1, operators and normalisations
  if (l1.A_f64) {
    const A64 = opMatrix(c.nr, prob.dr);
    check(`L1 op_matrix float64, rays at (i − ½)dr ${tag}`, rel(A64, dec(l1.A_f64)) <= 1e-12, `(${e(rel(A64, dec(l1.A_f64)))})`);
  }
  const checks1: [string, ArrayLike<number>, ArrayLike<number>][] = [
    ['A float32', prob.A, dec(l1.A)], ['X = (r̃², z̃, t̃)', prob.X, dec(l1.X)], ['b_R / b_scale', prob.bR, dec(l1.bR)],
    ['b_L mirrored / b_scale', prob.bL, dec(l1.bL)], ['nrm per channel', prob.nrm, dec(l1.nrm)],
    ['b_scale', [prob.bScale], [l1.b_scale]], ['out_scale', [prob.outScale], [l1.out_scale]],
    ['CoordNorm', Object.values(prob.norm), [l1.norm.r_max, l1.norm.z_min, l1.norm.z_max, l1.norm.t_max]],
  ];
  for (const [n, a, b] of checks1) { const r = rel(a, b); check(`L1 ${n} ${tag}`, r <= T1, `(${e(r)})`); }
  if (c.nt === 1) check(`L1 t_max guarded to 1 for one frame ${tag}`, prob.norm.t_max === 1 && prob.X[2] === 0);
  if (c.name === 't1') check(`L1 all-zero channel gets nrm = 1 ${tag}`, prob.nrm[2] === 1);

  if (!c.level2) continue;
  // ---- level 2, loss and gradient for the PyTorch weights
  const l2 = c.level2;
  const p0 = dec32(l2.p0);
  check(`L2 parameter count, sizes ${SIZES.join('-')} ${tag}`, p0.length === layout().nParams && c.sizes.join() === SIZES.join());
  const eng = new LocalEngine(prob, 3);
  const r = await eng.evaluate(p0, true);
  const g = r.grad as Float64Array;
  const lr = Math.abs(r.loss - l2.loss) / l2.loss;
  check(`L2 loss 0.5 (l_R + l_L) ${tag}`, lr <= T2, `(${r.loss.toPrecision(7)} vs ${Number(l2.loss).toPrecision(7)}, ${e(lr)})`);
  check(`L2 l_R and l_L separately ${tag}`, Math.abs(r.lR - l2.proj_right) / l2.proj_right <= T2 && Math.abs(r.lL - l2.proj_left) / l2.proj_left <= T2);
  const gt = dec(l2.grad);
  check(`L2 gradient, relative L2 ${tag}`, relL2v(g, gt) <= T2, `(${e(relL2v(g, gt))})`);
  check(`L2 gradient, max |Δ| / max |g| ${tag}`, rel(g, gt) <= T2, `(${e(rel(g, gt))})`);
  const Jn = predictNorm(p0, prob, 2);
  check(`L2 Jn = out_scale softplus(net(x)) ${tag}`, rel(Jn, dec(l2.Jn)) <= T2, `(${e(rel(Jn, dec(l2.Jn)))})`);

  // ---- short training with the app's loop (Adam, cosine to 0.1 lr, best before the step)
  const tr = c.train;
  const res = await trainWithEngine(new LocalEngine(prob, 4), p0, { iterations: tr.iterations, lr: tr.lr, lrMinFrac: tr.lr_min_frac });
  const hist = dec(tr.history);
  check(`train loss history over ${tr.iterations} iterations ${tag}`, rel(res.history, hist) <= T2, `(${e(rel(res.history, hist))})`);
  check(`train best checkpoint iteration ${tag}`, res.bestIter === tr.best_iter, `(${res.bestIter} vs ${tr.best_iter})`);
  check(`train best parameters ${tag}`, relL2v(res.params, dec(tr.p_best)) <= T2, `(${e(relL2v(res.params, dec(tr.p_best)))})`);

  // ---- stage-1 output in the app's units and in W m⁻³ sr⁻¹
  const out = finishStage1(prob, proj.b, proj, { ...res, params: p0 }, { absolute: true });
  check(`J per mm = Jn · b_scale ${tag}`, rel(out.jPerMm, dec(l2.J_per_mm)) <= T2, `(${e(rel(out.jPerMm, dec(l2.J_per_mm)))})`);
  check(`j in W m⁻³ sr⁻¹ = 1000 · J per mm ${tag}`, rel(out.jSI as Float32Array, Float64Array.from(dec(l2.J_per_mm), (v) => v * 1e3)) <= T2);
  const pl = channelPlanes(out, 1);
  check(`channel planes (Nz, Nr) of time 1 ${tag}`, pl.G.length === c.nz * c.nr && pl.G[5] === (out.jSI as Float32Array)[(c.nz * c.nr + 5) * 3 + 1]);
}

// ---- cosine schedule, eta_min = 0.1 lr at the end, lr at epoch 0
check('cosine lr: lr at 0, 0.55 lr at half, 0.1 lr at T_max', cosineLr(0, 100) === 1e-3 && Math.abs(cosineLr(50, 100) - 5.5e-4) < 1e-18 && Math.abs(cosineLr(100, 100) - 1e-4) < 1e-18);

// ---- units: emission per mm → W m⁻³ sr⁻¹ (§11.2), against a closed form
{
  // uniform disc of radius R = nr·dr with j0 in W m⁻³ sr⁻¹: b(y) = 2 j0 √(R² − y²) in W m⁻² sr⁻¹,
  // y in metres; the rings of op_matrix are exact for a piecewise-constant field
  const nr = 40, drMm = 0.05, j0 = 2.4e4;
  const A = opMatrix(nr, drMm);                                 // mm
  const Jmm = new Float64Array(nr).fill(j0 / PER_MM_TO_PER_M);  // per mm
  let worst = 0;
  for (let i = 0; i < nr; i++) {
    let b = 0; for (let j = i; j < nr; j++) b += A[i * nr + j] * Jmm[j];
    const y = (i + 0.5) * drMm * 1e-3, R = nr * drMm * 1e-3;
    worst = Math.max(worst, Math.abs(b - 2 * j0 * Math.sqrt(R * R - y * y)) / (2 * j0 * R));
  }
  check('units: A in mm with J per mm gives b in W m⁻² sr⁻¹ of the SI disc', worst < 1e-12, `(${e(worst)})`);
  // back-substitution of that b recovers J per mm, and the conversion gives j0 back
  const b = new Float64Array(nr);
  for (let i = 0; i < nr; i++) { const y = (i + 0.5) * drMm * 1e-3, R = nr * drMm * 1e-3; b[i] = 2 * j0 * Math.sqrt(R * R - y * y); }
  const J = new Float64Array(nr);
  for (let i = nr - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j < nr; j++) s -= A[i * nr + j] * J[j]; J[i] = s / A[i * nr + i]; }
  const jSI = emissionPerMmToSI(J);
  check('units: emissionPerMmToSI(onion J per mm) = j0 in W m⁻³ sr⁻¹', rel(jSI, new Float64Array(nr).fill(j0)) < 1e-6, `(${e(rel(jSI, new Float64Array(nr).fill(j0)))})`);
  check('units: SI → per mm → SI round trip', rel(emissionPerMmToSI(emissionSIToPerMm([j0, 1, 3e5])), [j0, 1, 3e5]) < 1e-7);
}

// ---- input guards
{
  let threw = false;
  try { coordNormFromGrid([0.05, 0.15], [1, 2], [-1]); } catch { threw = true; }
  check('negative acquisition times are rejected', threw);
  check('relL2 per channel', relL2([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 6]).join() === [0, 0, Math.sqrt(9 / 45)].join());
}

console.log(fails ? `\n${fails} FAILED` : '\nALL PASS');
if (fails) process.exit(1);
