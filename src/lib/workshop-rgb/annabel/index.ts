// ANNAbel stage 1 in the browser (Escudero et al., Proc. Combust. Inst. 40 (2024) 105493),
// ported from annabel_app/core (abel.py, models.py, stage1.py). T and f_s come by default from
// the direct fit solve_T_fs of the core (radiometry), which takes the channel planes of
// `channelPlanes(result)` in W m⁻³ sr⁻¹. The stage-2 network of RGB-ANNAbel (Magaña et al.,
// Case Stud. Therm. Eng. 76 (2025) 107396) is the option of stage2.ts, with runStage2 below.
// The ensemble (stage 3) and the relative mode are not ported.
//
// Paths. 'webgpu' runs the whole iteration on the GPU (gpu.ts). 'workers' computes loss and
// gradient in a pool of Web Workers with Adam on the calling thread (workers.ts). 'cpu' is the
// same in one thread (Node tests, or inside a Worker). All in Float32 by batches.
import type { Projections, Stage1Problem } from './problem.ts';
import { buildProblem } from './problem.ts';
import { initXavier, layout } from './net.ts';
import type { GradEngine, Stage1Result, TrainOptions, TrainResult } from './train.ts';
import { LocalEngine, finishStage1, trainWithEngine } from './train.ts';
import { WorkerPoolEngine } from './workers.ts';
import { createGpuStage1, webgpuAvailable } from './gpu.ts';
import type { KernelTable, Stage2Config, Stage2Result } from './stage2.ts';
import { buildStage2Problem, Stage2LocalEngine, finishStage2, initStage2, trainStage2 } from './stage2.ts';
import { Stage2PoolEngine } from './workers.ts';
import { createGpuStage2 } from './gpu2.ts';

export type { Projections, Stage1Problem } from './problem.ts';
export type { Stage1Result, TrainResult, TrainTick, TrainOptions } from './train.ts';
export { opMatrix, splitHalves, coordNormFromGrid, coordFeatures, buildProblem } from './problem.ts';
export { SIZES, layout, initXavier, Net, predictNorm } from './net.ts';
export { Adam, cosineLr, LocalEngine, trainWithEngine, finishStage1, channelPlanes, jFrame, relL2 } from './train.ts';
export { emissionPerMmToSI, emissionSIToPerMm, PER_MM_TO_PER_M } from './units.ts';
export { WorkerPoolEngine, helperCount } from './workers.ts';
export { createGpuStage1, webgpuAvailable } from './gpu.ts';
export { buildStage2Problem, Stage2LocalEngine, Mlp, stage2LossGrad, stage2Loss, finishStage2, initStage2, trainStage2, predictTfs, xcheck, displayMask, boundReport,
         stage2Sizes, STAGE2_DEFAULTS, XCHECK_WARN_K, XCHECK_WARN_P90_K, XCHECK_WARN_FS_LOG } from './stage2.ts';
export type { KernelTable, Stage2Config, Stage2Problem, Stage2Result, XCheck, BoundReport } from './stage2.ts';
export { Stage2PoolEngine } from './workers.ts';
export { createGpuStage2 } from './gpu2.ts';

/** App presets (EXACTITUD.md): 4000 iterations is the quick demo (25 to 55 % error), 30 000 the
 *  long run (3 to 5 %). The tool offers only the long run for results. */
export const ITER_QUICK = 4000;
export const ITER_LONG = 30000;

export type Stage1Path = 'webgpu' | 'workers' | 'cpu';

export interface Stage1Options extends Omit<TrainOptions, 'iterations'> {
  iterations?: number;            // ITER_LONG
  seed?: number;                  // initial weights (Xavier normal, mulberry32; not torch's generator)
  path?: Stage1Path | 'auto';     // auto: webgpu if available, else workers
  absolute?: boolean;             // b in W m⁻² sr⁻¹, so j comes out in W m⁻³ sr⁻¹
  jTruePerMm?: ArrayLike<number> | null;
  workers?: { n?: number; rowsPerChunk?: number; factory?: () => Worker };
  itersPerSubmit?: number;        // WebGPU
}

export async function choosePath(p: Stage1Options['path']): Promise<Stage1Path> {
  if (p && p !== 'auto') return p;
  if (await webgpuAvailable()) return 'webgpu';
  return typeof Worker !== 'undefined' ? 'workers' : 'cpu';
}

async function makeEngine(path: 'workers' | 'cpu', prob: Stage1Problem, o: Stage1Options): Promise<GradEngine> {
  if (path === 'cpu') return new LocalEngine(prob);
  return WorkerPoolEngine.create(prob, layout().nParams, o.workers ?? {});
}

/** Runs stage 1 end to end: problem, initial weights, training on the chosen path, J. */
export async function runStage1(proj: Projections, o: Stage1Options = {}): Promise<Stage1Result> {
  const prob = buildProblem(proj);
  const p0 = initXavier(o.seed ?? 42);
  const iterations = o.iterations ?? ITER_LONG;
  const path = await choosePath(o.path);
  const topts: TrainOptions = { ...o, iterations };
  let train: TrainResult;
  if (path === 'webgpu') {
    const g = await createGpuStage1(prob, p0, iterations, o);
    try { train = await g.train({ ...topts, itersPerSubmit: o.itersPerSubmit }); } finally { g.destroy(); }
  } else {
    const eng = await makeEngine(path, prob, o);
    try { train = await trainWithEngine(eng, p0, topts); } finally { eng.dispose(); }
  }
  return finishStage1(prob, proj.b, { rMm: proj.rMm, zMm: proj.zMm, times: proj.times }, train,
                      { absolute: o.absolute ?? false, jTruePerMm: o.jTruePerMm });
}

/** Seconds per iteration measured with `probe` iterations on the chosen path, for the
 *  estimated time shown before training (message annabel_slow). */
export async function estimateSecondsPerIter(proj: Projections, o: Stage1Options & { probe?: number } = {}): Promise<{ path: Stage1Path; sPerIter: number }> {
  const prob = buildProblem(proj);
  const p0 = initXavier(o.seed ?? 42);
  const probe = Math.max(3, o.probe ?? 20);
  const path = await choosePath(o.path);
  if (path === 'webgpu') {
    const g = await createGpuStage1(prob, p0, probe + 3, o);
    try {
      g.encodeIters(3); await g.done();
      const t0 = performance.now(); g.encodeIters(probe); await g.done();
      return { path, sPerIter: (performance.now() - t0) / 1000 / probe };
    } finally { g.destroy(); }
  }
  const eng = await makeEngine(path, prob, o);
  try {
    const p = p0.slice();
    await eng.evaluate(p, true);
    const t0 = performance.now();
    for (let i = 0; i < probe; i++) await eng.evaluate(p, true);
    return { path, sPerIter: (performance.now() - t0) / 1000 / probe };
  } finally { eng.dispose(); }
}

export interface Stage2Options extends Omit<TrainOptions, 'iterations'> {
  iterations?: number;              // Stage2Config 3000
  cfg?: Partial<Stage2Config>;
  seed?: number;
  p0?: Float32Array;                // fixed initial weights (parity)
  path?: Stage1Path | 'auto';
  workers?: { n?: number; rowsPerChunk?: number; factory?: () => Worker };
  itersPerSubmit?: number;
}

/** RGB-ANNAbel stage 2 end to end on J (Nt, Nz, Nr, 3) in the units of kern (J · j_to_si = kernel units). */
export async function runStage2(J: ArrayLike<number>, grid: { nt: number; nz: number; nr: number; rMm: ArrayLike<number>; zMm: ArrayLike<number>; times: ArrayLike<number> },
                                kern: KernelTable, o: Stage2Options = {}): Promise<Stage2Result> {
  const prob = buildStage2Problem(J, grid, kern, { ...o.cfg, ...(o.iterations ? { iterations: o.iterations } : {}) });
  const p0 = o.p0 ?? initStage2(o.seed ?? prob.cfg.seed, prob.sizes);
  const path = await choosePath(o.path);
  const iterations = prob.cfg.iterations;
  let train: TrainResult;
  if (path === 'webgpu') {
    const g = await createGpuStage2(prob, p0, iterations);
    try { train = await g.train({ ...o, iterations, itersPerSubmit: o.itersPerSubmit }); } finally { g.destroy(); }
  } else {
    const eng = path === 'cpu' ? new Stage2LocalEngine(prob) : await Stage2PoolEngine.create(prob, p0.length, o.workers ?? {});
    try { train = await trainStage2(eng, p0, prob, { ...o, iterations }); } finally { if (!o.workers?.factory) eng.dispose(); }
  }
  return finishStage2(prob, J, grid, kern, train);
}

/** Seconds per stage-2 iteration on the chosen path, from `probe` iterations. */
export async function estimateStage2(J: ArrayLike<number>, grid: Parameters<typeof runStage2>[1], kern: KernelTable, o: Stage2Options & { probe?: number } = {}): Promise<{ path: Stage1Path; sPerIter: number }> {
  const prob = buildStage2Problem(J, grid, kern, o.cfg);
  const p0 = o.p0 ?? initStage2(o.seed ?? prob.cfg.seed, prob.sizes);
  const probe = Math.max(3, o.probe ?? 20);
  const path = await choosePath(o.path);
  if (path === 'webgpu') {
    const g = await createGpuStage2(prob, p0, probe + 3);
    try { g.encodeIters(3); await g.done(); const t0 = performance.now(); g.encodeIters(probe); await g.done(); return { path, sPerIter: (performance.now() - t0) / 1000 / probe }; } finally { g.destroy(); }
  }
  const eng = path === 'cpu' ? new Stage2LocalEngine(prob) : await Stage2PoolEngine.create(prob, p0.length, o.workers ?? {});
  try {
    await eng.evaluate(p0, true);
    const t0 = performance.now();
    for (let i = 0; i < probe; i++) await eng.evaluate(p0, true);
    return { path, sPerIter: (performance.now() - t0) / 1000 / probe };
  } finally { if (!o.workers?.factory) eng.dispose(); }
}
