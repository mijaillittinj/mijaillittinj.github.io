// ANNAbel stage 1 in the browser (Escudero et al., Proc. Combust. Inst. 40 (2024) 105493),
// ported from annabel_app/core (abel.py, models.py, stage1.py). Stage 2 of the tool is the
// direct fit solve_T_fs of the core (radiometry), which takes the channel planes of
// `channelPlanes(result)` in W m⁻³ sr⁻¹. The app's stage-2 network, the ensemble and the
// relative mode are not ported (§11.2).
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

export type { Projections, Stage1Problem } from './problem.ts';
export type { Stage1Result, TrainResult, TrainTick, TrainOptions } from './train.ts';
export { opMatrix, splitHalves, coordNormFromGrid, coordFeatures, buildProblem } from './problem.ts';
export { SIZES, layout, initXavier, Net, predictNorm } from './net.ts';
export { Adam, cosineLr, LocalEngine, trainWithEngine, finishStage1, channelPlanes, jFrame, relL2 } from './train.ts';
export { emissionPerMmToSI, emissionSIToPerMm, PER_MM_TO_PER_M } from './units.ts';
export { WorkerPoolEngine, helperCount } from './workers.ts';
export { createGpuStage1, webgpuAvailable } from './gpu.ts';

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
