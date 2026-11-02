// ANNAbel stage-1 training on the CPU, and the result shared by the CPU and WebGPU paths.
// models.train_model: Adam (torch defaults) with CosineAnnealingLR(T_max = iterations,
// eta_min = lr_min_frac · lr), lr_min_frac = 0.1, and the checkpoint with the lowest training
// loss taken BEFORE the optimiser step (checkpoint_policy = best_train_loss). All samples are
// used for training. A non-finite loss stops with an error, like the app.
import { Net, predictNorm } from './net.ts';
import type { Stage1Problem } from './problem.ts';
import { splitHalves } from './problem.ts';
import { PER_MM_TO_PER_M } from './units.ts';

export interface TrainOptions {
  iterations: number;          // planned iterations, also T_max of the cosine (app default 4000, long run 30 000)
  lr?: number;                 // 1e-3
  lrMinFrac?: number;          // 0.1
  signal?: AbortSignal;        // stop button, end_reason 'stopped_by_user'
  onTick?: (t: TrainTick) => void;
  tickEveryMs?: number;        // 500 ms, like the app's progress callback
}

export interface TrainTick { iter: number; total: number; loss: number; best: number; bestIter: number; lr: number; elapsedS: number; etaS: number | null }

export interface TrainResult {
  params: Float32Array;        // selected checkpoint (lowest loss, before the step)
  bestLoss: number;
  bestIter: number;            // 1-based
  checkpointCriterion: 'best_train_loss';
  endReason: 'completed' | 'stopped_by_user';
  completedIters: number;
  plannedIters: number;
  elapsedS: number;
  history: Float32Array;       // loss per completed iteration
  path: 'cpu' | 'workers' | 'webgpu';
}

export const DEFAULT_LR = 1e-3;
export const DEFAULT_LR_MIN_FRAC = 0.1;

/** Learning rate of 0-based epoch e (the lr used by step e + 1), closed form of torch's
 *  CosineAnnealingLR. */
export function cosineLr(e: number, iterations: number, lr = DEFAULT_LR, minFrac = DEFAULT_LR_MIN_FRAC): number {
  const lo = minFrac * lr;
  return lo + (lr - lo) * (1 + Math.cos(Math.PI * e / iterations)) / 2;
}

/** torch.optim.Adam (β = 0.9, 0.999, ε = 1e-8, no weight decay) with the cosine schedule and
 *  the best-before-step checkpoint. State in Float64, parameters in Float32. */
export class Adam {
  m: Float64Array;
  v: Float64Array;
  t: number;
  lr0: number;
  minFrac: number;
  iterations: number;
  bestLoss: number;
  bestIter: number;
  best: Float32Array;
  constructor(p0: Float32Array, iterations: number, lr = DEFAULT_LR, minFrac = DEFAULT_LR_MIN_FRAC) {
    this.m = new Float64Array(p0.length); this.v = new Float64Array(p0.length);
    this.t = 0; this.lr0 = lr; this.minFrac = minFrac; this.iterations = iterations;
    this.bestLoss = Infinity; this.bestIter = 0; this.best = p0.slice();
  }
  lrNow() { return cosineLr(this.t, this.iterations, this.lr0, this.minFrac); }
  /** One iteration: `loss` was computed at p (before this step). */
  step(p: Float32Array, g: ArrayLike<number>, loss: number) {
    if (loss < this.bestLoss) { this.bestLoss = loss; this.best.set(p); this.bestIter = this.t + 1; }
    const lr = this.lrNow();
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - b1 ** this.t, bc2s = Math.sqrt(1 - b2 ** this.t);
    const ss = lr / bc1;
    for (let q = 0; q < p.length; q++) {
      const gq = g[q];
      const mq = this.m[q] = b1 * this.m[q] + (1 - b1) * gq;
      const vq = this.v[q] = b2 * this.v[q] + (1 - b2) * gq * gq;
      p[q] -= ss * mq / (Math.sqrt(vq) / bc2s + eps);
    }
  }
}

/** Loss and gradient of the global stage-1 loss for given parameters. */
export interface GradEngine {
  readonly nParams: number;
  readonly path: 'cpu' | 'workers';
  evaluate(p: Float32Array, withGrad: boolean): Promise<{ loss: number; lR: number; lL: number; grad: Float64Array | null }>;
  dispose(): void;
}

/** Same-thread engine, rows in blocks (used in Node tests, inside a Worker, and by each helper). */
export class LocalEngine implements GradEngine {
  readonly nParams: number;
  readonly path = 'cpu' as const;
  prob: Stage1Problem;
  net: Net;
  rowsPerBlock: number;
  constructor(prob: Stage1Problem, rowsPerBlock = 8) {
    this.prob = prob; this.rowsPerBlock = Math.max(1, Math.min(rowsPerBlock, prob.rows));
    this.net = new Net(this.rowsPerBlock * prob.nr);
    this.nParams = this.net.nParams;
  }
  sums(p: ArrayLike<number>, row0: number, row1: number, grad: Float64Array | null) {
    let sR = 0, sL = 0;
    for (let a = row0; a < row1; a += this.rowsPerBlock) {
      const s = this.net.lossGrad(p, this.prob, a, Math.min(row1, a + this.rowsPerBlock), grad);
      sR += s.sR; sL += s.sL;
    }
    return { sR, sL };
  }
  async evaluate(p: Float32Array, withGrad: boolean) {
    const grad = withGrad ? new Float64Array(this.nParams) : null;
    const { sR, sL } = this.sums(p, 0, this.prob.rows, grad);
    const M = this.prob.M;
    return { loss: 0.5 * (sR + sL) / M, lR: sR / M, lL: sL / M, grad };
  }
  dispose() { /* nothing held outside the heap */ }
}

/** Adam loop on the host over any GradEngine (one thread or a pool of Workers). */
export async function trainWithEngine(engine: GradEngine, p0: Float32Array, o: TrainOptions): Promise<TrainResult> {
  const n = Math.max(1, Math.floor(o.iterations));
  const opt = new Adam(p0, n, o.lr ?? DEFAULT_LR, o.lrMinFrac ?? DEFAULT_LR_MIN_FRAC);
  const p = p0.slice();
  const history = new Float32Array(n);
  const t0 = performance.now();
  let lastTick = -Infinity, done = 0;
  let endReason: TrainResult['endReason'] = 'completed';
  for (let it = 1; it <= n; it++) {
    if (o.signal?.aborted) { endReason = 'stopped_by_user'; break; }
    const { loss, grad } = await engine.evaluate(p, true);
    if (!Number.isFinite(loss)) throw new Error(`non-finite loss ${loss} at iteration ${it}`);
    const lr = opt.lrNow();
    opt.step(p, grad as Float64Array, loss);
    history[it - 1] = loss; done = it;
    const now = performance.now();
    if (o.onTick && (now - lastTick >= (o.tickEveryMs ?? 500) || it === n)) {
      const el = (now - t0) / 1000;
      o.onTick({ iter: it, total: n, loss, best: opt.bestLoss, bestIter: opt.bestIter, lr, elapsedS: el,
                 etaS: it >= 20 ? el / it * (n - it) : null });
      lastTick = now;
    }
  }
  if (done === 0) throw new Error('training stopped before the first iteration');
  return { params: opt.best.slice(), bestLoss: opt.bestLoss, bestIter: opt.bestIter, checkpointCriterion: 'best_train_loss',
           endReason, completedIters: done, plannedIters: n, elapsedS: (performance.now() - t0) / 1000,
           history: history.slice(0, done), path: engine.path };
}

/** What stage 1 hands to stage 2 (solve_T_fs) and to the figures. */
export interface Stage1Result {
  nt: number; nz: number; nr: number;
  rMm: Float64Array; zMm: Float64Array; times: Float64Array;
  /** (Nt, Nz, Nr, 3) emission in data units per mm, the app's J. */
  jPerMm: Float32Array;
  /** (Nt, Nz, Nr, 3) in W m⁻³ sr⁻¹ (jPerMm × 1000) when b was in W m⁻² sr⁻¹, else null. */
  jSI: Float32Array | null;
  /** (Nt, Nz, Nr, 3) reprojection A J in the units of b. */
  P: Float32Array;
  residualStdRight: number[];
  residualStdLeft: number[];
  gtRelL2: number[] | null;
  jUnit: string;
  train: TrainResult;
  meta: { sizes: number[]; activation: 'tanh'; norm: Stage1Problem['norm']; outScale: number; bScale: number; output: string };
}

/** Relative L2 error per channel, ‖J − J_true‖ / ‖J_true‖ (the app's gt_rel_l2). */
export function relL2(J: ArrayLike<number>, Jtrue: ArrayLike<number>): number[] {
  const num = [0, 0, 0], den = [0, 0, 0];
  for (let q = 0; q < J.length; q++) { const t = Jtrue[q], d = J[q] - t; num[q % 3] += d * d; den[q % 3] += t * t; }
  return num.map((v, c) => Math.sqrt(v / den[c]));
}

/** run_stage1 after training: J, reprojection, residual spreads, optional ground truth. */
export function finishStage1(prob: Stage1Problem, b: Float32Array, grid: { rMm: ArrayLike<number>; zMm: ArrayLike<number>; times: ArrayLike<number> },
                             train: TrainResult, o: { absolute: boolean; jTruePerMm?: ArrayLike<number> | null }): Stage1Result {
  const { nr, rows } = prob;
  const Jn = predictNorm(train.params, prob);
  const J = new Float32Array(Jn.length);
  for (let q = 0; q < J.length; q++) J[q] = Jn[q] * prob.bScale;
  const P = new Float32Array(J.length);
  for (let k = 0; k < rows; k++)
    for (let i = 0; i < nr; i++) {
      let p0 = 0, p1 = 0, p2 = 0;
      for (let j = i; j < nr; j++) {
        const a = prob.A[i * nr + j], q = (k * nr + j) * 3;
        p0 += a * J[q]; p1 += a * J[q + 1]; p2 += a * J[q + 2];
      }
      const q = (k * nr + i) * 3;
      P[q] = p0; P[q + 1] = p1; P[q + 2] = p2;
    }
  const { right, left } = splitHalves(b, rows, nr);
  const std = (h: Float32Array) => [0, 1, 2].map((c) => {
    let s = 0, s2 = 0, n = 0;
    for (let q = c; q < P.length; q += 3) { const d = P[q] - h[q]; s += d; s2 += d * d; n++; }
    const mu = s / n;
    return Math.sqrt(Math.max(0, s2 / n - mu * mu));
  });
  return {
    nt: prob.nt, nz: prob.nz, nr,
    rMm: Float64Array.from(grid.rMm), zMm: Float64Array.from(grid.zMm), times: Float64Array.from(grid.times),
    jPerMm: J,
    jSI: o.absolute ? Float32Array.from(J, (v) => v * PER_MM_TO_PER_M) : null,
    P, residualStdRight: std(right), residualStdLeft: std(left),
    gtRelL2: o.jTruePerMm ? relL2(J, o.jTruePerMm) : null,
    jUnit: o.absolute ? 'W m^-3 sr^-1' : 'relative per mm',
    train,
    meta: { sizes: [3, 64, 64, 64, 64, 3], activation: 'tanh', norm: prob.norm, outScale: prob.outScale, bScale: prob.bScale,
            output: 'J = out_scale * softplus(net(x)) * b_scale per mm; j_SI = 1000 J' },
  };
}

/** One time and one height of a Stage1Result as three planes (Nz, Nr) per channel, the shape
 *  that the tfs stage and the NPY export use. Values in W m⁻³ sr⁻¹ when absolute. */
export function channelPlanes(res: Stage1Result, t = 0): { R: Float32Array; G: Float32Array; B: Float32Array } {
  const src = res.jSI ?? res.jPerMm;
  const n = res.nz * res.nr, off = t * n * 3;
  const R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n);
  for (let s = 0; s < n; s++) { R[s] = src[off + s * 3]; G[s] = src[off + s * 3 + 1]; B[s] = src[off + s * 3 + 2]; }
  return { R, G, B };
}

/** One time of a Stage1Result as interleaved (Nz·Nr × 3), the layout of solveTfs in
 *  ../radiometry.ts (W m⁻³ sr⁻¹ when absolute). A view, not a copy. */
export function jFrame(res: Stage1Result, t = 0): Float32Array {
  const src = res.jSI ?? res.jPerMm, n = res.nz * res.nr * 3;
  return src.subarray(t * n, (t + 1) * n);
}
