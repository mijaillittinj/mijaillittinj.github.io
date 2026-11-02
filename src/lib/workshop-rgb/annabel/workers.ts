// CPU path of ANNAbel stage 1 with several Web Workers (no SharedArrayBuffer: GitHub Pages
// sends no COOP/COEP). Each helper keeps a copy of the problem; per iteration the host sends
// the parameters, hands out blocks of rows from a queue, and sums the helpers' loss sums and
// gradients. The Adam step stays on the host (train.ts).
import type { GradEngine } from './train.ts';
import type { Stage1Problem } from './problem.ts';

/** Messages host ↔ helper (grad.worker.ts). */
export type ToGradHelper =
  | { type: 'init'; prob: Stage1Problem; rowsPerChunk: number }
  | { type: 'params'; params: Float32Array; chunk: [number, number] | null }
  | { type: 'chunk'; chunk: [number, number] }
  | { type: 'finish' };
export type FromGradHelper =
  | { type: 'ready' }
  | { type: 'done' }
  | { type: 'grad'; sR: number; sL: number; grad: Float64Array }
  | { type: 'error'; message: string };

/** The default helper, bundled by Vite (and by esbuild for the offline copy). */
export function defaultHelperFactory(): Worker {
  return new Worker(new URL('./grad.worker.ts', import.meta.url), { type: 'module' });
}

/** Number of helpers the spec asks for, min(hardwareConcurrency − 1, 4), at least 1. */
export function helperCount(): number {
  const hc = typeof navigator !== 'undefined' && navigator.hardwareConcurrency ? navigator.hardwareConcurrency : 2;
  return Math.max(1, Math.min(hc - 1, 4));
}

export class WorkerPoolEngine implements GradEngine {
  readonly nParams: number;
  readonly path = 'workers' as const;
  workers: Worker[];
  prob: Stage1Problem;
  rowsPerChunk: number;

  constructor(workers: Worker[], prob: Stage1Problem, nParams: number, rowsPerChunk: number) {
    this.workers = workers; this.prob = prob; this.nParams = nParams; this.rowsPerChunk = rowsPerChunk;
  }

  /** Creates n helpers and sends each a copy of the problem. */
  static async create(prob: Stage1Problem, nParams: number, o: { n?: number; rowsPerChunk?: number; factory?: () => Worker } = {}) {
    const n = Math.max(1, Math.min(o.n ?? helperCount(), prob.rows));
    const rowsPerChunk = Math.max(1, o.rowsPerChunk ?? Math.min(4, Math.ceil(prob.rows / n)));
    const make = o.factory ?? defaultHelperFactory;
    const ws: Worker[] = [];
    await Promise.all(Array.from({ length: n }, () => new Promise<void>((res, rej) => {
      const w = make();
      w.onmessage = (ev: MessageEvent<FromGradHelper>) => (ev.data.type === 'ready' ? res() : rej(new Error(ev.data.type === 'error' ? ev.data.message : 'helper did not start')));
      w.onerror = (e) => rej(new Error(e.message || 'helper failed to load'));
      const msg: ToGradHelper = { type: 'init', prob, rowsPerChunk };
      w.postMessage(msg);   // structured clone, each helper owns its copy
      ws.push(w);
    })));
    return new WorkerPoolEngine(ws, prob, nParams, rowsPerChunk);
  }

  evaluate(p: Float32Array, withGrad: boolean): Promise<{ loss: number; lR: number; lL: number; grad: Float64Array | null }> {
    const { prob, workers, rowsPerChunk } = this;
    const queue: [number, number][] = [];
    for (let a = 0; a < prob.rows; a += rowsPerChunk) queue.push([a, Math.min(prob.rows, a + rowsPerChunk)]);
    const grad = new Float64Array(this.nParams);
    let sR = 0, sL = 0, pending = workers.length;
    return new Promise((resolve, reject) => {
      workers.forEach((w) => {
        w.onerror = (e) => reject(new Error(e.message || 'helper error'));
        w.onmessage = (ev: MessageEvent<FromGradHelper>) => {
          const m = ev.data;
          if (m.type === 'done') {
            const c = queue.shift();
            w.postMessage(c ? { type: 'chunk', chunk: c } satisfies ToGradHelper : { type: 'finish' } satisfies ToGradHelper);
          } else if (m.type === 'grad') {
            sR += m.sR; sL += m.sL;
            for (let k = 0; k < grad.length; k++) grad[k] += m.grad[k];
            if (--pending === 0) {
              const M = prob.M;
              resolve({ loss: 0.5 * (sR + sL) / M, lR: sR / M, lL: sL / M, grad: withGrad ? grad : null });
            }
          } else if (m.type === 'error') reject(new Error(m.message));
        };
        const c = queue.shift() ?? null;
        w.postMessage({ type: 'params', params: p.slice(), chunk: c } satisfies ToGradHelper);
        if (!c) w.postMessage({ type: 'finish' } satisfies ToGradHelper);
      });
    });
  }

  dispose() { for (const w of this.workers) w.terminate(); this.workers = []; }
}
