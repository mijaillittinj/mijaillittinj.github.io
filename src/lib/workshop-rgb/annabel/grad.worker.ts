// Helper Worker of the CPU path: loss sums and gradient of ANNAbel stage 1 for the blocks of
// rows it is handed (workers.ts). Typed with a local scope interface, because
// `/// <reference lib="webworker" />` next to lib.dom fails with TS6200 (WEB_SPEC §7.1).
import type { FromGradHelper, ToGradHelper } from './workers.ts';
import type { Stage1Problem } from './problem.ts';
import { Net } from './net.ts';

interface WorkerScope { postMessage(m: FromGradHelper, transfer?: Transferable[]): void; onmessage: ((e: MessageEvent<ToGradHelper>) => void) | null }
const ctx = self as unknown as WorkerScope;

let prob: Stage1Problem | null = null;
let net: Net | null = null;
let params: Float32Array | null = null;
let grad: Float64Array | null = null;
let sR = 0, sL = 0;

function work(c: [number, number]) {
  if (!prob || !net || !params || !grad) throw new Error('helper not initialised');
  const s = net.lossGrad(params, prob, c[0], c[1], grad);
  sR += s.sR; sL += s.sL;
}

ctx.onmessage = (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      prob = m.prob;
      net = new Net(m.rowsPerChunk * prob.nr);
      grad = new Float64Array(net.nParams);
      ctx.postMessage({ type: 'ready' });
    } else if (m.type === 'params') {
      params = m.params; grad!.fill(0); sR = 0; sL = 0;
      if (m.chunk) { work(m.chunk); ctx.postMessage({ type: 'done' }); }
    } else if (m.type === 'chunk') {
      work(m.chunk); ctx.postMessage({ type: 'done' });
    } else if (m.type === 'finish') {
      const out = grad!; grad = new Float64Array(out.length);
      ctx.postMessage({ type: 'grad', sR, sL, grad: out }, [out.buffer]);
    }
  } catch (e) {
    ctx.postMessage({ type: 'error', message: String((e as Error).message ?? e) });
  }
};
