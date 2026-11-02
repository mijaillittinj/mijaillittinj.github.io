// WebGPU path of RGB-ANNAbel stage 2. The whole iteration stays on the GPU in f32, like stage 1
// (gpu.ts, whose linear, backpropagation, Adam and step kernels it reuses): forward pass to
// (o₀, o₁), sigmoid bounds, K_c(T) from the table, direct and ratio terms of the loss with
// their derivatives, lowest-loss checkpoint before the step, backpropagation, Adam with the
// cosine schedule. Same loss and parameter layout as stage2.ts.
import type { Stage2Problem } from './stage2.ts';
import { layout } from './net.ts';
import type { TrainOptions, TrainResult } from './train.ts';
import { cosineLr } from './train.ts';
import type { GpuInfo } from './gpu.ts';
import { WG, UHDR, ST, K_LIN_FWD, K_DX, K_DW, K_DWRED, K_ADAM, K_BUMP } from './gpu.ts';

const U_MAP_READ = 0x1, U_COPY_SRC = 0x4, U_COPY_DST = 0x8, U_UNIFORM = 0x40, U_STORAGE = 0x80;
const MAP_READ = 0x1;

// per sample: T, f_s, K(T), the two terms of the loss and dloss/do
const K_S2 = `struct V { N:u32, nT:u32, npair:u32, pairs:u32, TLO:f32, dTr:f32, fsRef:f32, lam:f32,
  invN:f32, invNtr:f32, invWs:f32, kscale:f32, w0:f32, w1:f32, w2:f32, n0:f32, n1:f32, n2:f32,
  r0:f32, r1:f32, r2:f32, p0:f32, p1:f32, p2:f32 };
@group(0) @binding(0) var<uniform> v: V;
@group(0) @binding(1) var<storage, read> O: array<f32>;
@group(0) @binding(2) var<storage, read> Yd: array<f32>;
@group(0) @binding(3) var<storage, read> mk: array<f32>;
@group(0) @binding(4) var<storage, read> rat: array<f32>;
@group(0) @binding(5) var<storage, read> Kt: array<f32>;
@group(0) @binding(6) var<storage, read_write> lt: array<f32>;
@group(0) @binding(7) var<storage, read_write> dO: array<f32>;
fn sg(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let n = g.x + g.y * nw.x * ${WG}u; if (n >= v.N) { return; }
  let s0 = sg(O[2u * n]); let s1 = sg(O[2u * n + 1u]);
  let T = v.TLO + v.dTr * s0; let fs = v.fsRef * s1;
  let uu = (T - v.TLO) * v.kscale;
  let i0 = u32(clamp(floor(uu), 0.0, f32(v.nT - 2u)));
  let w = clamp(uu - f32(i0), 0.0, 1.0);
  var K: array<f32, 3>; var dK: array<f32, 3>; var gK: array<f32, 3>;
  var ws = array<f32, 3>(v.w0, v.w1, v.w2); var nm = array<f32, 3>(v.n0, v.n1, v.n2);
  var rn = array<f32, 3>(v.r0, v.r1, v.r2);
  var ld = 0.0; var gfs = 0.0;
  let gD = 2.0 * v.invWs * v.invN;
  for (var c = 0u; c < 3u; c++) {
    let k0 = Kt[c * v.nT + i0]; let k1 = Kt[c * v.nT + i0 + 1u];
    K[c] = k0 * (1.0 - w) + k1 * w; dK[c] = (k1 - k0) * v.kscale; gK[c] = 0.0;
    if (ws[c] > 0.0) {
      let e = (Yd[3u * n + c] - fs * K[c]) / nm[c];
      ld += ws[c] * e * e;
      let d = -gD * ws[c] * e / nm[c];
      gfs += d * K[c]; gK[c] = d * fs;
    }
  }
  var lq = 0.0;
  let m = mk[n];
  let gQ = 2.0 * v.lam * v.invNtr;
  for (var q = 0u; q < v.npair; q++) {
    let a = (v.pairs >> (4u * q)) & 3u; let b = (v.pairs >> (4u * q + 2u)) & 3u;
    let pr = K[a] / K[b];
    let e = m * (rat[q * v.N + n] - pr) / rn[q];
    lq += e * e;
    let d = -gQ * e * m / rn[q];
    gK[a] += d / K[b]; gK[b] -= d * K[a] / (K[b] * K[b]);
  }
  lt[n] = ld * v.invWs * v.invN + v.lam * lq * v.invNtr;
  let dT = gK[0] * dK[0] + gK[1] * dK[1] + gK[2] * dK[2];
  dO[2u * n] = dT * v.dTr * s0 * (1.0 - s0);
  dO[2u * n + 1u] = gfs * v.fsRef * s1 * (1.0 - s1);
}`;

// loss = Σ lt, history, and the checkpoint copy when the loss improves (before the step)
const K_LOSS2 = UHDR + ST + `
@group(0) @binding(1) var<storage, read> lt: array<f32>;
@group(0) @binding(2) var<storage, read> P: array<f32>;
@group(0) @binding(3) var<storage, read_write> best: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: St;
@group(0) @binding(5) var<storage, read_write> hist: array<f32>;
var<workgroup> sh: array<f32, ${WG}>;
var<workgroup> flag: u32;
@compute @workgroup_size(${WG}) fn main(@builtin(local_invocation_id) l: vec3u) {
  let t = l.x;
  var s = 0.0;
  for (var q = t; q < u.N; q += ${WG}u) { s += lt[q]; }
  sh[t] = s;
  workgroupBarrier();
  for (var h = ${WG / 2}u; h > 0u; h >>= 1u) { if (t < h) { sh[t] += sh[t + h]; } workgroupBarrier(); }
  if (t == 0u) {
    let L = sh[0];
    st.loss = L; hist[st.step] = L;
    if (L < st.best) { st.best = L; st.bestStep = st.step + 1u; flag = 1u; } else { flag = 0u; }
  }
  workgroupBarrier();
  if (flag == 1u) { for (var q = t; q < u.nP; q += ${WG}u) { best[q] = P[q]; } }
}`;

export interface GpuStage2 {
  device: GPUDevice; info: GpuInfo; nParams: number;
  evaluate(): Promise<{ loss: number; grad: Float32Array }>;
  train(o: TrainOptions & { itersPerSubmit?: number }): Promise<TrainResult>;
  encodeIters(k: number): void;
  done(): Promise<void>;
  readParams(which: 'P' | 'best' | 'G'): Promise<Float32Array>;
  destroy(): void;
}

export async function createGpuStage2(prob: Stage2Problem, p0: Float32Array, iterations: number): Promise<GpuStage2> {
  const nav = (globalThis as { navigator?: Navigator }).navigator;
  if (!nav || !('gpu' in nav) || !nav.gpu) throw new Error('webgpu_missing');
  const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('webgpu_no_adapter');
  const sizes = prob.sizes, L = sizes.length - 1;
  const { wOff, bOff, nParams } = layout(sizes);
  if (p0.length !== nParams) throw new Error('parameter vector does not match the stage-2 network');
  if (Math.max(...sizes) > 64) throw new Error('webgpu_limits');
  const N = prob.N;
  const bigBytes = N * 64 * 4;
  const lim = adapter.limits;
  if (lim.maxStorageBuffersPerShaderStage < 7 || bigBytes > lim.maxStorageBufferBindingSize) throw new Error('webgpu_limits');
  const req: Record<string, number> = { maxStorageBuffersPerShaderStage: Math.min(8, lim.maxStorageBuffersPerShaderStage) };
  if (bigBytes > 134217728) { req.maxStorageBufferBindingSize = lim.maxStorageBufferBindingSize; req.maxBufferSize = lim.maxBufferSize; }
  const device = await adapter.requestDevice({ requiredLimits: req });
  const ai = adapter.info;
  const info: GpuInfo = { vendor: ai?.vendor ?? '', architecture: ai?.architecture ?? '', device: ai?.device ?? '', description: ai?.description ?? '' };
  const nIters = Math.max(1, Math.floor(iterations));
  const lr0 = prob.cfg.lr, minFrac = prob.cfg.lrMinFrac;
  const buffers: GPUBuffer[] = [];
  const buf = (bytes: number, data?: ArrayBufferView) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: U_STORAGE | U_COPY_SRC | U_COPY_DST });
    if (data) device.queue.writeBuffer(b, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    buffers.push(b);
    return b;
  };
  const P = buf(nParams * 4, p0), G = buf(nParams * 4), m1 = buf(nParams * 4), v1 = buf(nParams * 4), best = buf(nParams * 4, p0);
  const H: GPUBuffer[] = [buf(N * sizes[0] * 4, prob.X)];
  for (let l = 1; l < L; l++) H.push(buf(N * sizes[l] * 4));
  const O = buf(N * 8), lt = buf(N * 4);
  const Yd = buf(N * 12, prob.Y), mk = buf(N * 4, Float32Array.from(prob.mask));
  const rat = new Float32Array(Math.max(1, prob.pairs.length) * N);
  prob.ratios.forEach((r, q) => rat.set(r, q * N));
  const RT = buf(rat.length * 4, rat), Kt = buf(prob.K.length * 4, prob.K);
  const DA = buf(bigBytes), DB = buf(bigBytes);
  const NSPLIT = 128, chunk = Math.ceil(N / NSPLIT);
  const part = buf(NSPLIT * (64 * 64 + 64) * 4);
  const st = buf(16, new Uint32Array([0, 0, 0, 0x7f7fffff]));
  const hist = buf(nIters * 4);
  const ss = new Float32Array(nIters), c2 = new Float32Array(nIters);
  for (let e = 0; e < nIters; e++) { ss[e] = cosineLr(e, nIters, lr0, minFrac) / (1 - 0.9 ** (e + 1)); c2[e] = Math.sqrt(1 - 0.999 ** (e + 1)); }
  const lrs = buf(nIters * 4, ss), bc = buf(nIters * 4, c2);
  const uniBuf = (ab: ArrayBuffer) => { const b = device.createBuffer({ size: ab.byteLength, usage: U_UNIFORM | U_COPY_DST }); device.queue.writeBuffer(b, 0, ab); buffers.push(b); return b; };
  const uni = (ov: Record<string, number>) => {
    const ab = new ArrayBuffer(80), dv = new DataView(ab);
    const UK = ['N', 'K', 'M', 'wOff', 'bOff', 'act', 'nsplit', 'chunk', 'nr', 'rows', 'nP', 'O'];
    const FK = ['os', 'invM', 'w0', 'w1', 'w2', 'b1', 'b2', 'eps'];
    const base: Record<string, number> = { N, nr: prob.nr, rows: prob.rows, nP: nParams, nsplit: NSPLIT, chunk, b1: 0.9, b2: 0.999, eps: 1e-8, ...ov };
    UK.forEach((k, i) => dv.setUint32(i * 4, base[k] ?? 0, true));
    FK.forEach((k, i) => dv.setFloat32(48 + i * 4, base[k] ?? 0, true));
    return uniBuf(ab);
  };
  const vab = new ArrayBuffer(96), vd = new DataView(vab);
  let packed = 0;
  prob.pairs.forEach(([a, b], q) => { packed |= (a | (b << 2)) << (4 * q); });
  [N, prob.nT, prob.pairs.length, packed].forEach((x, i) => vd.setUint32(i * 4, x, true));
  const dTr = prob.Thi - prob.Tlo;
  [prob.Tlo, dTr, prob.fsRef, prob.lam, 1 / N, 1 / prob.nTr, 1 / prob.wSum, (prob.nT - 1) / dTr, ...prob.w, ...prob.nrm2,
   ...[0, 1, 2].map((q) => prob.ratioNrm[q] ?? 1), 0, 0, 0].forEach((x, i) => vd.setFloat32(16 + i * 4, x, true));
  const uv = uniBuf(vab);

  const pipes = new Map<string, GPUComputePipeline>();
  const pipe = (code: string) => {
    let p = pipes.get(code);
    if (!p) { p = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } }); pipes.set(code, p); }
    return p;
  };
  interface D { p: GPUComputePipeline; bg: GPUBindGroup; wx: number; wy: number }
  const mkD = (code: string, u: GPUBuffer, bufs: GPUBuffer[], threads: number, flat = false): D => {
    const p = pipe(code);
    const bg = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [u, ...bufs].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const wgs = flat ? threads : Math.ceil(threads / WG);
    const wx = Math.min(wgs, 65535), wy = Math.ceil(wgs / wx);
    return { p, bg, wx, wy };
  };
  device.pushErrorScope('validation');
  const fwd: D[] = [], bwd: D[] = [];
  for (let l = 0; l < L; l++) {
    const K = sizes[l], M = sizes[l + 1];
    fwd.push(mkD(K_LIN_FWD, uni({ K, M, wOff: wOff[l], bOff: bOff[l], act: l < L - 1 ? 1 : 0 }), [P, H[l], l < L - 1 ? H[l + 1] : O], N * M));
  }
  const u0 = uni({});
  fwd.push(mkD(K_S2, uv, [O, Yd, mk, RT, Kt, lt, DA], N));
  fwd.push(mkD(K_LOSS2, u0, [lt, P, best, st, hist], 1, true));
  let dY = DA, dX = DB;
  for (let l = L - 1; l >= 0; l--) {
    const K = sizes[l], M = sizes[l + 1], Oc = M * K + M;
    const ul = uni({ K, M, wOff: wOff[l], bOff: bOff[l], O: Oc });
    bwd.push(mkD(K_DW, ul, [dY, H[l], part], NSPLIT, true));
    bwd.push(mkD(K_DWRED, ul, [part, G], Oc));
    if (l > 0) { bwd.push(mkD(K_DX, ul, [P, dY, H[l], dX], N * K)); const t = dY; dY = dX; dX = t; }
  }
  const adam = mkD(K_ADAM, u0, [P, G, m1, v1, st, lrs, bc], nParams);
  const bump = mkD(K_BUMP, u0, [st], 1, true);
  const err = await device.popErrorScope();
  if (err) { device.destroy(); throw new Error(`webgpu_pipeline: ${err.message}`); }

  let lost = false;
  device.lost.then(() => { lost = true; });
  const run = (pass: GPUComputePassEncoder, d: D) => { pass.setPipeline(d.p); pass.setBindGroup(0, d.bg); pass.dispatchWorkgroups(d.wx, d.wy); };
  const read = async (b: GPUBuffer, bytes: number) => {
    const rb = device.createBuffer({ size: bytes, usage: U_MAP_READ | U_COPY_DST });
    const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(b, 0, rb, 0, bytes);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(MAP_READ);
    const out = rb.getMappedRange().slice(0); rb.unmap(); rb.destroy();
    return out;
  };
  const encode = (k: number, withAdam: boolean) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = 0; i < k; i++) {
      for (const d of fwd) run(pass, d);
      for (const d of bwd) run(pass, d);
      if (withAdam) { run(pass, adam); run(pass, bump); }
    }
    pass.end();
    device.queue.submit([enc.finish()]);
  };
  const readState = async () => {
    const ab = await read(st, 16); const uu = new Uint32Array(ab), ff = new Float32Array(ab);
    return { step: uu[0], bestStep: uu[1], loss: ff[2], best: ff[3] };
  };
  const readParams = async (which: 'P' | 'best' | 'G') => new Float32Array(await read(which === 'P' ? P : which === 'best' ? best : G, nParams * 4));
  let used = false;
  const destroy = () => { for (const b of buffers) b.destroy(); device.destroy(); };
  return {
    device, info, nParams, readParams, destroy,
    async evaluate() { encode(1, false); const s = await readState(); return { loss: s.loss, grad: await readParams('G') }; },
    encodeIters: (k: number) => encode(k, true),
    done: () => device.queue.onSubmittedWorkDone(),
    async train(t) {
      if (used) throw new Error('a GpuStage2 trains once; create a new one');
      used = true;
      const n = Math.min(nIters, Math.max(1, Math.floor(t.iterations)));
      const per = Math.max(1, t.itersPerSubmit ?? 50);
      const t0 = performance.now();
      let lastTick = -Infinity, doneIt = 0;
      let endReason: TrainResult['endReason'] = 'completed';
      while (doneIt < n) {
        if (t.signal?.aborted) { endReason = 'stopped_by_user'; break; }
        if (lost) throw new Error('webgpu_device_lost');
        encode(Math.min(per, n - doneIt), true);
        const s = await readState();
        doneIt = s.step;
        if (!Number.isFinite(s.loss)) throw new Error(`non-finite loss ${s.loss} at iteration ${doneIt}`);
        const now = performance.now();
        if (t.onTick && (now - lastTick >= (t.tickEveryMs ?? 500) || doneIt === n)) {
          const el = (now - t0) / 1000;
          t.onTick({ iter: doneIt, total: n, loss: s.loss, best: s.best, bestIter: s.bestStep, lr: cosineLr(doneIt - 1, nIters, lr0, minFrac), elapsedS: el, etaS: doneIt >= 20 ? el / doneIt * (n - doneIt) : null });
          lastTick = now;
        }
      }
      if (doneIt === 0) throw new Error('training stopped before the first iteration');
      const s = await readState();
      const params = await readParams('best');
      const history = new Float32Array(await read(hist, doneIt * 4)).slice(0, doneIt);
      return { params, bestLoss: s.best, bestIter: s.bestStep, checkpointCriterion: 'best_train_loss', endReason, completedIters: doneIt, plannedIters: n, elapsedS: (performance.now() - t0) / 1000, history, path: 'webgpu' };
    },
  };
}
