// WebGPU path of ANNAbel stage 1. The whole iteration stays on the GPU in f32: forward pass,
// Abel projection of both halves, loss, lowest-loss checkpoint taken before the step,
// backpropagation, Adam with the cosine schedule. Many iterations go in one submit, and the
// host reads a 16-byte state between submits. Same parameter layout and loss as net.ts.
import type { Stage1Problem } from './problem.ts';
import { SIZES, layout } from './net.ts';
import type { TrainOptions, TrainResult } from './train.ts';
import { DEFAULT_LR, DEFAULT_LR_MIN_FRAC, cosineLr } from './train.ts';

// GPUBufferUsage and GPUMapMode values (the constants are not declared in TypeScript's lib.dom).
const U_MAP_READ = 0x1, U_COPY_SRC = 0x4, U_COPY_DST = 0x8, U_UNIFORM = 0x40, U_STORAGE = 0x80;
const MAP_READ = 0x1;

export const WG = 256;
export const UHDR = `struct U { N:u32, K:u32, M:u32, wOff:u32, bOff:u32, act:u32, nsplit:u32, chunk:u32,
  nr:u32, rows:u32, nP:u32, O:u32, os:f32, invM:f32, w0:f32, w1:f32, w2:f32, b1:f32, b2:f32, eps:f32 };
@group(0) @binding(0) var<uniform> u: U;
fn gidx(g: vec3u, n: vec3u) -> u32 { return g.x + g.y * n.x * ${WG}u; }
`;
export const ST = `struct St { step:u32, bestStep:u32, loss:f32, best:f32 };`;

export const K_LIN_FWD = UHDR + `
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> inp: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let idx = gidx(g, nw); if (idx >= u.N * u.M) { return; }
  let n = idx / u.M; let m = idx % u.M;
  var acc = P[u.bOff + m];
  let xi = n * u.K; let wi = u.wOff + m * u.K;
  for (var k = 0u; k < u.K; k++) { acc += inp[xi + k] * P[wi + k]; }
  if (u.act == 1u) { acc = tanh(clamp(acc, -15.0, 15.0)); }
  outp[idx] = acc;
}`;

// Jn = out_scale softplus(y) (torch threshold 20), S = out_scale sigmoid(y)
const K_OUT = UHDR + `
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read_write> Jn: array<f32>;
@group(0) @binding(3) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let idx = gidx(g, nw); if (idx >= u.N * 3u) { return; }
  let y = Y[idx]; let e = exp(-abs(y));
  Jn[idx] = u.os * select(max(y, 0.0) + log(1.0 + e), y, y > 20.0);
  S[idx] = u.os * select(e / (1.0 + e), 1.0 / (1.0 + e), y >= 0.0);
}`;

const K_ABEL_FWD = UHDR + `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> Jn: array<f32>;
@group(0) @binding(3) var<storage, read> bR: array<f32>;
@group(0) @binding(4) var<storage, read> bL: array<f32>;
@group(0) @binding(5) var<storage, read_write> dP: array<f32>;
@group(0) @binding(6) var<storage, read_write> lt: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let idx = gidx(g, nw); if (idx >= u.N * 3u) { return; }
  let c = idx % 3u; let s = idx / 3u; let z = s / u.nr; let i = s % u.nr;
  var p = 0.0;
  let jb = z * u.nr * 3u + c;
  for (var j = i; j < u.nr; j++) { p += A[i * u.nr + j] * Jn[jb + j * 3u]; }
  let eR = p - bR[idx]; let eL = p - bL[idx];
  var w = u.w0; if (c == 1u) { w = u.w1; } if (c == 2u) { w = u.w2; }
  lt[idx] = (eR * eR + eL * eL) * w;
  dP[idx] = (eR + eL) * w * u.invM;
}`;

// loss = 0.5 Σ lt / M, history, and the checkpoint copy when the loss improves (before the step)
const K_LOSS = UHDR + ST + `
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
  for (var q = t; q < u.N * 3u; q += ${WG}u) { s += lt[q]; }
  sh[t] = s;
  workgroupBarrier();
  for (var h = ${WG / 2}u; h > 0u; h >>= 1u) { if (t < h) { sh[t] += sh[t + h]; } workgroupBarrier(); }
  if (t == 0u) {
    let L = 0.5 * sh[0] * u.invM;
    st.loss = L; hist[st.step] = L;
    if (L < st.best) { st.best = L; st.bestStep = st.step + 1u; flag = 1u; } else { flag = 0u; }
  }
  workgroupBarrier();
  if (flag == 1u) { for (var q = t; q < u.nP; q += ${WG}u) { best[q] = P[q]; } }
}`;

const K_ABEL_BWD = UHDR + `
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> dP: array<f32>;
@group(0) @binding(3) var<storage, read> S: array<f32>;
@group(0) @binding(4) var<storage, read_write> dY: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let idx = gidx(g, nw); if (idx >= u.N * 3u) { return; }
  let c = idx % 3u; let s = idx / 3u; let z = s / u.nr; let j = s % u.nr;
  var d = 0.0;
  let ib = z * u.nr * 3u + c;
  for (var i = 0u; i <= j; i++) { d += A[i * u.nr + j] * dP[ib + i * 3u]; }
  dY[idx] = d * S[idx];
}`;

export const K_DX = UHDR + `
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> dY: array<f32>;
@group(0) @binding(3) var<storage, read> H: array<f32>;
@group(0) @binding(4) var<storage, read_write> dX: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let idx = gidx(g, nw); if (idx >= u.N * u.K) { return; }
  let n = idx / u.K; let k = idx % u.K;
  var a = 0.0;
  for (var m = 0u; m < u.M; m++) { a += dY[n * u.M + m] * P[u.wOff + m * u.K + k]; }
  let h = H[idx];
  dX[idx] = a * (1.0 - h * h);
}`;

// weight gradient split over samples: workgroup w sums samples [w·chunk, (w+1)·chunk)
// through a shared-memory tile of 16 samples; then a reduction over the partitions
export const TS = 16;
export const K_DW = UHDR + `
@group(0) @binding(1) var<storage, read> dY: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
var<workgroup> tx: array<f32, ${TS * 64}>;
var<workgroup> td: array<f32, ${TS * 64}>;
@compute @workgroup_size(${WG}) fn main(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) wg: vec3u) {
  let t = l.x; let w = wg.x;
  let n0 = w * u.chunk; let n1 = min(u.N, n0 + u.chunk);
  var acc: array<f32, 17>;
  for (var r = 0u; r < 17u; r++) { acc[r] = 0.0; }
  for (var nb = n0; nb < n1; nb += ${TS}u) {
    let cnt = min(${TS}u, n1 - nb);
    for (var q = t; q < ${TS}u * u.K; q += ${WG}u) {
      let s = q / u.K; tx[q] = select(0.0, X[(nb + s) * u.K + q % u.K], s < cnt); }
    for (var q = t; q < ${TS}u * u.M; q += ${WG}u) {
      let s = q / u.M; td[q] = select(0.0, dY[(nb + s) * u.M + q % u.M], s < cnt); }
    workgroupBarrier();
    for (var r = 0u; r < 17u; r++) {
      let o = t + r * ${WG}u;
      if (o < u.O) {
        var a = 0.0;
        if (o < u.M * u.K) {
          let m = o / u.K; let k = o % u.K;
          for (var s = 0u; s < ${TS}u; s++) { a += td[s * u.M + m] * tx[s * u.K + k]; }
        } else {
          let m = o - u.M * u.K;
          for (var s = 0u; s < ${TS}u; s++) { a += td[s * u.M + m]; }
        }
        acc[r] += a;
      }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < 17u; r++) { let o = t + r * ${WG}u; if (o < u.O) { part[w * u.O + o] = acc[r]; } }
}`;

export const K_DWRED = UHDR + `
@group(0) @binding(1) var<storage, read> part: array<f32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let o = gidx(g, nw); if (o >= u.O) { return; }
  var s = 0.0;
  for (var w = 0u; w < u.nsplit; w++) { s += part[w * u.O + o]; }
  if (o < u.M * u.K) { G[u.wOff + o] = s; } else { G[u.bOff + o - u.M * u.K] = s; }
}`;

// Adam; ss[t] = lr_t / (1 − β1^t) and c2[t] = √(1 − β2^t) precomputed in float64 on the host
export const K_ADAM = UHDR + ST + `
@group(0) @binding(1) var<storage, read_write> P: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read_write> m1: array<f32>;
@group(0) @binding(4) var<storage, read_write> v1: array<f32>;
@group(0) @binding(5) var<storage, read> st: St;
@group(0) @binding(6) var<storage, read> lrs: array<f32>;
@group(0) @binding(7) var<storage, read> bc: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(num_workgroups) nw: vec3u) {
  let q = gidx(g, nw); if (q >= u.nP) { return; }
  let gq = G[q];
  let m = u.b1 * m1[q] + (1.0 - u.b1) * gq;
  let v = u.b2 * v1[q] + (1.0 - u.b2) * gq * gq;
  m1[q] = m; v1[q] = v;
  let ss = lrs[st.step]; let c2 = bc[st.step];
  P[q] -= ss * m / (sqrt(v) / c2 + u.eps);
}`;

export const K_BUMP = UHDR + ST + `
@group(0) @binding(1) var<storage, read_write> st: St;
@compute @workgroup_size(1) fn main() { if (u.N > 0u) { st.step += 1u; } }`;

export interface GpuInfo { vendor: string; architecture: string; device: string; description: string }

/** True when the browser exposes WebGPU and gives an adapter. */
export async function webgpuAvailable(): Promise<boolean> {
  try {
    const nav = (globalThis as { navigator?: Navigator }).navigator;
    if (!nav || !('gpu' in nav) || !nav.gpu) return false;
    return !!(await nav.gpu.requestAdapter());
  } catch { return false; }
}

export interface GpuStage1 {
  device: GPUDevice;
  info: GpuInfo;
  nParams: number;
  /** Loss and gradient at the current parameters, without a step (parity checks). */
  evaluate(): Promise<{ loss: number; grad: Float32Array }>;
  /** Full training from the initial parameters (stage 1 of the app). */
  train(o: TrainOptions & { itersPerSubmit?: number }): Promise<TrainResult>;
  /** Encode k full iterations (timing). */
  encodeIters(k: number): void;
  done(): Promise<void>;
  readParams(which: 'P' | 'best' | 'G'): Promise<Float32Array>;
  readState(): Promise<{ step: number; bestStep: number; loss: number; best: number }>;
  destroy(): void;
}

/** Sets up buffers and pipelines for one problem, initial parameters p0 and a schedule of
 *  `iterations` (T_max of the cosine). Throws if WebGPU or an adapter is missing. */
export async function createGpuStage1(prob: Stage1Problem, p0: Float32Array, iterations: number,
                                      o: { lr?: number; lrMinFrac?: number } = {}): Promise<GpuStage1> {
  const nav = (globalThis as { navigator?: Navigator }).navigator;
  if (!nav || !('gpu' in nav) || !nav.gpu) throw new Error('webgpu_missing');
  const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('webgpu_no_adapter');
  const { wOff, bOff, nParams } = layout(SIZES);
  if (p0.length !== nParams) throw new Error('parameter vector does not match 3 → 64 ×4 → 3');
  const N = prob.N, nr = prob.nr, L = SIZES.length - 1;
  const bigBytes = N * 64 * 4;
  const lim = adapter.limits;
  if (lim.maxStorageBuffersPerShaderStage < 7) throw new Error('webgpu_limits');
  if (bigBytes > lim.maxStorageBufferBindingSize) throw new Error('webgpu_limits');
  const req: Record<string, number> = { maxStorageBuffersPerShaderStage: Math.min(8, lim.maxStorageBuffersPerShaderStage) };
  if (bigBytes > 134217728) { req.maxStorageBufferBindingSize = lim.maxStorageBufferBindingSize; req.maxBufferSize = lim.maxBufferSize; }
  const device = await adapter.requestDevice({ requiredLimits: req });
  const ai = adapter.info;
  const info: GpuInfo = { vendor: ai?.vendor ?? '', architecture: ai?.architecture ?? '', device: ai?.device ?? '', description: ai?.description ?? '' };
  const nIters = Math.max(1, Math.floor(iterations));
  const lr0 = o.lr ?? DEFAULT_LR, minFrac = o.lrMinFrac ?? DEFAULT_LR_MIN_FRAC;
  const buffers: GPUBuffer[] = [];
  const buf = (bytes: number, data?: ArrayBufferView) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: U_STORAGE | U_COPY_SRC | U_COPY_DST });
    if (data) device.queue.writeBuffer(b, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    buffers.push(b);
    return b;
  };
  const P = buf(nParams * 4, p0), G = buf(nParams * 4), m1 = buf(nParams * 4), v1 = buf(nParams * 4);
  const best = buf(nParams * 4, p0);
  const H: GPUBuffer[] = [buf(N * 3 * 4, prob.X)];
  for (let l = 1; l < L; l++) H.push(buf(N * SIZES[l] * 4));
  const Y = buf(N * 12), Jn = buf(N * 12), S = buf(N * 12), dP = buf(N * 12), lt = buf(N * 12);
  const A = buf(nr * nr * 4, prob.A), bR = buf(N * 12, prob.bR), bL = buf(N * 12, prob.bL);
  const DA = buf(bigBytes), DB = buf(bigBytes);
  const NSPLIT = 128, chunk = Math.ceil(N / NSPLIT);
  const part = buf(NSPLIT * (64 * 64 + 64) * 4);
  const st = buf(16, new Uint32Array([0, 0, 0, 0x7f7fffff]));
  const hist = buf(nIters * 4);
  const ss = new Float32Array(nIters), c2 = new Float32Array(nIters);
  for (let e = 0; e < nIters; e++) {
    ss[e] = cosineLr(e, nIters, lr0, minFrac) / (1 - 0.9 ** (e + 1));
    c2[e] = Math.sqrt(1 - 0.999 ** (e + 1));
  }
  const lrs = buf(nIters * 4, ss), bc = buf(nIters * 4, c2);

  const uni = (ov: Record<string, number>) => {
    const ab = new ArrayBuffer(80), dv = new DataView(ab);
    const UK = ['N', 'K', 'M', 'wOff', 'bOff', 'act', 'nsplit', 'chunk', 'nr', 'rows', 'nP', 'O'];
    const FK = ['os', 'invM', 'w0', 'w1', 'w2', 'b1', 'b2', 'eps'];
    const base: Record<string, number> = { N, nr, rows: prob.rows, nP: nParams, nsplit: NSPLIT, chunk,
      os: prob.outScale, invM: 1 / prob.M, w0: prob.invNrm2[0], w1: prob.invNrm2[1], w2: prob.invNrm2[2],
      b1: 0.9, b2: 0.999, eps: 1e-8, ...ov };
    UK.forEach((k, i) => dv.setUint32(i * 4, base[k] ?? 0, true));
    FK.forEach((k, i) => dv.setFloat32(48 + i * 4, base[k] ?? 0, true));
    const b = device.createBuffer({ size: 80, usage: U_UNIFORM | U_COPY_DST });
    device.queue.writeBuffer(b, 0, ab);
    buffers.push(b);
    return b;
  };
  const pipes = new Map<string, GPUComputePipeline>();
  const pipe = (code: string) => {
    let p = pipes.get(code);
    if (!p) {
      p = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
      pipes.set(code, p);
    }
    return p;
  };
  interface D { p: GPUComputePipeline; bg: GPUBindGroup; wx: number; wy: number }
  const mk = (code: string, u: GPUBuffer, bufs: GPUBuffer[], threads: number, flat = false): D => {
    const p = pipe(code);
    const bg = device.createBindGroup({ layout: p.getBindGroupLayout(0),
      entries: [u, ...bufs].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    const wgs = flat ? threads : Math.ceil(threads / WG);
    const wx = Math.min(wgs, 65535), wy = Math.ceil(wgs / wx);
    return { p, bg, wx, wy };
  };
  device.pushErrorScope('validation');
  const fwd: D[] = [], bwd: D[] = [];
  for (let l = 0; l < L; l++) {
    const K = SIZES[l], M = SIZES[l + 1];
    fwd.push(mk(K_LIN_FWD, uni({ K, M, wOff: wOff[l], bOff: bOff[l], act: l < L - 1 ? 1 : 0 }),
      [P, H[l], l < L - 1 ? H[l + 1] : Y], N * M));
  }
  const u0 = uni({});
  fwd.push(mk(K_OUT, u0, [Y, Jn, S], N * 3));
  fwd.push(mk(K_ABEL_FWD, u0, [A, Jn, bR, bL, dP, lt], N * 3));
  fwd.push(mk(K_LOSS, u0, [lt, P, best, st, hist], 1, true));
  bwd.push(mk(K_ABEL_BWD, u0, [A, dP, S, DA], N * 3));
  let dY = DA, dX = DB;
  for (let l = L - 1; l >= 0; l--) {
    const K = SIZES[l], M = SIZES[l + 1], O = M * K + M;
    const ul = uni({ K, M, wOff: wOff[l], bOff: bOff[l], O });
    bwd.push(mk(K_DW, ul, [dY, H[l], part], NSPLIT, true));
    bwd.push(mk(K_DWRED, ul, [part, G], O));
    if (l > 0) { bwd.push(mk(K_DX, ul, [P, dY, H[l], dX], N * K)); const t = dY; dY = dX; dX = t; }
  }
  const adam = mk(K_ADAM, u0, [P, G, m1, v1, st, lrs, bc], nParams);
  const bump = mk(K_BUMP, u0, [st], 1, true);
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
  const readParams = async (which: 'P' | 'best' | 'G') =>
    new Float32Array(await read(which === 'P' ? P : which === 'best' ? best : G, nParams * 4));
  let used = false;
  const destroy = () => { for (const b of buffers) b.destroy(); device.destroy(); };

  return {
    device, info, nParams,
    async evaluate() {
      encode(1, false);
      const s = await readState();
      return { loss: s.loss, grad: await readParams('G') };
    },
    encodeIters: (k: number) => encode(k, true),
    done: () => device.queue.onSubmittedWorkDone(),
    readParams, readState, destroy,
    async train(t) {
      if (used) throw new Error('a GpuStage1 trains once; create a new one');
      used = true;
      const n = Math.min(nIters, Math.max(1, Math.floor(t.iterations)));
      const per = Math.max(1, t.itersPerSubmit ?? 50);
      const t0 = performance.now();
      let lastTick = -Infinity, doneIt = 0;
      let endReason: TrainResult['endReason'] = 'completed';
      while (doneIt < n) {
        if (t.signal?.aborted) { endReason = 'stopped_by_user'; break; }
        if (lost) throw new Error('webgpu_device_lost');
        const k = Math.min(per, n - doneIt);
        encode(k, true);
        const s = await readState();   // waits for the submit
        doneIt = s.step;
        if (!Number.isFinite(s.loss)) throw new Error(`non-finite loss ${s.loss} at iteration ${doneIt}`);
        const now = performance.now();
        if (t.onTick && (now - lastTick >= (t.tickEveryMs ?? 500) || doneIt === n)) {
          const el = (now - t0) / 1000;
          t.onTick({ iter: doneIt, total: n, loss: s.loss, best: s.best, bestIter: s.bestStep,
                     lr: cosineLr(doneIt - 1, nIters, lr0, minFrac), elapsedS: el, etaS: doneIt >= 20 ? el / doneIt * (n - doneIt) : null });
          lastTick = now;
        }
      }
      if (doneIt === 0) throw new Error('training stopped before the first iteration');
      const s = await readState();
      const params = await readParams('best');
      const history = new Float32Array(await read(hist, doneIt * 4)).slice(0, doneIt);
      return { params, bestLoss: s.best, bestIter: s.bestStep, checkpointCriterion: 'best_train_loss', endReason,
               completedIters: doneIt, plannedIters: n, elapsedS: (performance.now() - t0) / 1000, history, path: 'webgpu' };
    },
  };
}
