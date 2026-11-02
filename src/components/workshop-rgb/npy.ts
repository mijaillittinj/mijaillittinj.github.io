/**
 * .npy and .npz writers without compression (WEB_SPEC §4 etapa 6 and etapa 3). Float32, Float64 and
 * int32 arrays in C order, little endian, and 0-d unicode strings for the meta_json of the ANNAbel
 * projections contract (annabel_app/core/io.py).
 */
import { writeZip } from './zip.ts';

type Num = Float32Array | Float64Array | Int32Array | Uint8Array;

function header(descr: string, shape: number[]): Uint8Array {
  const sh = shape.length === 0 ? '()' : shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(', ')})`;
  let h = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${sh}, }`;
  const total = 10 + h.length + 1;
  h += ' '.repeat((64 - (total % 64)) % 64) + '\n';
  const out = new Uint8Array(10 + h.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  out[8] = h.length & 255; out[9] = h.length >> 8;
  for (let i = 0; i < h.length; i++) out[10 + i] = h.charCodeAt(i);
  return out;
}

const descrOf = (a: Num) => (a instanceof Float32Array ? '<f4' : a instanceof Float64Array ? '<f8' : a instanceof Int32Array ? '<i4' : '|u1');

/** One .npy file (little endian platforms, which every browser in use is). */
export function npy(a: Num, shape: number[]): Uint8Array {
  const h = header(descrOf(a), shape);
  const body = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const out = new Uint8Array(h.length + body.length);
  out.set(h, 0); out.set(body, h.length);
  return out;
}

/** 0-d unicode array, numpy dtype <U{n}, UTF-32 little endian. */
export function npyString(s: string): Uint8Array {
  const cps = Array.from(s, (c) => c.codePointAt(0)!);
  const n = Math.max(1, cps.length);
  const h = header(`<U${n}`, []);
  const body = new Uint32Array(n);
  cps.forEach((c, i) => (body[i] = c));
  const out = new Uint8Array(h.length + body.byteLength);
  out.set(h, 0); out.set(new Uint8Array(body.buffer), h.length);
  return out;
}

export function npz(arrays: Record<string, Uint8Array>): Blob {
  return writeZip(Object.entries(arrays).map(([k, v]) => ({ path: `${k}.npy`, data: v })));
}
