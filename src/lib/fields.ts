/** Loading helpers for axisymmetric fields exported by scripts/export_*.py. */
import { loadU8Field } from './plot.ts';

export interface FieldMeta {
  nz: number;
  nr: number;
  r_mm: [number, number];
  z_mm: [number, number];
  source: string;
  fields: Record<string, { min: number; max: number; unit: string; label: string }>;
}

const metaCache = new Map<string, Promise<FieldMeta>>();
export function loadMeta(base: string): Promise<FieldMeta> {
  if (!metaCache.has(base)) metaCache.set(base, fetch(base + 'meta.json').then((r) => r.json()));
  return metaCache.get(base)!;
}

export async function loadField(base: string, key: string) {
  const meta = await loadMeta(base);
  const f = meta.fields[key];
  const data = await loadU8Field(`${base}${key}.u8`, f.min, f.max);
  return { meta, data, info: f };
}

/**
 * First row (from the bottom) whose finite values are contiguous from the axis,
 * i.e. no interior gap. Below it, part of the field is missing, so it cannot be
 * projected without inventing data.
 */
export function firstCompleteRow(data: Float32Array, nz: number, nr: number): number {
  for (let j = 0; j < nz; j++) {
    let last = -1, gap = false;
    for (let i = 0; i < nr; i++) if (Number.isFinite(data[j * nr + i])) last = i;
    for (let i = 0; i < last; i++) if (!Number.isFinite(data[j * nr + i])) gap = true;
    if (!gap && last > 0) return j;
  }
  return 0;
}

/** Crop rows below j0 and replace outer masked cells by 0 (no emitter outside the flame). */
export function projectable(data: Float32Array, nz: number, nr: number, j0: number): Float32Array {
  const NZ = nz - j0;
  const out = new Float32Array(NZ * nr);
  for (let j = 0; j < NZ; j++) for (let i = 0; i < nr; i++) {
    const v = data[(j + j0) * nr + i];
    out[j * nr + i] = Number.isFinite(v) ? v : 0;
  }
  return out;
}
