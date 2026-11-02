/**
 * Minimal TIFF reader for the RGB pyrometry workshop tool (from the GB1 prototype,
 * manim/qa/web/proto/tiff/tiff.ts, with the GB1 round 1 encoding rules of WEB_SPEC §5.3).
 *
 * No dependencies. Reads baseline TIFF 6.0 as written by Basler pylon and by
 * common lab software:
 *   byte order II and MM, classic TIFF (magic 42), first IFD only;
 *   compression 1 (none), 5 (LZW, MSB first, with predictor 1 or 2) and 32773 (PackBits);
 *   8 or 16 bit unsigned integers, 32 bit float;
 *   1 sample (grey or Bayer mosaic) or 3 or 4 samples (RGB, alpha dropped by the caller);
 *   strips or tiles, planar configuration 1 (chunky) or 2 (planar).
 * The result is always chunky (pixel interleaved), row 0 at the top, in a typed array
 * of the platform byte order. Values are not rescaled.
 *
 * Errors are `TiffError` with a `code` the interface translates into plain words
 * (see MESSAGES in WEB_SPEC.md). Nothing here is written in Spanish or English prose.
 */

export class TiffError extends Error {
  code: string;
  detail: Record<string, number | string>;
  constructor(code: string, detail: Record<string, number | string> = {}) {
    super(`${code} ${JSON.stringify(detail)}`);
    this.code = code;
    this.detail = detail;
  }
}

export interface TiffImage {
  width: number;
  height: number;
  samples: number; // samples per pixel actually returned (1, 3 or 4)
  bits: number; // 8, 16 or 32
  float: boolean;
  data: Uint8Array | Uint16Array | Float32Array; // chunky, length width*height*samples
  littleEndian: boolean;
  compression: number;
  predictor: number;
  planar: number;
  photometric: number;
  /** CFA pattern from TIFF/EP tags 33421/33422 when present, e.g. 'RGGB'. */
  cfa?: string;
  software?: string;
  description?: string;
  /** EXIF orientation (tag 274), 1 when absent. */
  orientation: number;
  /** Camera model (tag 272) when present. */
  model?: string;
}

const T = {
  NewSubfileType: 254, ImageWidth: 256, ImageLength: 257, BitsPerSample: 258, Compression: 259,
  Photometric: 262, ImageDescription: 270, Model: 272, Orientation: 274, StripOffsets: 273, SamplesPerPixel: 277,
  RowsPerStrip: 278, StripByteCounts: 279, PlanarConfiguration: 284, Software: 305,
  Predictor: 317, TileWidth: 322, TileLength: 323, TileOffsets: 324, TileByteCounts: 325,
  SampleFormat: 339, CFARepeatPatternDim: 33421, CFAPattern: 33422,
} as const;

const TYPE_SIZE: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8 };

type Tags = Map<number, number[] | string>;

function readIfd(v: DataView, off: number, le: boolean): Tags {
  if (off + 2 > v.byteLength) throw new TiffError('truncated', { at: off });
  const n = v.getUint16(off, le);
  const tags: Tags = new Map();
  for (let k = 0; k < n; k++) {
    const e = off + 2 + 12 * k;
    if (e + 12 > v.byteLength) throw new TiffError('truncated', { at: e });
    const tag = v.getUint16(e, le), type = v.getUint16(e + 2, le), count = v.getUint32(e + 4, le);
    const size = TYPE_SIZE[type];
    if (!size) continue; // unknown type: skip, as the standard allows
    const bytes = size * count;
    const p = bytes <= 4 ? e + 8 : v.getUint32(e + 8, le);
    if (p + bytes > v.byteLength) throw new TiffError('truncated', { tag });
    if (type === 2) {
      let s = '';
      for (let i = 0; i < count; i++) { const c = v.getUint8(p + i); if (c === 0) break; s += String.fromCharCode(c); }
      tags.set(tag, s);
      continue;
    }
    const vals: number[] = new Array(count);
    for (let i = 0; i < count; i++) {
      const q = p + i * size;
      switch (type) {
        case 1: case 7: vals[i] = v.getUint8(q); break;
        case 6: vals[i] = v.getInt8(q); break;
        case 3: vals[i] = v.getUint16(q, le); break;
        case 8: vals[i] = v.getInt16(q, le); break;
        case 4: vals[i] = v.getUint32(q, le); break;
        case 9: vals[i] = v.getInt32(q, le); break;
        case 5: vals[i] = v.getUint32(q, le) / (v.getUint32(q + 4, le) || 1); break;
        case 10: vals[i] = v.getInt32(q, le) / (v.getInt32(q + 4, le) || 1); break;
        case 11: vals[i] = v.getFloat32(q, le); break;
        case 12: vals[i] = v.getFloat64(q, le); break;
        default: vals[i] = 0; // 16 (LONG8) only appears in BigTIFF
      }
    }
    tags.set(tag, vals);
  }
  return tags;
}

const num = (t: Tags, k: number, d?: number): number => {
  const x = t.get(k);
  if (Array.isArray(x) && x.length) return x[0];
  if (d === undefined) throw new TiffError('missing_tag', { tag: k });
  return d;
};
const arr = (t: Tags, k: number): number[] | undefined => {
  const x = t.get(k);
  return Array.isArray(x) ? x : undefined;
};

/** TIFF LZW (MSB first, early change). Writes into `out`, returns bytes written. */
export function lzwDecode(src: Uint8Array, out: Uint8Array): number {
  // Every table entry above 257 is a run of bytes already written to `out`: the string of
  // the previous code (written at prevAt) plus one byte, which is contiguous in `out`.
  const start = new Int32Array(4096), len = new Int32Array(4096);
  for (let i = 0; i < 256; i++) len[i] = 1;
  const cap = out.length;
  let next = 258, width = 9, prev = -1, prevAt = 0, o = 0;
  let bitBuf = 0, bitCnt = 0, ip = 0;
  for (;;) {
    while (bitCnt < width) {
      if (ip >= src.length) return o; // no EOI: the caller checks the length
      bitBuf = ((bitBuf & 0xffffff) << 8) | src[ip++];
      bitCnt += 8;
    }
    const code = (bitBuf >>> (bitCnt - width)) & ((1 << width) - 1);
    bitCnt -= width;
    if (code === 257) return o;
    if (code === 256) { next = 258; width = 9; prev = -1; continue; }
    const at = o;
    if (prev >= 0) {
      if (code > next) throw new TiffError('lzw_bad_code', { code, next });
      if (next < 4096) { start[next] = prevAt; len[next] = len[prev] + 1; next++; }
    } else if (code > 255) throw new TiffError('lzw_bad_code', { code });
    if (code < 256) {
      if (o >= cap) return o;
      out[o++] = code;
    } else {
      const s0 = start[code], L = len[code];
      if (o + L > cap) { for (let k = 0; o < cap; k++) out[o++] = out[s0 + k]; return o; }
      for (let k = 0; k < L; k++) out[o + k] = out[s0 + k]; // forward copy handles KwKwK
      o += L;
    }
    prev = code; prevAt = at;
    if (next === (1 << width) - 1 && width < 12) width++;
  }
}

/** PackBits (compression 32773). */
function packbitsDecode(src: Uint8Array, out: Uint8Array): number {
  let i = 0, o = 0;
  while (i < src.length && o < out.length) {
    const n = (src[i++] << 24) >> 24;
    if (n >= 0) { for (let k = 0; k <= n && o < out.length; k++) out[o++] = src[i++]; }
    else if (n !== -128) { const b = src[i++]; for (let k = 0; k < 1 - n && o < out.length; k++) out[o++] = b; }
  }
  return o;
}

export function readTiff(input: ArrayBuffer | Uint8Array): TiffImage {
  const u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
  const v = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (u8.length < 8) throw new TiffError('not_tiff');
  const bo = String.fromCharCode(u8[0], u8[1]);
  if (bo !== 'II' && bo !== 'MM') throw new TiffError('not_tiff');
  const le = bo === 'II';
  const magic = v.getUint16(2, le);
  if (magic === 43) throw new TiffError('bigtiff');
  if (magic !== 42) throw new TiffError('not_tiff');
  const tags = readIfd(v, v.getUint32(4, le), le);

  const W = num(tags, T.ImageWidth), H = num(tags, T.ImageLength);
  const spp = num(tags, T.SamplesPerPixel, 1);
  const bps = arr(tags, T.BitsPerSample) ?? [1];
  const bits = bps[0];
  if (bps.some((b) => b !== bits)) throw new TiffError('mixed_bits', { bits: bps.join(',') });
  const comp = num(tags, T.Compression, 1);
  const pred = num(tags, T.Predictor, 1);
  const planar = num(tags, T.PlanarConfiguration, 1);
  const photometric = num(tags, T.Photometric, spp >= 3 ? 2 : 1);
  const sfmt = num(tags, T.SampleFormat, 1);
  if (photometric === 6) throw new TiffError('ycbcr');
  if (![1, 5, 32773].includes(comp)) throw new TiffError('compression', { compression: comp });
  if (![1, 2].includes(pred)) throw new TiffError('predictor', { predictor: pred });
  const float = sfmt === 3;
  if (!((sfmt === 1 && (bits === 8 || bits === 16)) || (float && bits === 32)))
    throw new TiffError('sample_type', { bits, format: sfmt });
  if (![1, 3, 4].includes(spp)) throw new TiffError('samples', { samples: spp });
  if (W * H * spp > 64e6) throw new TiffError('too_large', { width: W, height: H });

  const bpsB = bits / 8;
  const tiled = tags.has(T.TileOffsets);
  const offs = arr(tags, tiled ? T.TileOffsets : T.StripOffsets);
  const cnts = arr(tags, tiled ? T.TileByteCounts : T.StripByteCounts);
  if (!offs || !cnts || offs.length !== cnts.length) throw new TiffError('missing_tag', { tag: tiled ? T.TileOffsets : T.StripOffsets });
  const tw = tiled ? num(tags, T.TileWidth) : W;
  const th = tiled ? num(tags, T.TileLength) : Math.min(num(tags, T.RowsPerStrip, H), H);
  const across = Math.ceil(W / tw), down = Math.ceil(H / th);
  const planes = planar === 2 ? spp : 1;
  const sppBlock = planar === 2 ? 1 : spp;
  if (offs.length < across * down * planes) throw new TiffError('missing_tag', { tag: T.StripOffsets });

  // decoded file bytes for the whole image, chunky, in file byte order
  const rowBytes = W * spp * bpsB;
  const raw = new Uint8Array(rowBytes * H);
  const blockRowBytes = tw * sppBlock * bpsB;
  const scratch = new Uint8Array(blockRowBytes * th);
  for (let p = 0; p < planes; p++) for (let by = 0; by < down; by++) for (let bx = 0; bx < across; bx++) {
    const k = p * across * down + by * across + bx;
    const src = u8.subarray(offs[k], offs[k] + cnts[k]);
    if (offs[k] + cnts[k] > u8.length) throw new TiffError('truncated', { block: k });
    const rows = tiled ? th : Math.min(th, H - by * th);
    const need = blockRowBytes * rows;
    const dst = scratch.subarray(0, need);
    let got: number;
    if (comp === 1) { dst.set(src.subarray(0, need)); got = Math.min(src.length, need); }
    else if (comp === 5) { dst.fill(0); got = lzwDecode(src, dst); }
    else { got = packbitsDecode(src, dst); }
    if (got < need) throw new TiffError('truncated', { block: k, got, need });
    if (pred === 2) undoPredictor(dst, tw, rows, sppBlock, bpsB, le);
    // place into raw
    for (let r = 0; r < rows; r++) {
      const y = by * th + r;
      if (y >= H) break;
      const x0 = bx * tw, cols = Math.min(tw, W - x0);
      if (planar === 1) {
        raw.set(dst.subarray(r * blockRowBytes, r * blockRowBytes + cols * spp * bpsB), y * rowBytes + x0 * spp * bpsB);
      } else {
        for (let c = 0; c < cols; c++) for (let b = 0; b < bpsB; b++)
          raw[y * rowBytes + ((x0 + c) * spp + p) * bpsB + b] = dst[r * blockRowBytes + c * bpsB + b];
      }
    }
  }

  const n = W * H * spp;
  let data: Uint8Array | Uint16Array | Float32Array;
  if (bits === 8) data = raw;
  else if (bits === 16) {
    const d = new Uint16Array(n);
    if (le) for (let i = 0; i < n; i++) d[i] = raw[2 * i] | (raw[2 * i + 1] << 8);
    else for (let i = 0; i < n; i++) d[i] = (raw[2 * i] << 8) | raw[2 * i + 1];
    data = d;
  } else {
    const d = new Float32Array(n), dv = new DataView(raw.buffer);
    for (let i = 0; i < n; i++) d[i] = dv.getFloat32(4 * i, le);
    data = d;
  }

  let cfa: string | undefined;
  const dim = arr(tags, T.CFARepeatPatternDim), pat = arr(tags, T.CFAPattern);
  if (dim && pat && dim[0] === 2 && dim[1] === 2 && pat.length === 4) cfa = pat.map((c) => 'RGB'[c] ?? '?').join('');
  const sw = tags.get(T.Software), desc = tags.get(T.ImageDescription), model = tags.get(T.Model);
  return {
    width: W, height: H, samples: spp, bits, float, data, littleEndian: le, compression: comp,
    predictor: pred, planar, photometric, cfa,
    software: typeof sw === 'string' ? sw : undefined,
    description: typeof desc === 'string' ? desc : undefined,
    orientation: num(tags, T.Orientation, 1),
    model: typeof model === 'string' ? model : undefined,
  };
}

/** Horizontal differencing (predictor 2), per row and per sample, modulo 2^bits. */
function undoPredictor(b: Uint8Array, w: number, rows: number, spp: number, bpsB: number, le: boolean) {
  if (bpsB === 1) {
    for (let r = 0; r < rows; r++) { const o = r * w * spp; for (let i = spp; i < w * spp; i++) b[o + i] = (b[o + i] + b[o + i - spp]) & 255; }
    return;
  }
  if (bpsB !== 2) throw new TiffError('predictor', { predictor: 2, bits: bpsB * 8 });
  const get = le ? (i: number) => b[i] | (b[i + 1] << 8) : (i: number) => (b[i] << 8) | b[i + 1];
  const put = le ? (i: number, x: number) => { b[i] = x & 255; b[i + 1] = x >>> 8; } : (i: number, x: number) => { b[i] = x >>> 8; b[i + 1] = x & 255; };
  for (let r = 0; r < rows; r++) {
    const o = r * w * spp * 2;
    for (let i = spp; i < w * spp; i++) put(o + 2 * i, (get(o + 2 * i) + get(o + 2 * (i - spp))) & 0xffff);
  }
}

export type Encoding = 'u8' | 'u12_shifted' | 'u12' | 'u16' | 'float';

export interface EncodingInfo {
  encoding: Encoding;
  /** Full scale in file counts (65520, a measured lower ceiling, 4095, 65535, 255, NaN for float). */
  ceiling: number;
  /** Nominal full scale of the encoding before the histogram check. */
  nominalCeiling: number;
  /** True when the ceiling came from a pile-up in the histogram below the nominal one. */
  measured: boolean;
  /** u12 (max ≤ 4095) cannot be told from a dim 16 bit frame; float has no ceiling. */
  ambiguous: boolean;
  /** Fraction of the non-zero values that are multiples of 16. */
  fracShifted: number;
  /** Factor from file counts to the common unit, DN of 12 bits (1/16 for shifted and 16 bit, 1 for u12). */
  toDn12: number;
  max: number;
  nonZero: number;
}

/** Tolerance of the shifted test: at least this fraction of the non-zero values are multiples of 16. */
export const SHIFT_TOLERANCE = 0.9999;
/** Minimum count at the top value, and its ratio to the next populated level below, to call it a pile-up. */
export const PILEUP_MIN_COUNT = 8;
export const PILEUP_RATIO = 4;

/**
 * Encoding and full scale of an integer frame, from the data (as radiometric_calibration.load_raw_tiff
 * and annabel_app io.read_rgb_tiff do), with the cases of WEB_SPEC §5.3.
 *   ≥ 99,99 % of the non-zero values multiples of 16   12 bits shifted left by 4, ceiling 65520 or the
 *                                                       measured pile-up below it
 *   8 bit container                                     255 (rejected for measuring, u8_gamma)
 *   16 bit container, max ≤ 4095                        12 bits not shifted, ceiling 4095 (ambiguous)
 *   otherwise                                           65535
 * The measured ceiling is the top value v when count(v) ≥ PILEUP_MIN_COUNT and count(v) ≥ PILEUP_RATIO ×
 * the count of the next populated value below (a saturation pile-up), and v < the nominal ceiling.
 */
export function detectEncoding(img: Pick<TiffImage, 'float' | 'bits' | 'data'>): EncodingInfo {
  if (img.float) return { encoding: 'float', ceiling: NaN, nominalCeiling: NaN, measured: false, ambiguous: true, fracShifted: 0, toDn12: 1, max: NaN, nonZero: 0 };
  const d = img.data as Uint16Array | Uint8Array;
  let max = 0, nz = 0, mult = 0;
  const hist = new Uint32Array(img.bits === 8 ? 256 : 65536);
  for (let i = 0; i < d.length; i++) {
    const x = d[i];
    hist[x]++;
    if (x) { nz++; if ((x & 15) === 0) mult++; if (x > max) max = x; }
  }
  const fracShifted = nz ? mult / nz : 0;
  if (img.bits === 8) return { encoding: 'u8', ceiling: 255, nominalCeiling: 255, measured: false, ambiguous: false, fracShifted, toDn12: 4095 / 255, max, nonZero: nz };
  let encoding: Encoding, nominal: number, toDn12: number, ambiguous = false;
  if (nz && fracShifted >= SHIFT_TOLERANCE && max > 4095) { encoding = 'u12_shifted'; nominal = 65520; toDn12 = 1 / 16; }
  else if (nz && fracShifted >= SHIFT_TOLERANCE && max <= 4095 && max % 16 === 0 && max >= 16 && mult === nz) {
    // every value a multiple of 16 and nothing above 4095: a very dim shifted frame or 12 bits unshifted
    encoding = 'u12_shifted'; nominal = 65520; toDn12 = 1 / 16; ambiguous = true;
  } else if (max <= 4095) { encoding = 'u12'; nominal = 4095; toDn12 = 1; ambiguous = true; }
  else { encoding = 'u16'; nominal = 65535; toDn12 = 1 / 16; }
  let ceiling = nominal, measured = false;
  if (max > 0 && max < nominal && hist[max] >= PILEUP_MIN_COUNT) {
    let below = 0;
    for (let v = max - 1; v > 0; v--) if (hist[v]) { below = hist[v]; break; }
    if (hist[max] >= PILEUP_RATIO * below) { ceiling = max; measured = true; }
  }
  return { encoding, ceiling, nominalCeiling: nominal, measured, ambiguous, fracShifted, toDn12, max, nonZero: nz };
}

/** Compatibility wrapper of the GB1 prototype API. */
export function detectCeiling(img: TiffImage): { ceiling: number; encoding: Encoding; ambiguous: boolean } {
  const e = detectEncoding(img);
  return { ceiling: e.ceiling, encoding: e.encoding, ambiguous: e.ambiguous };
}
