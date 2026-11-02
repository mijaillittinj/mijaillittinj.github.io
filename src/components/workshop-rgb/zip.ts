/**
 * .zip without a library (WEB_SPEC §5.2 and §4 etapa 6). The reader takes the central directory and
 * each entry with Blob.slice, so an archive is never loaded whole, methods 0 (stored) and 8 (deflate
 * through DecompressionStream('deflate-raw')), no ZIP64. The writer stores without compression
 * ("Descargar todo"), with CRC-32. From the stage 1a maquette.
 */

export interface ZipEntry { path: string; size: number; blob: () => Promise<Blob> }

export class ZipError extends Error {
  code: string;
  constructor(code: string) { super(code); this.code = code; }
}

const u16 = (v: DataView, o: number) => v.getUint16(o, true);
const u32 = (v: DataView, o: number) => v.getUint32(o, true);

export async function readZip(file: Blob): Promise<ZipEntry[]> {
  const tailLen = Math.min(file.size, 65557);
  const tail = new DataView(await file.slice(file.size - tailLen).arrayBuffer());
  let eocd = -1;
  for (let i = tailLen - 22; i >= 0; i--) if (u32(tail, i) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new ZipError('zip_bad');
  const n = u16(tail, eocd + 10), cdSize = u32(tail, eocd + 12), cdOff = u32(tail, eocd + 16);
  if (cdOff === 0xffffffff || n === 0xffff) throw new ZipError('zip_bad');
  const cd = new DataView(await file.slice(cdOff, cdOff + cdSize).arrayBuffer());
  const dec = new TextDecoder();
  const out: ZipEntry[] = [];
  let p = 0;
  for (let k = 0; k < n; k++) {
    if (u32(cd, p) !== 0x02014b50) throw new ZipError('zip_bad');
    const method = u16(cd, p + 10), csize = u32(cd, p + 20), usize = u32(cd, p + 24);
    const nl = u16(cd, p + 28), xl = u16(cd, p + 30), cl = u16(cd, p + 32), lho = u32(cd, p + 42);
    const name = dec.decode(new Uint8Array(cd.buffer, cd.byteOffset + p + 46, nl));
    p += 46 + nl + xl + cl;
    if (name.endsWith('/')) continue;
    out.push({
      path: name, size: usize,
      blob: async () => {
        const lh = new DataView(await file.slice(lho, lho + 30).arrayBuffer());
        const start = lho + 30 + u16(lh, 26) + u16(lh, 28);
        const raw = file.slice(start, start + csize);
        if (method === 0) return raw;
        if (method !== 8) throw new ZipError('zip_method');
        if (typeof DecompressionStream === 'undefined') throw new ZipError('zip_unsupported');
        return await new Response(raw.stream().pipeThrough(new DecompressionStream('deflate-raw'))).blob();
      },
    });
  }
  return out;
}

let CRC: Uint32Array | null = null;
export function crc32(b: Uint8Array): number {
  if (!CRC) {
    CRC = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; CRC[n] = c >>> 0; }
  }
  let c = 0xffffffff;
  for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/** Stored (method 0) .zip of the given files. */
export function writeZip(files: { path: string; data: Uint8Array }[]): Blob {
  const enc = new TextEncoder();
  const parts: BlobPart[] = [];
  const central: Uint8Array[] = [];
  let off = 0;
  for (const f of files) {
    const name = enc.encode(f.path), crc = crc32(f.data), n = f.data.length;
    const lh = new DataView(new ArrayBuffer(30));
    lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true);
    lh.setUint16(10, 0, true); lh.setUint16(12, 0x21, true); lh.setUint32(14, crc, true); lh.setUint32(18, n, true); lh.setUint32(22, n, true);
    lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
    parts.push(lh.buffer, name as BlobPart, f.data as BlobPart);
    const ch = new DataView(new ArrayBuffer(46));
    ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true);
    ch.setUint16(12, 0, true); ch.setUint16(14, 0x21, true); ch.setUint32(16, crc, true); ch.setUint32(20, n, true); ch.setUint32(24, n, true);
    ch.setUint16(28, name.length, true); ch.setUint32(42, off, true);
    const c = new Uint8Array(46 + name.length);
    c.set(new Uint8Array(ch.buffer), 0); c.set(name, 46);
    central.push(c);
    off += 30 + name.length + n;
  }
  let cdLen = 0;
  for (const c of central) { parts.push(c as BlobPart); cdLen += c.length; }
  const e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
  e.setUint32(12, cdLen, true); e.setUint32(16, off, true);
  parts.push(e.buffer);
  return new Blob(parts, { type: 'application/zip' });
}
