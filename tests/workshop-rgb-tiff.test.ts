// TIFF reader of the workshop tool against true arrays: run with `node --experimental-strip-types tests/workshop-rgb-tiff.test.ts`
// Fixtures from manim/tools/make_web_fixtures.py: crops of the three real families (flame LZW II written by
// Photo Viewer, sphere MM, oven MM), 12 reduced synthetic variants from three encoders, four error files.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { readTiff, detectEncoding, TiffError } from '../src/lib/workshop-rgb/tiff.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const ref = JSON.parse(readFileSync(fx('tiff_ref.json'), 'utf8'));

const sha = (d: Uint8Array | Uint16Array | Float32Array) => {
  let bytes: Uint8Array;
  if (d instanceof Uint16Array) { bytes = new Uint8Array(d.length * 2); for (let i = 0; i < d.length; i++) { bytes[2 * i] = d[i] & 255; bytes[2 * i + 1] = d[i] >>> 8; } }
  else if (d instanceof Uint8Array) bytes = d;
  else bytes = new Uint8Array(d.buffer, d.byteOffset, d.byteLength);
  return createHash('sha256').update(bytes).digest('hex');
};

// 1. every fixture decodes to the true array (SHA-256 of the H × W × C array, exact)
for (const [name, r] of Object.entries(ref.files) as [string, any][]) {
  const img = readTiff(readFileSync(fx(`tiff/${name}`)));
  const shape = [img.height, img.width, img.samples];
  const okShape = shape.join(',') === r.shape.join(',');
  const okSha = sha(img.data) === r.sha256;
  const sums = Array.from({ length: img.samples }, (_, c) => { let s = 0; for (let i = c; i < img.data.length; i += img.samples) s += img.data[i]; return s; });
  check(`${name} (${r.family})`, okShape && okSha && sums.join(',') === r.sum.join(','), `${shape.join('×')} sha ${okSha ? 'equal' : 'DIFFERENT'}`);
  const e = detectEncoding(img);
  check(`${name} encoding ${e.encoding}, ceiling ${e.ceiling}`, e.encoding === r.encoding.encoding && e.ceiling === r.encoding.ceiling && (r.encoding.measured === undefined || e.measured === r.encoding.measured));
}

// 2. the real flame keeps its Photo Viewer software tag (rewritten warning) and its LZW strips without EOI
{
  const img = readTiff(readFileSync(fx('tiff/flame_lzw_II_photoviewer_16rows.tif')));
  check('flame strips are LZW II, software tag read', img.compression === 5 && img.littleEndian && /Photo Viewer/.test(img.software ?? ''), `(${img.software})`);
}

// 3. errors come out with their code
for (const [name, code] of Object.entries(ref.errors) as [string, string][]) {
  let got = '';
  try { readTiff(readFileSync(fx(`tiff/${name}`))); got = 'no error'; } catch (e) { got = e instanceof TiffError ? e.code : String(e); }
  check(`${name} gives ${code}`, got === code, `(got ${got})`);
}

// 4. encoding rules of WEB_SPEC §5.3 on small synthetic arrays
{
  const n = 20000;
  const d = new Uint16Array(n);
  for (let i = 0; i < n; i++) d[i] = 16 * (100 + (i % 3000));
  d[7] = 16 * 200 + 3; // one value in 20 000 not a multiple of 16: still shifted (tolerance 99,99 %)
  let e = detectEncoding({ float: false, bits: 16, data: d });
  check('99,99 % tolerance keeps 12 bits shifted with one odd value', e.encoding === 'u12_shifted' && e.ceiling === 65520 && e.fracShifted < 1);
  d[8] = 16 * 200 + 5; d[9] = 16 * 200 + 7; d[10] = 16 * 200 + 9;
  e = detectEncoding({ float: false, bits: 16, data: d });
  check('four odd values in 20 000 (99,98 %) is 16 bit', e.encoding === 'u16' && e.ceiling === 65535);
  const p = new Uint16Array(n);
  for (let i = 0; i < n; i++) p[i] = Math.min(16 * (1000 + (i % 4000)), 65504);
  e = detectEncoding({ float: false, bits: 16, data: p });
  check('pile-up at 65504 gives the measured ceiling', e.encoding === 'u12_shifted' && e.measured && e.ceiling === 65504, `(${e.ceiling})`);
  const u = new Uint16Array(n);
  for (let i = 0; i < n; i++) u[i] = i % 4000;
  e = detectEncoding({ float: false, bits: 16, data: u });
  check('max ≤ 4095 not shifted is u12, ambiguous', e.encoding === 'u12' && e.ceiling === 4095 && e.ambiguous && e.toDn12 === 1);
  e = detectEncoding({ float: true, bits: 32, data: new Float32Array(4) });
  check('float has no ceiling', e.encoding === 'float' && Number.isNaN(e.ceiling));
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
