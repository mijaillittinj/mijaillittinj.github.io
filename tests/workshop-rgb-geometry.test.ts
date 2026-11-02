// Flame axis of the workshop tool (geometry.ts) against find_axis of annabel_app/core/preprocess.py on
// off-centre synthetic rows, the refined sub-pixel axis and the halves sampled there:
// run with `node --experimental-strip-types tests/workshop-rgb-geometry.test.ts`
// Tolerances written into axis_ref.json by manim/tools/make_web_fixtures.py before running (GB2r1-numerico-06).
import { readFileSync } from 'node:fs';
import { findAxis, halvesAt } from '../src/lib/workshop-rgb/geometry.ts';
import { splitHalves } from '../src/lib/workshop-rgb/invert.ts';
import { chordMatrixHalf, onionPeel } from '../src/lib/abel.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const ref = JSON.parse(readFileSync(new URL('./fixtures/workshop-rgb/axis_ref.json', import.meta.url), 'utf8'));

/** Same image as axis_synth in make_web_fixtures.py. */
function synth(H: number, W: number, axis: number, dr: number): Float64Array {
  const img = new Float64Array(H * W * 3);
  const g = [1.0, 0.6, 0.25];
  for (let y = 0; y < H; y++) for (let k = 0; k < W; k++) {
    const yy = (k + 0.5 - axis) * dr;
    const P = Math.sqrt(Math.PI) * Math.exp(-yy * yy) * (1 + (0.1 * y) / H);
    for (let c = 0; c < 3; c++) img[3 * (y * W + k) + c] = P * g[c];
  }
  return img;
}

for (const cs of ref.cases) {
  const img = synth(ref.H, ref.W, cs.axis_true, ref.dr);
  const r = findAxis(img, ref.W, ref.H, { halfWidth: ref.half_width, search: ref.search, rowFrac: ref.row_frac, minSnr: ref.min_snr });
  let m = 0;
  r.curve.forEach((v, i) => { m = Math.max(m, Math.abs(v - cs.curve[i]) / Math.max(...cs.curve)); });
  check(`δ = ${cs.delta} px: integer axis, asymmetry curve, rows and flags equal find_axis`,
    r.axis === cs.axis && r.positions.join(',') === cs.positions.join(',') && m < 1e-12 && r.rows![0] === cs.rows[0] && r.rows![1] === cs.rows[1] && r.codes.length === cs.n_flags,
    `(axis ${r.axis}, ${r.codes.join(' ')}, curve ${m.toExponential(1)})`);
  check(`δ = ${cs.delta} px: ANNAbel parabola sub-pixel position equals subpixel_pos_px`, cs.subpixel === null ? r.subpixel === null : Math.abs(r.subpixel! - cs.subpixel) < 1e-9,
    `(${r.subpixel?.toFixed(3)}, error ${(r.subpixel! - cs.axis_true).toFixed(3)} px)`);
  const err = Math.abs(r.refined! - cs.axis_true);
  check(`δ = ${cs.delta} px: refined axis within ${ref.tol_refined_px} px`, err <= ref.tol_refined_px, `(${r.refined!.toFixed(4)}, error ${err.toExponential(1)} px)`);
  // halves on the axis ray, the middle row of the green channel
  const row = Float64Array.from({ length: ref.W }, (_, k) => img[3 * ((ref.H >> 1) * ref.W + k) + 1]);
  const h = halvesAt(row, r.refined!, ref.half_width);
  const d = Math.abs(h.right[0] - h.left[0]) / h.right[0];
  const h0 = splitHalves(row.subarray(cs.axis - ref.half_width, cs.axis + ref.half_width));
  const d0 = Math.abs(h0.right[0] - h0.left[0]) / h0.right[0];
  check(`δ = ${cs.delta} px: halves at the refined axis differ less than ${100 * ref.tol_halves} % on the axis ray`, d <= ref.tol_halves, `(${(100 * d).toFixed(3)} %, ${(100 * d0).toFixed(2)} % split at the integer axis)`);
  // what Abel sees: onion peeling of each half, emission on the axis
  const L = chordMatrixHalf(ref.half_width, ref.dr);
  const on = (q: Float64Array) => onionPeel(L, q)[0];
  const e = Math.abs(on(h.right) - on(h.left)) / Math.abs(on(h.right)), e0 = Math.abs(on(h0.right) - on(h0.left)) / Math.abs(on(h0.right));
  check(`δ = ${cs.delta} px: onion of the two halves on the axis differs less than ${100 * ref.tol_halves} % at the refined axis`, e <= ref.tol_halves, `(${(100 * e).toFixed(3)} %, ${(100 * e0).toFixed(1)} % at the integer axis)`);
}
const row = [4, 3, 2, 1, 1, 2, 3, 5];
const h = halvesAt(row, 4, 4);
check('halvesAt at an integer axis equals splitHalves', h.right.join(',') === '1,2,3,5' && h.left.join(',') === '1,2,3,4');

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
