// ANNAbel stage 1, problem setup. Port of annabel_app/core/abel.py (op_matrix, split_halves),
// models.CoordNorm and stage1.Stage1Problem.__init__ (Escudero et al., Proc. Combust. Inst. 40
// (2024) 105493). Geometry of the app: ring j (1-based) covers [(j−1)dr, j dr] with constant
// emission, ray i sits at x_i = (i − ½)dr, the axis lies between two pixel columns, and both
// image halves enter the loss. Lengths in mm, as in the app.

/** Projection data, the contract of annabel_app/core/io.py (row-major, RGB last).
 *  b (Nt, Nz, 2 Nr, 3) line-of-sight projections, r (Nr) ray positions in mm with uniform
 *  spacing, z (Nz) heights in mm, times (Nt) acquisition times in s (≥ 0). */
export interface Projections {
  nt: number;
  nz: number;
  nr: number;
  b: Float32Array;
  rMm: ArrayLike<number>;
  zMm: ArrayLike<number>;
  times: ArrayLike<number>;
}

/** abel.op_matrix(nr, dr): (nr, nr) upper-triangular onion-peeling operator, b = A J,
 *  A[i*nr + j] = chord length of ray i inside ring j, in the length unit of dr. Float64, same
 *  operation order as the NumPy code. */
export function opMatrix(nr: number, dr: number): Float64Array {
  if (!(nr >= 1)) throw new Error('nr must be >= 1');
  if (!Number.isFinite(dr) || dr <= 0) throw new Error(`dr must be positive, got ${dr}`);
  const A = new Float64Array(nr * nr);
  for (let i = 1; i <= nr; i++) {
    const x2 = (i - 0.5) ** 2;
    for (let j = i; j <= nr; j++) {
      let a = Math.sqrt(j * j - x2);
      if (j > i) a -= Math.sqrt((j - 1) ** 2 - x2);
      A[(i - 1) * nr + (j - 1)] = 2 * (a * dr);
    }
  }
  return A;
}

/** abel.split_halves: full width (rows, 2 Nr, 3) → right half and mirrored left half, each
 *  (rows, Nr, 3), column 0 of both is the ray just outside the axis. */
export function splitHalves(bFull: ArrayLike<number>, rows: number, nr: number, out?: { right: Float32Array; left: Float32Array }) {
  const right = out?.right ?? new Float32Array(rows * nr * 3);
  const left = out?.left ?? new Float32Array(rows * nr * 3);
  for (let k = 0; k < rows; k++)
    for (let i = 0; i < nr; i++)
      for (let c = 0; c < 3; c++) {
        right[(k * nr + i) * 3 + c] = bFull[(k * 2 * nr + nr + i) * 3 + c];
        left[(k * nr + i) * 3 + c] = bFull[(k * 2 * nr + nr - 1 - i) * 3 + c];
      }
  return { right, left };
}

/** models.CoordNorm: r̃ = r / r_max, z̃ = (z − z_min)/(z_max − z_min), t̃ = t / t_max, with
 *  t_max guarded to 1 for single-frame data. The constants are doubles taken from float32
 *  grids, like the app. */
export interface CoordNorm { r_max: number; z_min: number; z_max: number; t_max: number }

const f32max = (a: ArrayLike<number>) => { let m = -Infinity; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.fround(a[i])); return m; };
const f32min = (a: ArrayLike<number>) => { let m = Infinity; for (let i = 0; i < a.length; i++) m = Math.min(m, Math.fround(a[i])); return m; };

export function coordNormFromGrid(rMm: ArrayLike<number>, zMm: ArrayLike<number>, times: ArrayLike<number>): CoordNorm {
  const tMax = f32max(times);
  for (let i = 0; i < times.length; i++) if (!(times[i] >= 0)) throw new Error('times must be >= 0');
  return { r_max: f32max(rMm), z_min: f32min(zMm), z_max: f32max(zMm), t_max: tMax > 0 ? tMax : 1 };
}

/** CoordNorm.features: tensor-product grid (t, z, r) flattened to (N, 3) float32, computed in
 *  float32 arithmetic with double constants, as NumPy does, so the inputs are bit-identical. */
export function coordFeatures(norm: CoordNorm, rMm: ArrayLike<number>, zMm: ArrayLike<number>, times: ArrayLike<number>): Float32Array {
  const nt = times.length, nz = zMm.length, nr = rMm.length;
  const f = Math.fround;
  const dz = (norm.z_max - norm.z_min) || 1;
  const X = new Float32Array(nt * nz * nr * 3);
  let n = 0;
  for (let t = 0; t < nt; t++) {
    const tt = f(f(times[t]) / norm.t_max);
    for (let k = 0; k < nz; k++) {
      const zz = f(f(f(zMm[k]) - norm.z_min) / dz);
      for (let i = 0; i < nr; i++) {
        const rt = f(f(rMm[i]) / norm.r_max);
        X[n++] = rt * rt; X[n++] = zz; X[n++] = tt;
      }
    }
  }
  return X;
}

/** Stage1Problem, everything the loss needs, in float32 like the app. rows = Nt·Nz rows of Nr
 *  rays, sample n = row·Nr + i in (t, z, r) order. */
export interface Stage1Problem {
  nt: number; nz: number; nr: number;
  rows: number;          // Nt · Nz
  N: number;             // rows · Nr samples
  M: number;             // N · 3 terms in each mean of the loss
  dr: number;            // float32 spacing of the r grid, mm
  A: Float32Array;       // op_matrix(nr, dr) in float32, A[i*nr + j]
  X: Float32Array;       // (N, 3) inputs r̃², z̃, t̃
  bR: Float32Array;      // (N, 3) right half / b_scale
  bL: Float32Array;      // (N, 3) mirrored left half / b_scale
  nrm: Float32Array;     // (3) per-channel normalisation, max|b_c| / b_scale (1 for an all-zero channel)
  invNrm2: Float64Array; // (3) 1 / nrm²
  bScale: number;        // max |b|
  outScale: number;      // 1 / max row sum of A
  norm: CoordNorm;
}

export function buildProblem(p: Projections): Stage1Problem {
  const { nt, nz, nr, b } = p;
  if (nr < 2) throw new Error('need at least two radii');
  if (b.length !== nt * nz * 2 * nr * 3) throw new Error(`b has ${b.length} values, expected ${nt * nz * 2 * nr * 3}`);
  if (p.rMm.length !== nr || p.zMm.length !== nz || p.times.length !== nt) throw new Error('grid lengths differ from b');
  const rows = nt * nz, N = rows * nr;
  const dr = Math.fround(Math.fround(p.rMm[1]) - Math.fround(p.rMm[0]));
  const A = Float32Array.from(opMatrix(nr, dr));
  let bScale = 0;
  const cmax = [0, 0, 0];
  for (let q = 0; q < b.length; q++) {
    const a = Math.abs(b[q]);
    if (a > bScale) bScale = a;
    if (a > cmax[q % 3]) cmax[q % 3] = a;
  }
  if (!(bScale > 0) || !Number.isFinite(bScale)) throw new Error('projections are all zero or not finite');
  const nrm = new Float32Array(3), invNrm2 = new Float64Array(3);
  for (let c = 0; c < 3; c++) {
    const v = Math.fround(cmax[c] / bScale);
    nrm[c] = v > 0 ? v : 1;
    invNrm2[c] = 1 / (nrm[c] * nrm[c]);
  }
  const { right, left } = splitHalves(b, rows, nr);
  for (let q = 0; q < right.length; q++) { right[q] = right[q] / bScale; left[q] = left[q] / bScale; }
  let rowMax = 0;
  for (let i = 0; i < nr; i++) {
    let s = 0;
    for (let j = 0; j < nr; j++) s += A[i * nr + j];
    if (s > rowMax) rowMax = s;
  }
  const norm = coordNormFromGrid(p.rMm, p.zMm, p.times);
  const X = coordFeatures(norm, p.rMm, p.zMm, p.times);
  return { nt, nz, nr, rows, N, M: N * 3, dr, A, X, bR: right, bL: left, nrm, invNrm2, bScale, outScale: 1 / Math.fround(rowMax), norm };
}
