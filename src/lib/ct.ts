/**
 * Computed tomography in the browser: phantoms, the Radon transform (parallel beam),
 * unfiltered and filtered back-projection, and SIRT (an iterative method).
 *
 * Geometry: an n × n image with unit pixels, centred at the origin. A parallel-beam
 * projection at angle θ is sampled at n detector positions s (unit spacing); each ray
 * integral is computed by marching along the ray with unit steps and bilinear sampling.
 * The back-projector `radonT` scatters with the same bilinear weights, so it is the exact
 * adjoint of `radon` (needed for SIRT to converge to a least-squares solution).
 */

export type PhantomKind = 'head' | 'pipe';

interface Ellipse { v: number; a: number; b: number; x: number; y: number; phi: number }

/** Modified Shepp–Logan head phantom (Toft's contrast), in coordinates [-1, 1]². */
const SHEPP: Ellipse[] = [
  { v: 1.0, a: 0.69, b: 0.92, x: 0, y: 0, phi: 0 },
  { v: -0.8, a: 0.6624, b: 0.874, x: 0, y: -0.0184, phi: 0 },
  { v: -0.2, a: 0.11, b: 0.31, x: 0.22, y: 0, phi: -18 },
  { v: -0.2, a: 0.16, b: 0.41, x: -0.22, y: 0, phi: 18 },
  { v: 0.1, a: 0.21, b: 0.25, x: 0, y: 0.35, phi: 0 },
  { v: 0.1, a: 0.046, b: 0.046, x: 0, y: 0.1, phi: 0 },
  { v: 0.1, a: 0.046, b: 0.046, x: 0, y: -0.1, phi: 0 },
  { v: 0.1, a: 0.046, b: 0.023, x: -0.08, y: -0.605, phi: 0 },
  { v: 0.1, a: 0.023, b: 0.023, x: 0, y: -0.606, phi: 0 },
  { v: 0.1, a: 0.023, b: 0.046, x: 0.06, y: -0.605, phi: 0 },
];

/**
 * Phantom image, row-major, row 0 at the bottom (y increasing upwards), values in [0, 1].
 * 'head': Shepp–Logan. 'pipe': a steel pipe wall with a crack and two inclusions,
 * as in industrial non-destructive testing.
 */
export function phantom(n: number, kind: PhantomKind = 'head'): Float64Array {
  const f = new Float64Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = ((i + 0.5) / n) * 2 - 1, y = ((j + 0.5) / n) * 2 - 1;
    let v = 0;
    if (kind === 'head') {
      for (const e of SHEPP) {
        const c = Math.cos((e.phi * Math.PI) / 180), s = Math.sin((e.phi * Math.PI) / 180);
        const u = (x - e.x) * c + (y - e.y) * s, w = -(x - e.x) * s + (y - e.y) * c;
        if ((u / e.a) ** 2 + (w / e.b) ** 2 <= 1) v += e.v;
      }
    } else {
      const r = Math.hypot(x, y), th = Math.atan2(y, x);
      if (r <= 0.85 && r >= 0.62) v = 1;                                 // pipe wall
      if (r < 0.62) v = 0.25;                                             // fluid inside
      if (r <= 0.85 && r >= 0.66 && Math.abs(th - 0.6) < 0.02) v = 0.25;  // radial crack
      if (Math.hypot(x + 0.3, y + 0.2) < 0.08) v = 0.7;                   // inclusion
      if (Math.hypot(x - 0.15, y - 0.3) < 0.05) v = 0.55;                 // small inclusion
    }
    f[j * n + i] = Math.max(0, Math.min(1, v));
  }
  return f;
}

/** Projection angles (radians) equally spaced over [0, range). */
export function angles(count: number, rangeDeg = 180): Float64Array {
  return Float64Array.from({ length: count }, (_, k) => ((k * rangeDeg) / count) * (Math.PI / 180));
}

/** Visit the bilinear sampling points of ray (θ, s): calls fn(pixelIndex, weight). */
function march(n: number, theta: number, s: number, fn: (p: number, w: number) => void) {
  const c = Math.cos(theta), sn = Math.sin(theta);
  const half = n / 2, L = Math.ceil(n * 0.75);
  for (let t = -L + 0.5; t < L; t += 1) {
    // point on the ray, in pixel coordinates (pixel centres at integer + 0.5 from the corner)
    const x = s * c - t * sn + half - 0.5, y = s * sn + t * c + half - 0.5;
    const i = Math.floor(x), j = Math.floor(y);
    if (i < -1 || j < -1 || i >= n || j >= n) continue;
    const dx = x - i, dy = y - j;
    if (i >= 0 && j >= 0) fn(j * n + i, (1 - dx) * (1 - dy));
    if (i + 1 < n && j >= 0) fn(j * n + i + 1, dx * (1 - dy));
    if (i >= 0 && j + 1 < n) fn((j + 1) * n + i, (1 - dx) * dy);
    if (i + 1 < n && j + 1 < n) fn((j + 1) * n + i + 1, dx * dy);
  }
}

const detector = (n: number, d: number) => d - n / 2 + 0.5;

/** Radon transform: sinogram [angle][detector], n detectors of unit width. */
export function radon(f: ArrayLike<number>, n: number, th: ArrayLike<number>): Float64Array {
  const out = new Float64Array(th.length * n);
  for (let a = 0; a < th.length; a++) for (let d = 0; d < n; d++) {
    let acc = 0;
    march(n, th[a], detector(n, d), (p, w) => { acc += w * f[p]; });
    out[a * n + d] = acc;
  }
  return out;
}

/** Exact adjoint of `radon` (unfiltered back-projection without normalisation). */
export function radonT(sino: ArrayLike<number>, n: number, th: ArrayLike<number>): Float64Array {
  const out = new Float64Array(n * n);
  for (let a = 0; a < th.length; a++) for (let d = 0; d < n; d++) {
    const v = sino[a * n + d];
    if (v !== 0) march(n, th[a], detector(n, d), (p, w) => { out[p] += w * v; });
  }
  return out;
}

export type FilterKind = 'none' | 'ramp' | 'shepp-logan';

/**
 * Spatial filter kernel h[k] (unit detector spacing) whose Fourier transform is the ramp |ω|,
 * optionally apodised: Ram–Lak for 'ramp', Shepp–Logan (ramp × sinc) for 'shepp-logan'.
 */
export function filterKernel(kind: FilterKind, half: number): Float64Array {
  const h = new Float64Array(2 * half + 1);
  for (let k = -half; k <= half; k++) {
    let v: number;
    if (kind === 'ramp') v = k === 0 ? 0.25 : k % 2 === 0 ? 0 : -1 / (Math.PI * Math.PI * k * k);
    else if (kind === 'shepp-logan') v = -2 / (Math.PI * Math.PI * (4 * k * k - 1));
    else v = k === 0 ? 1 : 0;
    h[k + half] = v;
  }
  return h;
}

/** Filter each projection with the kernel (discrete convolution, zero padding). */
export function filterSinogram(sino: ArrayLike<number>, n: number, nAng: number, kind: FilterKind): Float64Array {
  if (kind === 'none') return Float64Array.from(sino);
  const h = filterKernel(kind, n), out = new Float64Array(nAng * n);
  for (let a = 0; a < nAng; a++) for (let d = 0; d < n; d++) {
    let acc = 0;
    for (let e = 0; e < n; e++) acc += h[d - e + n] * sino[a * n + e];
    out[a * n + d] = acc;
  }
  return out;
}

/** Pixel-driven back-projection with linear interpolation on the detector. */
export function backproject(sino: ArrayLike<number>, n: number, th: ArrayLike<number>): Float64Array {
  const out = new Float64Array(n * n);
  const cs = Array.from(th, (t) => [Math.cos(t), Math.sin(t)]);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = i - n / 2 + 0.5, y = j - n / 2 + 0.5;
    let acc = 0;
    for (let a = 0; a < th.length; a++) {
      const s = x * cs[a][0] + y * cs[a][1];
      const u = s + n / 2 - 0.5, d = Math.floor(u), w = u - d;
      if (d >= 0 && d < n) acc += (1 - w) * sino[a * n + d];
      if (d + 1 >= 0 && d + 1 < n) acc += w * sino[a * n + d + 1];
    }
    out[j * n + i] = acc;
  }
  return out;
}

/**
 * Filtered back-projection. With the Ram–Lak kernel and angular step Δθ = range/N,
 * f ≈ Δθ Σ_θ (p_θ * h)(x cos θ + y sin θ) reproduces the image values for a full 180° scan.
 * Unfiltered back-projection ('none') is returned rescaled to the same mean for display.
 */
export function fbp(sino: ArrayLike<number>, n: number, th: ArrayLike<number>, kind: FilterKind, rangeDeg = 180): Float64Array {
  const filt = filterSinogram(sino, n, th.length, kind);
  const bp = backproject(filt, n, th);
  const dth = ((rangeDeg / th.length) * Math.PI) / 180;
  for (let p = 0; p < bp.length; p++) bp[p] *= dth;
  if (kind === 'none') {
    // plain back-projection has units of length: normalise its mean inside the circle to the sinogram mass
    let mass = 0, sum = 0;
    for (let d = 0; d < n; d++) mass += sino[d];
    for (let p = 0; p < bp.length; p++) sum += bp[p];
    const scale = sum > 0 ? mass / sum : 1;
    for (let p = 0; p < bp.length; p++) bp[p] *= scale;
  }
  return bp;
}

/**
 * SIRT: x ← x + C Aᵀ R (b − A x), with R = 1/row sums, C = 1/column sums; optional x ≥ 0.
 * Returns a stepper that performs one iteration per call and reports the residual norm.
 */
export function makeSirt(sino: ArrayLike<number>, n: number, th: ArrayLike<number>, nonneg: boolean) {
  const ones = new Float64Array(n * n).fill(1);
  const rowSum = radon(ones, n, th);
  const colSum = radonT(new Float64Array(th.length * n).fill(1), n, th);
  const x = new Float64Array(n * n);
  let iter = 0;
  return {
    x,
    get iter() { return iter; },
    step(): number {
      const Ax = radon(x, n, th);
      const r = new Float64Array(Ax.length);
      let res = 0;
      for (let q = 0; q < r.length; q++) { const e = sino[q] - Ax[q]; res += e * e; r[q] = rowSum[q] > 1e-9 ? e / rowSum[q] : 0; }
      const upd = radonT(r, n, th);
      for (let p = 0; p < x.length; p++) {
        if (colSum[p] > 1e-9) x[p] += upd[p] / colSum[p];
        if (nonneg && x[p] < 0) x[p] = 0;
      }
      iter++;
      return Math.sqrt(res);
    },
  };
}

/** RMSE and PSNR (peak 1) of a reconstruction against the phantom, inside the scanned circle. */
export function quality(rec: ArrayLike<number>, truth: ArrayLike<number>, n: number) {
  let e = 0, m = 0;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = i - n / 2 + 0.5, y = j - n / 2 + 0.5;
    if (x * x + y * y > (n / 2) ** 2) continue;
    const p = j * n + i;
    e += (rec[p] - truth[p]) ** 2; m++;
  }
  const rmse = Math.sqrt(e / m);
  return { rmse, psnr: 20 * Math.log10(1 / Math.max(rmse, 1e-12)) };
}
