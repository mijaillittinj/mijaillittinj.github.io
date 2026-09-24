/**
 * Image deblurring for the computing tab.
 *
 * Forward: g = h * f + ε (circular convolution of the sharp image f with a known
 * point-spread function h, plus Gaussian noise). In the Fourier domain G = H F + E.
 * Inverse:
 *   - inverse filter      F̂ = G / H                     (divides noise by |H| ≈ 0: explodes)
 *   - Tikhonov / Wiener   F̂ = H̄ G / (|H|² + λ)          (λ trades resolution for stability)
 *   - Richardson–Lucy     f ← f · [h̃ * (g / (h * f))]    (iterative, keeps f ≥ 0)
 * Square power-of-two images, grey values in [0, 1]. Small radix-2 FFT, no dependencies.
 */
import { rng } from '../linalg.ts';

/* ------------------------------------------------------------------ FFT */

/** In-place iterative radix-2 FFT of length n (power of two). inverse = true scales by 1/n. */
export function fft1(re: Float64Array, im: Float64Array, inverse = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

export interface Spec { re: Float64Array; im: Float64Array; n: number }

/** 2D FFT of an n×n real (or complex) array, row-major. */
export function fft2(n: number, reIn: ArrayLike<number>, imIn?: ArrayLike<number>, inverse = false): Spec {
  const re = Float64Array.from(reIn), im = imIn ? Float64Array.from(imIn) : new Float64Array(n * n);
  const r = new Float64Array(n), c = new Float64Array(n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { r[x] = re[y * n + x]; c[x] = im[y * n + x]; }
    fft1(r, c, inverse);
    for (let x = 0; x < n; x++) { re[y * n + x] = r[x]; im[y * n + x] = c[x]; }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { r[y] = re[y * n + x]; c[y] = im[y * n + x]; }
    fft1(r, c, inverse);
    for (let y = 0; y < n; y++) { re[y * n + x] = r[y]; im[y * n + x] = c[y]; }
  }
  return { re, im, n };
}

/* ------------------------------------------------------------------ PSF */

/**
 * Linear motion blur of `length` pixels at angle `angleDeg`, sampled by supersampling a
 * line segment; stored wrapped around (0, 0) so that convolution does not shift the image.
 * length < 1 gives a small Gaussian defocus of width `defocus` instead (if > 0).
 */
export function motionPsf(n: number, length: number, angleDeg: number, defocus = 0): Float64Array {
  const h = new Float64Array(n * n);
  const put = (x: number, y: number, w: number) => {
    const xi = Math.round(x), yi = Math.round(y);
    h[(((yi % n) + n) % n) * n + (((xi % n) + n) % n)] += w;
  };
  const a = (angleDeg * Math.PI) / 180, L = Math.max(0, length);
  const steps = Math.max(1, Math.ceil(L * 8));
  for (let s = 0; s <= steps; s++) {
    const t = (s / steps - 0.5) * L;
    put(t * Math.cos(a), -t * Math.sin(a), 1);
  }
  if (defocus > 0) {
    const g = new Float64Array(n * n), R = Math.ceil(3 * defocus);
    for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++) {
      const w = Math.exp(-(x * x + y * y) / (2 * defocus * defocus));
      g[((y + n) % n) * n + ((x + n) % n)] = w;
    }
    return normalise(convolve(n, h, g));
  }
  return normalise(h);
}

function normalise(h: Float64Array): Float64Array {
  let s = 0;
  for (const v of h) s += v;
  for (let i = 0; i < h.length; i++) h[i] /= s || 1;
  return h;
}

/** Circular convolution of two n×n real arrays. */
export function convolve(n: number, a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  return convolveSpec(fft2(n, b), a);
}

/** Circular convolution with a pre-computed spectrum H. */
export function convolveSpec(H: Spec, a: ArrayLike<number>): Float64Array {
  const n = H.n, A = fft2(n, a);
  for (let i = 0; i < n * n; i++) {
    const r = A.re[i] * H.re[i] - A.im[i] * H.im[i];
    A.im[i] = A.re[i] * H.im[i] + A.im[i] * H.re[i];
    A.re[i] = r;
  }
  return fft2(n, A.re, A.im, true).re;
}

/** Correlation (convolution with the flipped kernel), i.e. multiplication by H̄. */
export function correlateSpec(H: Spec, a: ArrayLike<number>): Float64Array {
  const n = H.n, A = fft2(n, a);
  for (let i = 0; i < n * n; i++) {
    const r = A.re[i] * H.re[i] + A.im[i] * H.im[i];
    A.im[i] = A.im[i] * H.re[i] - A.re[i] * H.im[i];
    A.re[i] = r;
  }
  return fft2(n, A.re, A.im, true).re;
}

/* --------------------------------------------------------------- forward */

export function blur(n: number, f: ArrayLike<number>, h: ArrayLike<number>, sigma: number, seed: number): Float64Array {
  const g = convolve(n, f, h), R = rng(seed);
  for (let i = 0; i < g.length; i++) g[i] += sigma * R.normal();
  return g;
}

/* --------------------------------------------------------------- inverse */

/** Fourier filter: λ = 0 gives the plain inverse filter (with a 1e-12 floor to avoid 0/0). */
export function tikhonov(n: number, g: ArrayLike<number>, h: ArrayLike<number>, lambda: number): Float64Array {
  const G = fft2(n, g), H = fft2(n, h);
  for (let i = 0; i < n * n; i++) {
    const hr = H.re[i], hi = H.im[i], d = hr * hr + hi * hi + Math.max(lambda, 1e-12);
    const r = (hr * G.re[i] + hi * G.im[i]) / d;
    G.im[i] = (hr * G.im[i] - hi * G.re[i]) / d;
    G.re[i] = r;
  }
  return fft2(n, G.re, G.im, true).re;
}

/** One Richardson–Lucy step, in place: f ← f · correlate(h, g / (h * f)). */
export function richardsonLucyStep(H: Spec, g: ArrayLike<number>, f: Float64Array) {
  const hf = convolveSpec(H, f);
  const ratio = new Float64Array(f.length);
  for (let i = 0; i < f.length; i++) ratio[i] = Math.max(g[i], 0) / Math.max(hf[i], 1e-6);
  const c = correlateSpec(H, ratio);
  for (let i = 0; i < f.length; i++) f[i] = Math.max(0, f[i] * c[i]);
}

/** Peak signal-to-noise ratio (dB) for images in [0, 1]. */
export function psnr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.max(0, Math.min(1, a[i])) - b[i];
    s += d * d;
  }
  const mse = s / a.length;
  return mse > 0 ? 10 * Math.log10(1 / mse) : Infinity;
}

/** Deterministic synthetic test image for Node tests (bars, squares and a disc). */
export function testPattern(n: number): Float64Array {
  const f = new Float64Array(n * n).fill(1);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = x / n, v = y / n;
    if (v > 0.1 && v < 0.4 && u > 0.1 && u < 0.9 && Math.floor(u * 16) % 2 === 0) f[y * n + x] = 0;
    if (Math.hypot(u - 0.3, v - 0.7) < 0.15) f[y * n + x] = 0.1;
    if (u > 0.55 && u < 0.85 && v > 0.55 && v < 0.85 && (Math.floor(u * 20) + Math.floor(v * 20)) % 2 === 0) f[y * n + x] = 0;
  }
  return f;
}
