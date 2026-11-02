/**
 * Small statistics helpers with the same definitions as numpy, so the parity tests against the
 * Python references of manim/tools/make_web_fixtures.py can ask for 1e-9.
 *   median      numpy.median (mean of the two middle values for an even count)
 *   percentile  numpy.percentile, method 'linear'
 *   interp      numpy.interp (x ascending, constant or given value outside)
 *   gradient    numpy.gradient with coordinates, edge_order 1
 */

export function sortedCopy(a: ArrayLike<number>): Float64Array {
  return Float64Array.from(a).sort();
}

export function median(a: ArrayLike<number>): number {
  const n = a.length;
  if (!n) return NaN;
  const s = sortedCopy(a);
  const h = n >> 1;
  return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

/** numpy.percentile(a, q) with the default linear method, q in 0..100. */
export function percentile(a: ArrayLike<number>, q: number): number {
  const n = a.length;
  if (!n) return NaN;
  const s = sortedCopy(a);
  return percentileSorted(s, q);
}

export function percentileSorted(s: ArrayLike<number>, q: number): number {
  const n = s.length;
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const g = pos - lo;
  // numpy's _lerp: a + (b − a) g, written as b − (b − a)(1 − g) when g ≥ 0.5
  const a0 = s[lo], b0 = s[hi];
  const d = b0 - a0;
  return g >= 0.5 ? b0 - d * (1 - g) : a0 + d * g;
}

export function mean(a: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

/** Standard deviation, ddof as numpy. */
export function std(a: ArrayLike<number>, ddof = 0): number {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return Math.sqrt(s / (a.length - ddof));
}

/** numpy.interp for one value. xp ascending. */
export function interp1(x: number, xp: ArrayLike<number>, fp: ArrayLike<number>, left?: number, right?: number): number {
  const n = xp.length;
  if (Number.isNaN(x)) return NaN;
  if (x <= xp[0]) return x === xp[0] ? fp[0] : (left ?? fp[0]);
  if (x >= xp[n - 1]) return x === xp[n - 1] ? fp[n - 1] : (right ?? fp[n - 1]);
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  const slope = (fp[hi] - fp[lo]) / (xp[hi] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

export function interp(x: ArrayLike<number>, xp: ArrayLike<number>, fp: ArrayLike<number>, left?: number, right?: number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = interp1(x[i], xp, fp, left, right);
  return out;
}

/** numpy.gradient(y, x) with edge_order = 1 (second order inside, first order at the ends). */
export function gradient(y: ArrayLike<number>, x: ArrayLike<number>): Float64Array {
  const n = y.length;
  const g = new Float64Array(n);
  if (n < 2) return g;
  for (let i = 1; i < n - 1; i++) {
    const h1 = x[i] - x[i - 1], h2 = x[i + 1] - x[i];
    const a = -h2 / (h1 * (h1 + h2));
    const b = (h2 - h1) / (h1 * h2);
    const c = h1 / (h2 * (h1 + h2));
    g[i] = a * y[i - 1] + b * y[i] + c * y[i + 1];
  }
  g[0] = (y[1] - y[0]) / (x[1] - x[0]);
  g[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
  return g;
}

/** Trapezoid rule as numpy.trapezoid(y, x). */
export function trapz(y: ArrayLike<number>, x: ArrayLike<number>): number {
  let s = 0;
  for (let i = 1; i < y.length; i++) s += ((x[i] - x[i - 1]) * (y[i] + y[i - 1])) / 2;
  return s;
}

/** Ordinary least squares y = a + b x, with the standard errors of a and b (NaN for n < 3). */
export function linfit(x: ArrayLike<number>, y: ArrayLike<number>): { a: number; b: number; sa: number; sb: number; n: number } {
  const n = x.length;
  const xm = mean(x), ym = mean(y);
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (x[i] - xm) ** 2; sxy += (x[i] - xm) * (y[i] - ym); }
  const b = sxy / sxx;
  const a = ym - b * xm;
  let ssr = 0;
  for (let i = 0; i < n; i++) ssr += (y[i] - a - b * x[i]) ** 2;
  if (n < 3) return { a, b, sa: NaN, sb: NaN, n };
  const s2 = ssr / (n - 2);
  let sx2 = 0;
  for (let i = 0; i < n; i++) sx2 += x[i] * x[i];
  return { a, b, sa: Math.sqrt((s2 * sx2) / (n * sxx)), sb: Math.sqrt(s2 / sxx), n };
}

/** Round half to even, as Python's round(). */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Brent-free bisection on a bracket, to |b − a| ≤ tol. f(a) and f(b) of opposite sign. */
export function bisect(f: (x: number) => number, a: number, b: number, tol = 1e-12, maxIter = 200): number {
  let fa = f(a);
  const fb = f(b);
  if (fa === 0) return a;
  if (fb === 0) return b;
  if (fa * fb > 0) return NaN;
  for (let k = 0; k < maxIter && b - a > tol * Math.max(1, Math.abs(a)); k++) {
    const m = 0.5 * (a + b), fm = f(m);
    if (fm === 0) return m;
    if (fa * fm < 0) b = m; else { a = m; fa = fm; }
  }
  return 0.5 * (a + b);
}
