/**
 * Integrating sphere calibration, ported from
 * fuentes/03_esfera_integradora/radiometric_calibration.py (find_sphere_disc, sphere_mask,
 * _fill_outside, calibration_constants, nd_transmission, effective_nd_transmission,
 * load_sphere_spectrum with its delimiter and header sniffing). λ in nm, spectral radiance per nm,
 * band integrals in W m⁻² sr⁻¹, K_c in DN s⁻¹ per (W m⁻² sr⁻¹).
 *
 * Images are (H, W, 3) chunky Float64. A dark may be an image of the same size, or a scalar.
 */
import { median, percentile, trapz, interp } from './stats.ts';
import { CH, type Rgb } from './radiometry.ts';

export interface Img3 { width: number; height: number; data: ArrayLike<number> }

// ------------------------------------------------------------------------------------- spectrum
export type SpectrumUnit = 'radiance_nm' | 'radiance_um' | 'irradiance_nm' | 'irradiance_um' | 'uW_cm2_nm' | 'counts';
export type LamUnit = 'nm' | 'um' | 'A';

const isNum = (s: string | undefined) => s !== undefined && s.trim() !== '' && Number.isFinite(Number(s.trim()));

/** First line with two numbers decides the separator (',' ';' tab, or null for whitespace), as _sniff_delimiter. */
export function sniffDelimiter(text: string): string | null {
  for (const line of text.split(/\r?\n/)) {
    for (const d of [',', ';', '\t', null]) {
      const t = d ? line.split(d) : line.trim().split(/\s+/);
      if (t.length < 2) continue;
      if (isNum(t[0]) && isNum(t[1])) return d;
    }
  }
  return null;
}

/** Leading lines that do not parse as two numbers, as _sniff_header. */
export function sniffHeader(text: string): number {
  let n = 0;
  for (const line of text.split(/\r?\n/)) {
    const t = line.replace(/[,;\t]/g, ' ').trim().split(/\s+/).filter((s) => s !== '');
    if (t.length >= 2 && isNum(t[0]) && isNum(t[1])) return n;
    n++;
  }
  return n;
}

/**
 * Read the spectrometer export of the sphere, return λ (nm, ascending) and L in W m⁻² sr⁻¹ nm⁻¹.
 * Unit conversions as load_sphere_spectrum. A per-µm column read as per-nm inflates f_s by 1000.
 */
export function loadSphereSpectrum(text: string, opts: { lamCol?: number; valCol?: number; delimiter?: string | null; skipHeader?: number; lamUnit?: LamUnit; valUnit?: SpectrumUnit; scale?: number } = {}): { lamNm: Float64Array; L: Float64Array; delimiter: string | null; skipHeader: number } {
  const lc = opts.lamCol ?? 0, vc = opts.valCol ?? 1;
  const delim = opts.delimiter !== undefined ? opts.delimiter : sniffDelimiter(text);
  const skip = opts.skipHeader ?? sniffHeader(text);
  const lines = text.split(/\r?\n/).slice(skip);
  const lam: number[] = [], val: number[] = [];
  let ncol = -1;
  for (const line of lines) {
    if (line.trim() === '') continue;
    const t = delim ? line.split(delim) : line.trim().split(/\s+/);
    if (ncol < 0) ncol = t.length;
    if (t.length !== ncol) continue; // genfromtxt(invalid_raise=False) skips rows with another column count
    const a = t[lc] !== undefined && t[lc].trim() !== '' ? Number(t[lc].trim()) : NaN;
    const b = t[vc] !== undefined && t[vc].trim() !== '' ? Number(t[vc].trim()) : NaN;
    if (Number.isFinite(a) && Number.isFinite(b)) { lam.push(a); val.push(b); }
  }
  const lf = { nm: 1, um: 1e3, A: 0.1 }[opts.lamUnit ?? 'nm'];
  const u = opts.valUnit ?? 'radiance_nm';
  const conv = (v: number) => {
    switch (u) {
      case 'radiance_nm': return v;
      case 'radiance_um': return v / 1e3;
      case 'irradiance_um': return v / 1e3 / Math.PI;
      case 'irradiance_nm': return v / Math.PI;
      case 'uW_cm2_nm': return (v * 1e-6) / 1e-4 / Math.PI;
      case 'counts': return v;
    }
  };
  const ord = lam.map((_, k) => k).sort((a, b) => lam[a] - lam[b]);
  const sc = opts.scale ?? 1;
  return { lamNm: Float64Array.from(ord, (k) => lf * lam[k]), L: Float64Array.from(ord, (k) => sc * conv(val[k])), delimiter: delim, skipHeader: skip };
}

/** Interpolate onto lamTo, `fill` outside the source range (resample). */
export function resample(lamTo: ArrayLike<number>, lamFrom: ArrayLike<number>, y: ArrayLike<number>, fill = 0): Float64Array {
  return interp(lamTo, lamFrom, y, fill, fill);
}

// ------------------------------------------------------------------------------------- ND filter
/**
 * ND transmission on the grid. A flat 10^−OD, a measured curve (held at its edge values outside), or
 * 1 with neither. The tool makes this field mandatory (WEB_SPEC 1b): 'none' must be chosen explicitly.
 */
export function ndTransmission(lamNm: ArrayLike<number>, nd: { kind: 'none' } | { kind: 'od'; od: number } | { kind: 'curve'; lamNm: ArrayLike<number>; tau: ArrayLike<number> }): Float64Array {
  if (nd.kind === 'curve') return interp(lamNm, nd.lamNm, nd.tau, nd.tau[0], nd.tau[nd.tau.length - 1]);
  if (nd.kind === 'od') return Float64Array.from({ length: lamNm.length }, () => 10 ** -nd.od);
  return Float64Array.from({ length: lamNm.length }, () => 1);
}

/** τ̄_c = ∫ ψ_c τ_nd L dλ / ∫ ψ_c L dλ, weighted by the sphere spectrum (effective_nd_transmission). */
export function effectiveNdTransmission(lamNm: ArrayLike<number>, psi: Rgb<ArrayLike<number>>, L: ArrayLike<number>, tauNd: ArrayLike<number>): Rgb<number> {
  const out = { R: 0, G: 0, B: 0 };
  const a = new Float64Array(lamNm.length), b = new Float64Array(lamNm.length);
  for (const c of CH) {
    for (let k = 0; k < a.length; k++) { a[k] = psi[c][k] * tauNd[k] * L[k]; b[k] = psi[c][k] * L[k]; }
    out[c] = trapz(a, lamNm) / trapz(b, lamNm);
  }
  return out;
}

// ------------------------------------------------------------------------------------- disc and mask
export interface Disc { cx: number; cy: number; rFit: number; rUse: number; mask: Uint8Array; frac: number }

/** Moment fit of the exit port (find_sphere_disc), threshold on the mean of the three channels. */
export function findSphereDisc(img: Img3, threshFrac = 0.5, shrink = 0.85): Disc {
  const { width: w, height: h, data } = img;
  const n = w * h;
  const lum = new Float64Array(n);
  for (let p = 0; p < n; p++) lum[p] = (data[3 * p] + data[3 * p + 1] + data[3 * p + 2]) / 3;
  const hot = percentile(lum, 99.9);
  let m = new Uint8Array(n);
  let cnt = 0;
  for (let p = 0; p < n; p++) if (lum[p] > threshFrac * hot) { m[p] = 1; cnt++; }
  if (cnt < 100) throw new Error('sphere_no_disc');
  let cy = 0, cx = 0, r = 0;
  for (let it = 0; it < 2; it++) {
    let sy = 0, sx = 0, k = 0;
    for (let p = 0; p < n; p++) if (m[p]) { sy += Math.floor(p / w); sx += p % w; k++; }
    cy = sy / k; cx = sx / k;
    r = Math.sqrt(k / Math.PI);
    const m2 = new Uint8Array(n);
    for (let p = 0; p < n; p++) if (m[p] && Math.hypot(Math.floor(p / w) - cy, (p % w) - cx) < 1.3 * r) m2[p] = 1;
    m = m2;
  }
  const rUse = shrink * r;
  const mask = new Uint8Array(n);
  let inside = 0;
  for (let p = 0; p < n; p++) if (Math.hypot(Math.floor(p / w) - cy, (p % w) - cx) <= rUse) { mask[p] = 1; inside++; }
  return { cx, cy, rFit: r, rUse, mask, frac: inside / n };
}

/**
 * Pixels usable for the calibration (sphere_mask). Inside the port, unsaturated in all three channels
 * and above minDn in all three after the dark. `sat` per channel (H·W·3); without it, ≥ 0,995 × max.
 */
export function sphereMask(img: Img3, dark: ArrayLike<number> | number | null, disc: Disc, sat?: Uint8Array, minDn = 200): { mask: Uint8Array; nDisc: number; nKept: number; lostSaturated: Rgb<number>; lostDark: Rgb<number>; medianDn: Rgb<number> } {
  const n = img.width * img.height;
  let mx = -Infinity;
  if (!sat) for (let i = 0; i < 3 * n; i++) if (img.data[i] > mx) mx = img.data[i];
  const isSat = (i: number) => (sat ? sat[i] === 1 : img.data[i] >= 0.995 * mx);
  const S = (i: number) => img.data[i] - (dark === null ? 0 : typeof dark === 'number' ? dark : dark[i]);
  const mask = new Uint8Array(n);
  const lostSaturated = { R: 0, G: 0, B: 0 }, lostDark = { R: 0, G: 0, B: 0 };
  let nDisc = 0, nKept = 0;
  for (let p = 0; p < n; p++) {
    if (!disc.mask[p]) continue;
    nDisc++;
    let ok = true;
    for (let c = 0; c < 3; c++) {
      if (isSat(3 * p + c)) { ok = false; lostSaturated[CH[c]]++; }
      if (!(S(3 * p + c) > minDn)) { ok = false; lostDark[CH[c]]++; }
    }
    if (ok) { mask[p] = 1; nKept++; }
  }
  const medianDn = { R: NaN, G: NaN, B: NaN };
  for (let c = 0; c < 3; c++) {
    const v: number[] = [];
    for (let p = 0; p < n; p++) if (mask[p]) v.push(S(3 * p + c));
    if (v.length) medianDn[CH[c]] = median(v);
  }
  return { mask, nDisc, nKept, lostSaturated, lostDark, medianDn };
}

/** Solve the 3 × 3 normal equations of the radial fit K = a + b ρ² + c ρ⁴. */
function solve3(A: number[][], b: number[]): number[] {
  const M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < 3; i++) {
    let p = i;
    for (let k = i + 1; k < 3; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
    [M[i], M[p]] = [M[p], M[i]];
    for (let k = i + 1; k < 3; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j < 4; j++) M[k][j] -= f * M[i][j]; }
  }
  const x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) { let s = M[i][3]; for (let j = i + 1; j < 3; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
  return x;
}

/**
 * Extend K from the disc to the frame (_fill_outside). 'median' one scalar per channel, 'radial' the
 * fit a + bρ² + cρ⁴ with ρ normalised by the half sizes and centred on the disc (a model outside it),
 * 'nan' leaves NaN.
 */
export function fillOutside(K2d: Float64Array, width: number, height: number, mask: Uint8Array, how: 'median' | 'radial' | 'nan', disc?: { cx: number; cy: number } | null): Float64Array {
  const out = Float64Array.from(K2d);
  const n = width * height;
  const inside: number[] = [];
  for (let p = 0; p < n; p++) { if (mask[p]) inside.push(out[p]); else out[p] = NaN; }
  if (how === 'nan') return out;
  if (how === 'median') {
    const m = median(inside);
    for (let p = 0; p < n; p++) if (!mask[p]) out[p] = m;
    return out;
  }
  const cy = disc ? disc.cy : height / 2, cx = disc ? disc.cx : width / 2;
  const rho2 = (p: number) => {
    const y = Math.floor(p / width), x = p % width;
    const r = Math.hypot((y - cy) / (height / 2), (x - cx) / (width / 2));
    return r * r;
  };
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  for (let p = 0; p < n; p++) {
    if (!mask[p]) continue;
    const r2 = rho2(p), v = [1, r2, r2 * r2];
    for (let i = 0; i < 3; i++) { b[i] += v[i] * out[p]; for (let j = 0; j < 3; j++) A[i][j] += v[i] * v[j]; }
  }
  const co = solve3(A, b);
  for (let p = 0; p < n; p++) if (!mask[p]) { const r2 = rho2(p); out[p] = co[0] + co[1] * r2 + co[2] * r2 * r2; }
  return out;
}

export interface SphereCalibration {
  K: Rgb<Float64Array>;
  bandIntegral: Rgb<number>;
  denominator: Rgb<number>;
  tauBar: Rgb<number>;
  KDiscMedian: Rgb<number>;
  KDiscRsd: Rgb<number>;
  maskFrac: number;
  fill: 'median' | 'radial' | 'nan';
}

/**
 * Per pixel and per channel K_c = (S_sph − D) / (t_sph · τ̄_c · ∫ ψ_c L_sph dλ), calibration_constants.
 * L_sph on the ψ grid (use `resample`). tSphS in seconds.
 */
export function calibrationConstants(img: Img3, dark: ArrayLike<number> | number | null, tSphS: number, lamNm: ArrayLike<number>, psi: Rgb<ArrayLike<number>>, Lsph: ArrayLike<number>, opts: { tauBar?: Rgb<number> | null; mask?: Uint8Array | null; fill?: 'median' | 'radial' | 'nan'; disc?: { cx: number; cy: number } | null } = {}): SphereCalibration {
  const { width: w, height: h } = img;
  const n = w * h;
  const tauBar = opts.tauBar ?? { R: 1, G: 1, B: 1 };
  const fill = opts.fill ?? 'median';
  const K = {} as Rgb<Float64Array>;
  const bi = { R: 0, G: 0, B: 0 }, den = { R: 0, G: 0, B: 0 }, med = { R: NaN, G: NaN, B: NaN }, rsd = { R: NaN, G: NaN, B: NaN };
  for (let c = 0; c < 3; c++) {
    const ch = CH[c];
    const y = new Float64Array(lamNm.length);
    for (let k = 0; k < y.length; k++) y[k] = psi[ch][k] * Lsph[k];
    bi[ch] = trapz(y, lamNm);
    den[ch] = tSphS * tauBar[ch] * bi[ch];
    if (!Number.isFinite(den[ch]) || den[ch] <= 0) throw new Error(`sphere_denominator_${ch}`);
    let Kc: Float64Array = new Float64Array(n);
    for (let p = 0; p < n; p++) {
      const d = dark === null ? 0 : typeof dark === 'number' ? dark : dark[3 * p + c];
      Kc[p] = (img.data[3 * p + c] - d) / den[ch];
    }
    if (opts.mask) {
      const v: number[] = [];
      for (let p = 0; p < n; p++) if (opts.mask[p]) v.push(Kc[p]);
      med[ch] = median(v);
      let m = 0;
      for (const x of v) m += x;
      m /= v.length;
      let s = 0;
      for (const x of v) s += (x - m) ** 2;
      rsd[ch] = Math.sqrt(s / v.length) / Math.abs(m);
      Kc = fillOutside(Kc, w, h, opts.mask, fill, opts.disc ?? null);
    }
    K[ch] = Kc;
  }
  let mf = NaN;
  if (opts.mask) { let s = 0; for (let p = 0; p < n; p++) s += opts.mask[p]; mf = s / n; }
  return { K, bandIntegral: bi, denominator: den, tauBar, KDiscMedian: med, KDiscRsd: rsd, maskFrac: mf, fill };
}

/** Fraction of the disc above `frac` of full scale per channel (flat_near_ceiling warning). */
export function fractionNearCeiling(img: Img3, disc: Uint8Array, ceiling: number, frac = 0.9): Rgb<number> {
  const out = { R: 0, G: 0, B: 0 };
  let nd = 0;
  for (let p = 0; p < img.width * img.height; p++) {
    if (!disc[p]) continue;
    nd++;
    for (let c = 0; c < 3; c++) if (img.data[3 * p + c] > frac * ceiling) out[CH[c]]++;
  }
  for (const c of CH) out[c] /= nd || 1;
  return out;
}
