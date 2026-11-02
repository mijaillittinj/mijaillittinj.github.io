/**
 * Blackbody oven calibration (WEB_SPEC §4, 1a). Aperture, plateau proposal and flatness, the dark by
 * intercept of S against t_exp or from a strip outside the aperture, the per channel fit of K_j with
 * its uncertainty and residuals, linearity, the propagation of the oven σ_T, the consistency criterion
 * declared before running, the ratio test against the oven temperature (leave one out), the channel
 * order check and the IR tail model uncertainty. ε is a parameter (1 by default).
 *
 * The same algorithm is written independently in numpy in manim/tools/make_web_fixtures.py
 * (`oven_reference`), and tests/workshop-rgb-oven.test.ts compares both at 1e-9.
 *
 * Images are chunky (H, W, 3) in 12 bit DN (frames.ts toDn12). Pixel coordinates are array indices,
 * r = √((y − cy)² + (x − cx)²). Times in µs in names ending in Us, seconds elsewhere. T in K inside.
 */
import { median, percentile, linfit, bisect } from './stats.ts';
import { CH, bandX, scaleTail, tailFraction, type Band, type Ch, type Rgb } from './radiometry.ts';
import { robustSigma } from './frames.ts';

export const T0C = 273.15;

export interface OvenFrame {
  tC: number;
  texpUs: number;
  width: number;
  height: number;
  /** (H, W, 3) in 12 bit DN. */
  dn: ArrayLike<number>;
  /** Per channel flags (H·W·3) to leave out: saturated (dilated) and hot. */
  flag?: Uint8Array;
}

export interface Aperture { cy: number; cx: number; r: number; bg: number; top: number }

/**
 * Aperture by moment fit on one channel (G by default). Threshold halfway between the median of the
 * frame (background) and its 99,95 percentile, centroid and r = √(area/π), twice, the second time
 * dropping pixels beyond 1,3 r (as find_sphere_disc).
 */
export function findAperture(f: OvenFrame, ch = 1): Aperture {
  const { width: w, height: h } = f;
  const n = w * h;
  const v = new Float64Array(n);
  for (let p = 0; p < n; p++) v[p] = f.dn[3 * p + ch];
  const bg = median(v), top = percentile(v, 99.95);
  const thr = bg + 0.5 * (top - bg);
  let m = new Uint8Array(n);
  for (let p = 0; p < n; p++) if (v[p] > thr) m[p] = 1;
  let cy = 0, cx = 0, r = 0;
  for (let it = 0; it < 2; it++) {
    let sy = 0, sx = 0, k = 0;
    for (let p = 0; p < n; p++) if (m[p]) { sy += Math.floor(p / w); sx += p % w; k++; }
    if (!k) throw new Error('oven_no_aperture');
    cy = sy / k; cx = sx / k; r = Math.sqrt(k / Math.PI);
    const m2 = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
      const dy = Math.floor(p / w) - cy, dx = (p % w) - cx;
      if (m[p] && Math.sqrt(dy * dy + dx * dx) < 1.3 * r) m2[p] = 1;
    }
    m = m2;
  }
  return { cy, cx, r, bg, top };
}

const radius = (p: number, w: number, cy: number, cx: number) => {
  const dy = Math.floor(p / w) - cy, dx = (p % w) - cx;
  return Math.sqrt(dy * dy + dx * dx);
};

const flagged = (f: OvenFrame, p: number) => !!f.flag && (f.flag[3 * p] === 1 || f.flag[3 * p + 1] === 1 || f.flag[3 * p + 2] === 1);

/**
 * Median per ring [k·step, (k+1)·step) and channel, leaving out the pixels flagged in that channel.
 * `n` counts the ring pixels with no flag in any channel.
 */
export function ringMedians(f: OvenFrame, cy: number, cx: number, step: number, nRings: number): { n: number[]; med: Rgb<number[]> } {
  const vals: number[][][] = Array.from({ length: nRings }, () => [[], [], []]);
  const n: number[] = new Array(nRings).fill(0);
  for (let p = 0; p < f.width * f.height; p++) {
    const k = Math.floor(radius(p, f.width, cy, cx) / step);
    if (k >= nRings) continue;
    if (!flagged(f, p)) n[k]++;
    for (let c = 0; c < 3; c++) if (!(f.flag && f.flag[3 * p + c])) vals[k][c].push(f.dn[3 * p + c]);
  }
  const med = { R: [] as number[], G: [] as number[], B: [] as number[] };
  for (let k = 0; k < nRings; k++) for (let c = 0; c < 3; c++) med[CH[c]].push(vals[k][c].length ? median(vals[k][c]) : NaN);
  return { n, med };
}

/**
 * Frame for the aperture and the plateau, the brightest in G (mean) among the frames with nothing
 * flagged, or the least flagged one when every frame has flags.
 */
export function choosePlateauFrame(frames: OvenFrame[]): OvenFrame {
  const nflag = (f: OvenFrame) => { let s = 0; if (f.flag) for (let i = 0; i < f.flag.length; i++) s += f.flag[i]; return s; };
  const clean = frames.filter((f) => nflag(f) === 0);
  let pool = clean;
  if (!pool.length) { let b = frames[0]; for (const f of frames) if (nflag(f) < nflag(b)) b = f; pool = [b]; }
  const meanG = (f: OvenFrame) => { let s = 0; for (let p = 0; p < f.width * f.height; p++) s += f.dn[3 * p + 1]; return s / (f.width * f.height); };
  let best = pool[0];
  for (const f of pool) if (meanG(f) > meanG(best)) best = f;
  return best;
}

export interface PlateauProfile {
  /** Ring radius in pixels of the frame (rings of block/2, ring 0 at r = 0). */
  r: number[];
  /** Blocks per ring. */
  n: number[];
  /** Ring median of the block means over the centre value, per channel, and of the block R/G and B/G. */
  I: Rgb<number[]>; RG: number[]; BG: number[];
  /** Centre values (median of the blocks within block/2 of the centre). */
  I0: Rgb<number>; RG0: number; BG0: number;
}

export interface PlateauShape { kI: Rgb<number>; kRG: number; kBG: number }

export interface Plateau {
  cy: number; cx: number; radius: number;
  /** Departure from the centre at the edge of the disc, from the fitted parabola, per channel and for the ratios. */
  flatness: Rgb<number>; flatRG: number; flatBG: number;
  shape: PlateauShape; profile: PlateauProfile; block: number;
}

/**
 * Radial profile on block × block means of the frame, with the block grid anchored so that one block is
 * centred on (cx, cy). A block enters when none of its pixels is flagged in any channel. Rings of block/2,
 * out to rMax. Averaging block² pixels brings the noise of a block well under the ±1 % tolerance, which a
 * profile on 1 px rings does not (GB2r1-numerico-01).
 */
export function plateauProfile(f: OvenFrame, cy: number, cx: number, rMax: number, block = 8): PlateauProfile {
  const B = block, w = f.width, h = f.height;
  const ox = Math.round(cx - (B - 1) / 2), oy = Math.round(cy - (B - 1) / 2);
  const ax = ((ox % B) + B) % B, ay = ((oy % B) + B) % B;
  const blocks: { d: number; m: number[] }[] = [];
  for (let y0 = ay; y0 + B <= h; y0 += B) for (let x0 = ax; x0 + B <= w; x0 += B) {
    const d = Math.hypot(x0 + (B - 1) / 2 - cx, y0 + (B - 1) / 2 - cy);
    if (d > rMax) continue;
    const s = [0, 0, 0];
    let ok = true;
    for (let y = 0; y < B && ok; y++) for (let x = 0; x < B; x++) {
      const p = (y0 + y) * w + x0 + x;
      if (flagged(f, p)) { ok = false; break; }
      s[0] += f.dn[3 * p]; s[1] += f.dn[3 * p + 1]; s[2] += f.dn[3 * p + 2];
    }
    if (ok) blocks.push({ d, m: s.map((v) => v / (B * B)) });
  }
  let centre = blocks.filter((q) => q.d < B / 2);
  if (!centre.length && blocks.length) { const dmin = Math.min(...blocks.map((q) => q.d)); centre = blocks.filter((q) => q.d <= dmin + 1e-9); }
  const I0 = { R: median(centre.map((q) => q.m[0])), G: median(centre.map((q) => q.m[1])), B: median(centre.map((q) => q.m[2])) };
  const RG0 = median(centre.map((q) => q.m[0] / q.m[1])), BG0 = median(centre.map((q) => q.m[2] / q.m[1]));
  const W = B / 2;
  const out: PlateauProfile = { r: [], n: [], I: { R: [], G: [], B: [] }, RG: [], BG: [], I0, RG0, BG0 };
  for (let q = 0; q * W <= rMax; q++) {
    const ring = blocks.filter((x) => x.d >= q * W && x.d < (q + 1) * W);
    if (!ring.length) continue;
    out.r.push(q === 0 ? 0 : (q + 0.5) * W);
    CH.forEach((c, k) => out.I[c].push(median(ring.map((x) => x.m[k])) / I0[c]));
    out.RG.push(median(ring.map((x) => x.m[0] / x.m[1])) / RG0);
    out.BG.push(median(ring.map((x) => x.m[2] / x.m[1])) / BG0);
    out.n.push(ring.length);
  }
  return out;
}

/** Relative curvature k of y ≈ a (1 + k ρ²), weighted least squares (weights n) over ρ ≤ fitRho. */
export function quadFitRho2(rho: number[], y: number[], w: number[], fitRho = 0.6): number {
  let S = 0, Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
  for (let i = 0; i < rho.length; i++) {
    if (rho[i] > fitRho || !Number.isFinite(y[i])) continue;
    const x = rho[i] * rho[i], ww = w[i];
    S += ww; Sx += ww * x; Sy += ww * y[i]; Sxx += ww * x * x; Sxy += ww * x * y[i];
  }
  const det = S * Sxx - Sx * Sx;
  if (!(det > 0)) return 0;
  const a = (Sxx * Sy - Sx * Sxy) / det, b = (S * Sxy - Sx * Sy) / det;
  return b / a;
}

/**
 * Plateau proposal (WEB_SPEC 1a.3, as the stage 1a maquette). Block means of `block` × `block` pixels,
 * ring medians, a parabola in ρ² = (r/r_aperture)² fitted from the centre to `fitRho` radii, and the
 * largest disc where the fitted intensity stays within ±tolI of the centre in every channel used and the
 * fitted R/G and B/G within ±tolRatio. Never smaller than half a block. Flatness is read from the fit at
 * the edge of the disc.
 */
export function proposePlateau(f: OvenFrame, ap: Aperture, opts: { block?: number; tolI?: number; tolRatio?: number; fitRho?: number; use?: Rgb<boolean> } = {}): Plateau {
  const B = opts.block ?? 8, tolI = opts.tolI ?? 0.01, tolR = opts.tolRatio ?? 0.005, fitRho = opts.fitRho ?? 0.6;
  const use = opts.use ?? { R: true, G: true, B: true };
  const prof = plateauProfile(f, ap.cy, ap.cx, 1.2 * ap.r, B);
  const rho = prof.r.map((v) => v / ap.r);
  const shape: PlateauShape = {
    kI: { R: quadFitRho2(rho, prof.I.R, prof.n, fitRho), G: quadFitRho2(rho, prof.I.G, prof.n, fitRho), B: quadFitRho2(rho, prof.I.B, prof.n, fitRho) },
    kRG: quadFitRho2(rho, prof.RG, prof.n, fitRho), kBG: quadFitRho2(rho, prof.BG, prof.n, fitRho),
  };
  const lim = (tol: number, k: number) => (Math.abs(k) > 1e-9 ? Math.sqrt(tol / Math.abs(k)) : 1);
  let r1 = 1;
  for (const c of CH) if (use[c]) r1 = Math.min(r1, lim(tolI, shape.kI[c]));
  if (use.R && use.G) r1 = Math.min(r1, lim(tolR, shape.kRG));
  if (use.B && use.G) r1 = Math.min(r1, lim(tolR, shape.kBG));
  const radius = Math.max(B / 2, r1 * ap.r);
  const x = (radius / ap.r) ** 2;
  return {
    cy: ap.cy, cx: ap.cx, radius, block: B, shape, profile: prof,
    flatness: { R: Math.abs(shape.kI.R) * x, G: Math.abs(shape.kI.G) * x, B: Math.abs(shape.kI.B) * x },
    flatRG: Math.abs(shape.kRG) * x, flatBG: Math.abs(shape.kBG) * x,
  };
}

export interface FrameSignal {
  tC: number; texpUs: number;
  S: Rgb<number>; p999: Rgb<number>; nSat: Rgb<number>; n: Rgb<number>;
  /** Strip outside the aperture (r > stripFactor · r_aperture): median and robust σ per channel. */
  strip: Rgb<number>; stripSigma: Rgb<number>; nStrip: Rgb<number>;
}

/**
 * Plateau median, p99,9 and flagged count per channel, and the outside strip, for one frame. A pixel
 * flagged in one channel leaves only that channel (the other channels of the image still enter).
 */
export function frameSignal(f: OvenFrame, pl: { cy: number; cx: number; radius: number }, ap: Aperture, stripFactor = 2.5): FrameSignal {
  const v: number[][] = [[], [], []], s: number[][] = [[], [], []];
  const nSat = { R: 0, G: 0, B: 0 };
  for (let p = 0; p < f.width * f.height; p++) {
    const r = radius(p, f.width, pl.cy, pl.cx);
    const inside = r < pl.radius;
    const outside = !inside && radius(p, f.width, ap.cy, ap.cx) > stripFactor * ap.r;
    for (let c = 0; c < 3; c++) {
      const fl = !!f.flag && f.flag[3 * p + c] === 1;
      if (inside) { if (fl) nSat[CH[c]]++; else v[c].push(f.dn[3 * p + c]); }
      else if (outside && !fl) s[c].push(f.dn[3 * p + c]);
    }
  }
  const S = { R: NaN, G: NaN, B: NaN }, p999 = { R: NaN, G: NaN, B: NaN }, strip = { R: NaN, G: NaN, B: NaN }, stripSigma = { R: NaN, G: NaN, B: NaN };
  const n = { R: 0, G: 0, B: 0 }, nStrip = { R: 0, G: 0, B: 0 };
  for (let c = 0; c < 3; c++) {
    const k = CH[c];
    n[k] = v[c].length; nStrip[k] = s[c].length;
    if (v[c].length) { S[k] = median(v[c]); p999[k] = percentile(v[c], 99.9); }
    if (s[c].length) { const r = robustSigma(s[c]); strip[k] = r.med; stripSigma[k] = r.sigma; }
  }
  return { tC: f.tC, texpUs: f.texpUs, S, p999, nSat, n, strip, stripSigma, nStrip };
}

export type DarkMode = 'intercept' | 'strip' | 'files';

export interface OvenFitOptions {
  band: Band;
  /** Cavity emissivity, a parameter (decision of the professor, default 1). */
  eps?: number;
  /** Oven thermocouple uncertainty in K (default 5, an assumption). */
  sigmaT?: number;
  ceilingDn?: number;
  darkMode?: DarkMode;
  /** D_j = a_j + b_j t_exp(s) from dark frames, for darkMode 'files'. */
  darkLine?: Rgb<{ a: number; b: number }>;
  /** A channel of a frame enters the fit if p99,9 < maxFrac × ceiling, nothing flagged in the plateau, S − D > minSnr σ. */
  maxFrac?: number;
  minSnr?: number;
  /** σ of the dark in DN when no strip is available. */
  noiseDn?: number;
  minTexpUs?: number;
  /** IR tail cut for the model uncertainty (×0 and ×4 above it). */
  tailCutNm?: number;
  /**
   * K_j(T) per temperature. 'slope' (default, WEB_SPEC 1a.5) is the slope of S − D against t_exp over the
   * accepted frames. 'band' is the convention of manim/tools/calibration.py, the one behind the numbers of
   * the presentation: D_j(T) the intercept of S against t_exp over the frames between 5 and 90 % of the
   * scale (all unsaturated frames if fewer than 3), and K_j(T) the mean of (S − D)/t_exp over the frames
   * between 50 and 70 % of the scale (then 5 to 90 %, then any unsaturated frame over 5 %), over B_j(T).
   */
  perT?: 'slope' | 'band';
  /**
   * Which temperatures give K_j. 'highest' (default, director decision at the close of G2) takes K_j and
   * K_R/K_G from the highest oven temperature where R and G have usable frames, and the other temperatures
   * are the test (`ratioTestTop`). 'all' combines every temperature, a line through the origin with
   * 'slope' and the geometric mean with 'band' (as calibration.py).
   */
  calibrateAt?: 'highest' | 'all';
}

export interface ChannelTFit { tC: number; TK: number; n: number; a: number; b: number; sb: number; B: number; K: number; sK: number; linMaxPct: number; tail: number }

export interface ChannelFit {
  ch: Ch;
  perT: ChannelTFit[];
  /** K by the `calibrateAt` convention (T_cal = the temperature it comes from, NaN for 'all'). */
  K: number; sKstat: number; sKT: number; sK: number; Tcal: number;
  /** K combining every temperature (line through the origin or geometric mean), always given for comparison. */
  Kall: number; sKallStat: number;
  /** Relative residual (b − K B)/(K B) per temperature. */
  resid: number[];
  weights: 'sigma' | 'relative';
  /** K with the IR tail × 0 and × 4, relative to K. */
  tailLo: number; tailHi: number;
}

export interface RatioCheck { pair: 'RG' | 'BG'; temps: number[]; lnq: number[]; dispersion: number; allowedT: number; allowedStat: number; allowed: number; pass: boolean; tested: boolean }

export interface RatioTestRow { tC: number; TK: number; rMeas: number; Tpred: number; dT: number }

export interface OvenFit {
  signals: FrameSignal[];
  accepted: Rgb<boolean[]>;
  channels: Rgb<ChannelFit>;
  ratio: { RG: RatioCheck; BG: RatioCheck };
  consistent: boolean;
  calibrated: Ch[];
  /** R/G of the blackbody with K_j fitted leaving that temperature out (every temperature). */
  ratioTest: RatioTestRow[];
  /** R/G of the blackbody with K_R/K_G of the highest temperature, at every other temperature. */
  ratioTestTop: RatioTestRow[];
  Ttop: number;
  perT: 'slope' | 'band';
  calibrateAt: 'highest' | 'all';
  order: { ok: boolean; proposal: Ch[] };
  darkMode: DarkMode;
  strayPct: Rgb<number>;
}

/**
 * Weighted line through the origin b = K B. With σ_b known the uncertainty is the fit σ times
 * max(1, √(χ²/(n − 1))). With relative weights 1/b² (σ_b unknown) the prior σ is meaningless (100 %),
 * so the full Birge factor √(χ²/(n − 1)) is used without the max, which gives K · s(resid)/√n
 * (GB2r1-numerico-02). With one point and relative weights σ is K (unknown).
 */
export function throughOrigin(b: number[], B: number[], sb: number[]): { K: number; sStat: number; weights: 'sigma' | 'relative' } {
  const useSigma = sb.every((s) => Number.isFinite(s) && s > 0);
  const w = b.map((bi, i) => (useSigma ? 1 / (sb[i] * sb[i]) : 1 / (bi * bi)));
  let num = 0, den = 0;
  for (let i = 0; i < b.length; i++) { num += w[i] * b[i] * B[i]; den += w[i] * B[i] * B[i]; }
  const K = num / den;
  let chi2 = 0;
  for (let i = 0; i < b.length; i++) chi2 += w[i] * (b[i] - K * B[i]) ** 2;
  const n = b.length;
  const birge = n > 1 ? Math.sqrt(chi2 / (n - 1)) : 1;
  const sStat = Math.sqrt(1 / den) * (useSigma ? Math.max(1, birge) : birge);
  return { K, sStat, weights: useSigma ? 'sigma' : 'relative' };
}

/** Geometric mean of K_j(T) and its standard error, as calibration.py k_fit. */
function geoMean(K: number[]): { K: number; sStat: number } {
  const lk = K.map(Math.log), n = lk.length;
  const m = lk.reduce((a, b) => a + b, 0) / n;
  const sd = n > 1 ? Math.sqrt(lk.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1)) : 0;
  const Kg = Math.exp(m);
  return { K: Kg, sStat: (Kg * sd) / Math.sqrt(n) };
}

export const FIT_BAND = [0.5, 0.7] as const;
export const LIN_BAND = [0.05, 0.9] as const;

/** Full oven fit. Frames already reduced to `FrameSignal` (one per image), so the fit itself is cheap. */
export function fitOven(signals: FrameSignal[], opts: OvenFitOptions): OvenFit {
  const eps = opts.eps ?? 1, sigmaT = opts.sigmaT ?? 5, ceil = opts.ceilingDn ?? 4095;
  const maxFrac = opts.maxFrac ?? 0.9, minSnr = opts.minSnr ?? 10, minT = opts.minTexpUs ?? 100;
  const mode = opts.darkMode ?? 'intercept', cut = opts.tailCutNm ?? 700;
  const perTMode = opts.perT ?? 'slope', calAt = opts.calibrateAt ?? 'highest';
  const temps = [...new Set(signals.map((s) => s.tC))].sort((a, b) => a - b);
  const D = (s: FrameSignal, c: Ch) => mode === 'strip' ? s.strip[c] : mode === 'files' && opts.darkLine ? opts.darkLine[c].a + opts.darkLine[c].b * s.texpUs * 1e-6 : 0;
  const noise = (s: FrameSignal, c: Ch) => Number.isFinite(s.stripSigma[c]) ? s.stripSigma[c] : opts.noiseDn ?? 1;
  const accepted = { R: [] as boolean[], G: [] as boolean[], B: [] as boolean[] };
  const frac = (s: FrameSignal, c: Ch) => s.S[c] / ceil;
  const inBand = (s: FrameSignal, c: Ch, band: readonly [number, number]) => s.nSat[c] === 0 && frac(s, c) >= band[0] && frac(s, c) <= band[1];
  for (const s of signals) for (const c of CH) {
    if (perTMode === 'slope') accepted[c].push(s.nSat[c] === 0 && s.p999[c] < maxFrac * ceil && s.S[c] - D(s, c) > minSnr * noise(s, c) && s.texpUs >= minT);
    else accepted[c].push(s.nSat[c] === 0 && frac(s, c) >= LIN_BAND[0] && s.texpUs >= minT);
  }

  const bands = { lo: scaleTail(opts.band, 0, cut), hi: scaleTail(opts.band, 4, cut) };
  const combine = (b: number[], B: number[], sb: number[]) => {
    if (perTMode === 'band') { const g = geoMean(b.map((v, i) => v / B[i])); return { K: g.K, sStat: g.sStat, weights: 'relative' as const }; }
    return throughOrigin(b, B, sb);
  };
  const channels = {} as Rgb<ChannelFit>;
  for (const c of CH) {
    const perT: ChannelTFit[] = [];
    for (const tC of temps) {
      const TK = tC + T0C;
      const B = bandX(opts.band, c, TK, eps);
      const tail = tailFraction(opts.band, c, TK, cut, 'X');
      if (perTMode === 'band') {
        const all = signals.map((_, i) => i).filter((i) => signals[i].tC === tC && signals[i].nSat[c] === 0 && signals[i].texpUs >= minT);
        if (!all.length) continue;
        let a = 0;
        if (mode === 'intercept') {
          let li = all.filter((i) => inBand(signals[i], c, LIN_BAND));
          if (li.length < 3) li = all;
          a = li.length >= 2 ? linfit(li.map((i) => signals[i].texpUs * 1e-6), li.map((i) => signals[i].S[c])).a : 0;
        }
        const Dd = (i: number) => (mode === 'intercept' ? a : D(signals[i], c));
        let pts = all.filter((i) => inBand(signals[i], c, FIT_BAND));
        if (!pts.length) pts = all.filter((i) => inBand(signals[i], c, LIN_BAND));
        if (!pts.length) pts = all.filter((i) => frac(signals[i], c) >= LIN_BAND[0]);
        if (!pts.length) continue;
        const y = pts.map((i) => (signals[i].S[c] - Dd(i)) / (signals[i].texpUs * 1e-6));
        const b = y.reduce((p, q) => p + q, 0) / y.length;
        const sb = y.length > 1 ? Math.sqrt(y.reduce((p, q) => p + (q - b) ** 2, 0) / (y.length - 1)) / Math.sqrt(y.length) : NaN;
        perT.push({ tC, TK, n: pts.length, a: mode === 'intercept' ? a : NaN, b, sb, B, K: b / B, sK: sb / B, linMaxPct: NaN, tail });
        continue;
      }
      const idx = signals.map((_, i) => i).filter((i) => signals[i].tC === tC && accepted[c][i]);
      if (!idx.length) continue;
      const t = idx.map((i) => signals[i].texpUs * 1e-6), y = idx.map((i) => signals[i].S[c] - D(signals[i], c));
      let a = NaN, b: number, sb = NaN, lin = NaN;
      if (idx.length === 1) b = y[0] / t[0];
      else {
        const fit = linfit(t, y);
        a = fit.a; b = fit.b; sb = fit.sb;
        if (idx.length >= 3) { lin = 0; for (let k = 0; k < t.length; k++) { const m = a + b * t[k]; lin = Math.max(lin, Math.abs((y[k] - m) / m) * 100); } }
      }
      perT.push({ tC, TK, n: idx.length, a, b, sb, B, K: b / B, sK: sb / B, linMaxPct: lin, tail });
    }
    channels[c] = { ch: c, perT } as ChannelFit;
  }
  // highest oven temperature where R and G both have usable frames (G2 convention)
  const tRG = channels.R.perT.map((p) => p.tC).filter((t) => channels.G.perT.some((q) => q.tC === t));
  const Ttop = tRG.length ? Math.max(...tRG) : NaN;
  for (const c of CH) {
    const ch = channels[c], perT = ch.perT;
    const bs = perT.map((p) => p.b), sbs = perT.map((p) => p.sb);
    const at = (dT: number, band = opts.band) => perT.map((p) => bandX(band, c, p.TK + dT, eps));
    const g = perT.length ? combine(bs, at(0), sbs) : { K: NaN, sStat: NaN, weights: 'relative' as const };
    let K = g.K, sKstat = g.sStat, sKT: number, klo: number, khi: number, Tcal = NaN;
    const top = perT.find((p) => p.tC === Ttop) ?? (calAt === 'highest' && perT.length ? perT[perT.length - 1] : undefined);
    if (calAt === 'highest' && top) {
      Tcal = top.TK; K = top.K; sKstat = Number.isFinite(top.sK) ? top.sK : 0;
      sKT = K * Math.abs(Math.log(bandX(opts.band, c, top.TK + sigmaT, eps)) - Math.log(bandX(opts.band, c, top.TK - sigmaT, eps))) / 2;
      klo = top.b / bandX(bands.lo, c, top.TK, eps); khi = top.b / bandX(bands.hi, c, top.TK, eps);
    } else {
      sKT = perT.length ? Math.abs(combine(bs, at(sigmaT), sbs).K - combine(bs, at(-sigmaT), sbs).K) / 2 : NaN;
      klo = perT.length ? combine(bs, at(0, bands.lo), sbs).K : NaN; khi = perT.length ? combine(bs, at(0, bands.hi), sbs).K : NaN;
    }
    channels[c] = {
      ch: c, perT, K, sKstat, sKT, sK: Math.hypot(sKstat, sKT), Tcal, Kall: g.K, sKallStat: g.sStat,
      resid: perT.map((p) => (p.b - K * p.B) / (K * p.B)), weights: g.weights,
      tailLo: klo / K - 1, tailHi: khi / K - 1,
    };
  }

  const ratioCheck = (num: Ch, pair: 'RG' | 'BG'): RatioCheck => {
    const a = channels[num].perT, b = channels.G.perT;
    const common = a.filter((p) => b.some((q) => q.tC === p.tC));
    const lnq: number[] = [], dT: number[] = [], st: number[] = [];
    for (const p of common) {
      const q = b.find((x) => x.tC === p.tC)!;
      lnq.push(Math.log(p.K / q.K));
      dT.push(-(Math.log(bandX(opts.band, num, p.TK + sigmaT, eps)) - Math.log(p.B)) + (Math.log(bandX(opts.band, 'G', q.TK + sigmaT, eps)) - Math.log(q.B)));
      const r1 = Number.isFinite(p.sK) ? p.sK / p.K : 0, r2 = Number.isFinite(q.sK) ? q.sK / q.K : 0;
      st.push(r1 * r1 + r2 * r2);
    }
    const n = lnq.length;
    const m = n ? lnq.reduce((x, y) => x + y, 0) / n : NaN;
    const dispersion = n ? Math.sqrt(lnq.reduce((x, y) => x + (y - m) ** 2, 0) / n) : NaN;
    const allowedT = n ? Math.sqrt(dT.reduce((x, y) => x + y * y, 0) / n) : NaN;
    const allowedStat = n ? Math.sqrt(st.reduce((x, y) => x + y, 0) / n) : NaN;
    const allowed = Math.hypot(allowedT, allowedStat);
    const tested = n >= 2;
    return { pair, temps: common.map((p) => p.tC), lnq, dispersion, allowedT, allowedStat, allowed, pass: !tested || dispersion <= allowed, tested };
  };
  const RG = ratioCheck('R', 'RG'), BG = ratioCheck('B', 'BG');
  const calibrated: Ch[] = CH.filter((c) => channels[c].perT.length > 0);
  const consistent = RG.pass && calibrated.includes('R') && calibrated.includes('G');
  const cal = consistent ? (BG.pass && calibrated.includes('B') ? (['R', 'G', 'B'] as Ch[]) : (['R', 'G'] as Ch[])) : [];

  // ratio test, R/G of the blackbody with K_j fitted leaving that temperature out
  const ratioTest: RatioTestRow[] = [];
  for (const p of channels.R.perT) {
    const q = channels.G.perT.find((x) => x.tC === p.tC);
    if (!q) continue;
    const oR = channels.R.perT.filter((x) => x.tC !== p.tC), oG = channels.G.perT.filter((x) => x.tC !== p.tC);
    if (!oR.length || !oG.length) continue;
    const KR = combine(oR.map((x) => x.b), oR.map((x) => x.B), oR.map((x) => x.sb)).K;
    const KG = combine(oG.map((x) => x.b), oG.map((x) => x.B), oG.map((x) => x.sb)).K;
    const rMeas = p.b / q.b;
    const f = (T: number) => Math.log((KR * bandX(opts.band, 'R', T, eps)) / (KG * bandX(opts.band, 'G', T, eps))) - Math.log(rMeas);
    const Tpred = bisect(f, 700, 3000, 1e-14);
    ratioTest.push({ tC: p.tC, TK: p.TK, rMeas, Tpred, dT: Tpred - p.TK });
  }
  // ratio test with K_R/K_G of the highest temperature (the default convention), at the other temperatures
  const ratioTestTop: RatioTestRow[] = [];
  const pRt = channels.R.perT.find((x) => x.tC === Ttop), pGt = channels.G.perT.find((x) => x.tC === Ttop);
  if (pRt && pGt) for (const p of channels.R.perT) {
    const q = channels.G.perT.find((x) => x.tC === p.tC);
    if (!q || p.tC === Ttop) continue;
    const rMeas = p.b / q.b;
    const f = (T: number) => Math.log((pRt.K * bandX(opts.band, 'R', T, eps)) / (pGt.K * bandX(opts.band, 'G', T, eps))) - Math.log(rMeas);
    const Tpred = bisect(f, 700, 3000, 1e-14);
    ratioTestTop.push({ tC: p.tC, TK: p.TK, rMeas, Tpred, dT: Tpred - p.TK });
  }

  // channel order at the oven, R > G > B on the slopes at every temperature
  let ok = true;
  const sumB = { R: 0, G: 0, B: 0 };
  for (const tC of temps) {
    const v = CH.map((c) => channels[c].perT.find((x) => x.tC === tC)?.b ?? NaN);
    if (v.every(Number.isFinite)) { if (!(v[0] > v[1] && v[1] > v[2])) ok = false; CH.forEach((c, k) => (sumB[c] += v[k])); }
  }
  const proposal = [...CH].sort((x, y) => sumB[y] - sumB[x]) as Ch[];

  // stray light, growth of the strip with t_exp as a percentage of the plateau, per channel
  const strayPct = { R: NaN, G: NaN, B: NaN };
  for (const c of CH) {
    let worst = NaN;
    for (const tC of temps) {
      const ss = signals.filter((s) => s.tC === tC && Number.isFinite(s.strip[c]) && Number.isFinite(s.S[c])).sort((x, y) => x.texpUs - y.texpUs);
      if (ss.length < 2) continue;
      const lo = ss[0], hi = ss[ss.length - 1];
      const v = ((hi.strip[c] - lo.strip[c]) / (hi.S[c] - hi.strip[c])) * 100;
      if (!(worst >= v)) worst = v;
    }
    strayPct[c] = worst;
  }
  return { signals, accepted, channels, ratio: { RG, BG }, consistent, calibrated: cal, ratioTest, ratioTestTop, Ttop, perT: perTMode, calibrateAt: calAt, order: { ok, proposal }, darkMode: mode, strayPct };
}

/** Linear fit of dark frames D = a + b·t_exp(s) per channel (darkMode 'files'). */
export function fitDarkLine(darks: { texpUs: number; level: Rgb<number> }[]): Rgb<{ a: number; b: number }> {
  const out = {} as Rgb<{ a: number; b: number }>;
  for (const c of CH) {
    if (darks.length === 1) { out[c] = { a: darks[0].level[c], b: 0 }; continue; }
    const f = linfit(darks.map((d) => d.texpUs * 1e-6), darks.map((d) => d.level[c]));
    out[c] = { a: f.a, b: f.b };
  }
  return out;
}
