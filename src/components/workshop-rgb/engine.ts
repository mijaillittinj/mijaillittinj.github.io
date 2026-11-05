/**
 * Processing of the workshop tool that runs in the coordinator Worker (pyro.worker.ts), written as
 * plain functions so they can also run in Node. It ties the numerical core of
 * src/lib/workshop-rgb/ to the stages of WEB_SPEC §4 in the order of the brief (§3).
 *   1 dark or background with the same t_exp, 2 divide by t_exp and K_j(x, y), 3 orientation, axis,
 *   halves and mm, 4 Abel per channel, 5 ratio for T, 6 f_s = j_j / G_j(T).
 * Images are chunky (H, W, 3), row 0 at the top, in 12 bit DN (frames.ts toDn12). A "working pixel"
 * is the 2 × 2 superpixel by default (decision of the director after GB1), so mm per pixel of the
 * sensor is multiplied by `scale` in one place (`prepareFlame`).
 * Nothing here writes prose, only codes and numbers.
 */
import { readTiff, detectEncoding, type Encoding } from '../../lib/workshop-rgb/tiff.ts';
import { saturationMask, dilate3x3, replicationPhase, bayerSuperpixel, bin2x2, pylonSites, rgbShiftTest, hotPixelsFromDark, hotPixelsNoDark, replaceHot, type BayerPattern, type ShiftTest } from '../../lib/workshop-rgb/frames.ts';
import { findAperture, proposePlateau, plateauProfile, quadFitRho2, frameSignal, fitOven, fitDarkLine, T0C, type OvenFrame, type Aperture, type Plateau, type FrameSignal, type OvenFit, type DarkMode } from '../../lib/workshop-rgb/oven.ts';
import { tailGrid, tailPoints, tailCalibration, type TailCal, type TailGrid, type TailSignal } from '../../lib/workshop-rgb/tailcal.ts';
import { readGeometry, readSignal, referenceIndex, commonRadius, type ReadGeometry } from '../../lib/workshop-rgb/ovenread.ts';
import { CH, makeBand, bandX, tailFraction, ratioTable, solveTfs, refineTfs, maskTfs, signalMask, fsFromChannel, fsChannels, kernelTableTrapz, fitColourTemperature, type Band, type Rgb, type Ch, type EmLaw, type RatioTable } from '../../lib/workshop-rgb/radiometry.ts';
import { findSphereDisc, sphereMask, calibrationConstants, loadSphereSpectrum, resample, ndTransmission, effectiveNdTransmission, fractionNearCeiling, fillOutside, type SpectrumUnit } from '../../lib/workshop-rgb/sphere.ts';
import { findAxis } from '../../lib/workshop-rgb/geometry.ts';
import { invertAll, lcurveCorner, logspace, halvesSigma, overRegularised, LAMBDA_REL_MIN, LAMBDA_REL_MAX, type AbelMethod } from '../../lib/workshop-rgb/invert.ts';
import { chordMatrixHalf } from '../../lib/abel.ts';
import { satFactor, satApply, splineBasis } from '../../lib/sat.ts';
import { matvec, rng, type Mat } from '../../lib/linalg.ts';
import { fineGrid, forNearest, basePointsOnFine, splineBasisAt, splineX, satCoef, satEval, refineFactor, type FineGrid } from '../../lib/workshop-rgb/refine.ts';
import { Net } from '../../lib/workshop-rgb/annabel/net.ts';
import { coordFeatures, type CoordNorm } from '../../lib/workshop-rgb/annabel/problem.ts';
import { predictTfs } from '../../lib/workshop-rgb/annabel/stage2.ts';
import { PER_MM_TO_PER_M } from '../../lib/workshop-rgb/annabel/units.ts';
import { median, percentile, interp1 } from '../../lib/workshop-rgb/stats.ts';

const interp1Safe = (x: number, xp: ArrayLike<number>, fp: ArrayLike<number>) => interp1(x, xp, fp, NaN, NaN);

export type { Rgb, Ch, EmLaw };

// ------------------------------------------------------------------------------------- frames
export type Binning = 'rgb2x2' | 'rgb1x1' | 'bayer2x2';

export interface FrameInfo {
  width: number; height: number; samples: number; bits: number;
  encoding: Encoding; ceiling: number; ceil12: number; toDn12: number; measured: boolean; ambiguous: boolean;
  littleEndian: boolean; compression: number; software?: string; model?: string; orientation: number; cfa?: string;
  replicated: boolean; phaseR: [number, number] | null; phaseB: [number, number] | null;
  /** Working grid. */
  w: number; h: number; scale: number; gx0: number; gy0: number; binning: Binning;
  /** Per channel, in 12 bit DN or counts of raw pixels. */
  p999: number[]; max12: number[]; satRaw: number[]; zeroFrac: number[];
  meanG: number; nSatWork: number;
  bytes: number; ms: number;
}

export interface Frame { w: number; h: number; dn: Float32Array; sat: Uint8Array }

/** `centre` moves R and B of the grouped pylon RGB onto the green centre. Off by default (decision on closing G4), because
 * the direction of the shift is not confirmed on an a2A1920 image with edges. `rulerShift` tests it on the ruler. */
export interface LoadOptions { group: boolean; bayer: BayerPattern; centre?: boolean }
/** Encoding decided for the whole series with its brightest frame (WEB_SPEC §5.3), applied to its darks. */
export interface EncOverride { encoding: Encoding; toDn12: number; ceiling: number }

/** Histogram percentile of raw counts of one channel (exact for integer data). */
function rawStats(d: ArrayLike<number>, nc: number, c: number, ceiling: number, toDn12: number, float: boolean) {
  const n = d.length / nc;
  let max = 0, sat = 0, zero = 0;
  const thr = 0.995 * ceiling;
  if (float) {
    const v = new Float64Array(n);
    for (let p = 0; p < n; p++) { const x = d[p * nc + c]; v[p] = x; if (x > max) max = x; if (x === 0) zero++; }
    return { max12: max, p999: percentile(v, 99.9), sat: 0, zeroFrac: zero / n };
  }
  const hist = new Uint32Array(65536);
  for (let p = 0; p < n; p++) {
    const x = d[p * nc + c];
    hist[x]++;
    if (x > max) max = x;
    if (x >= thr) sat++;
    if (x === 0) zero++;
  }
  const target = 0.999 * n;
  let acc = 0, q = 0;
  for (; q < 65536; q++) { acc += hist[q]; if (acc >= target) break; }
  return { max12: max * toDn12, p999: q * toDn12, sat, zeroFrac: zero / n };
}

/** One TIFF from bytes to the working frame, with its summary (WEB_SPEC §5.3). */
export function loadFrame(buf: ArrayBuffer, opts: LoadOptions, force?: EncOverride | null): { info: FrameInfo; frame: Frame } {
  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
  const img = readTiff(buf);
  const det = detectEncoding(img);
  const enc = force && !img.float && img.bits === 16 ? { ...det, encoding: force.encoding, toDn12: force.toDn12, ceiling: force.ceiling, ambiguous: false } : det;
  const W = img.width, H = img.height;
  let nc = img.samples;
  let data: ArrayLike<number> = img.data;
  if (nc === 4 || nc === 2) {
    const k = nc;
    const out = new (img.float ? Float32Array : img.bits === 8 ? Uint8Array : Uint16Array)(W * H * 3);
    for (let p = 0; p < W * H; p++) for (let c = 0; c < 3; c++) out[p * 3 + c] = img.data[p * k + Math.min(c, k - 1)];
    data = out; nc = 3;
  }
  const ceil = enc.ceiling;
  const stats = [0, 1, 2].map((c) => rawStats(data, nc, Math.min(c, nc - 1), ceil, enc.toDn12, img.float));
  let frame: Frame, binning: Binning, scale = 1, gx0 = 0, gy0 = 0;
  let replicated = false, phaseR: [number, number] | null = null, phaseB: [number, number] | null = null;
  if (nc === 1) {
    const satM = saturationMask({ width: W, height: H, nc: 1, data }, ceil);
    const pat = (img.cfa && /^[RGB]{4}$/.test(img.cfa) ? img.cfa : opts.bayer) as BayerPattern;
    const b = bayerSuperpixel({ width: W, height: H, data }, pat, satM, true);
    for (let i = 0; i < b.data.length; i++) b.data[i] *= enc.toDn12;
    frame = { w: b.width, h: b.height, dn: b.data, sat: b.sat };
    binning = 'bayer2x2'; scale = 2;
  } else {
    const planes = { width: W, height: H, nc: 3, data };
    let satM = saturationMask(planes, ceil);
    const ph = replicationPhase(planes);
    replicated = ph[0].replicated;
    phaseR = ph[0].replicated ? ph[0].phase : null;
    phaseB = ph[2].replicated ? ph[2].phase : null;
    if (replicated) satM = dilate3x3(satM, W, H, 3, true);
    if (opts.group) {
      gy0 = phaseR ? phaseR[0] : 0; gx0 = phaseR ? phaseR[1] : 0;
      // R and B centred on the green like the Bayer path only when asked (G4r3-conf-24, off by default on closing G4).
      const pat = (img.cfa && /^[RGB]{4}$/.test(img.cfa) ? img.cfa : opts.bayer) as BayerPattern;
      const b = bin2x2(planes, satM, gy0, gx0, opts.centre ? pylonSites(pat, phaseR, phaseB) : undefined);
      for (let i = 0; i < b.data.length; i++) b.data[i] *= enc.toDn12;
      frame = { w: b.width, h: b.height, dn: b.data, sat: b.sat };
      binning = 'rgb2x2'; scale = 2;
    } else {
      const dn = new Float32Array(W * H * 3);
      for (let i = 0; i < dn.length; i++) dn[i] = data[i] * enc.toDn12;
      frame = { w: W, h: H, dn, sat: satM };
      binning = 'rgb1x1';
    }
  }
  let sg = 0, nsat = 0;
  for (let p = 0; p < frame.w * frame.h; p++) { sg += frame.dn[3 * p + 1]; if (frame.sat[3 * p] || frame.sat[3 * p + 1] || frame.sat[3 * p + 2]) nsat++; }
  const info: FrameInfo = {
    width: W, height: H, samples: img.samples, bits: img.bits,
    encoding: enc.encoding, ceiling: ceil, ceil12: ceil * enc.toDn12, toDn12: enc.toDn12, measured: enc.measured, ambiguous: enc.ambiguous,
    littleEndian: img.littleEndian, compression: img.compression, software: img.software, model: img.model, orientation: img.orientation, cfa: img.cfa,
    replicated, phaseR, phaseB, w: frame.w, h: frame.h, scale, gx0, gy0, binning,
    p999: stats.map((s) => s.p999), max12: stats.map((s) => s.max12), satRaw: stats.map((s) => s.sat), zeroFrac: stats.map((s) => s.zeroFrac),
    meanG: sg / (frame.w * frame.h), nSatWork: nsat, bytes: buf.byteLength,
    ms: (typeof performance !== 'undefined' ? performance.now() : 0) - t0,
  };
  return { info, frame };
}

/**
 * Shift test of R and B on the ruler image (decision on closing G4). The file is read again grouped on the red
 * phase and without centring, whatever the page options, and `rgbShiftTest` says whether the centring helps.
 * A file without the pylon replication of R and B (Bayer mosaic, synthetic or other software) gives `not_applicable`.
 */
export function rulerShift(buf: ArrayBuffer, bayer: BayerPattern, force?: EncOverride | null): ShiftTest & { pattern: BayerPattern } {
  const img = readTiff(buf);
  const pat = (img.cfa && /^[RGB]{4}$/.test(img.cfa) ? img.cfa : bayer) as BayerPattern;
  const { info, frame } = loadFrame(buf, { group: true, bayer, centre: false }, force);
  if (info.binning !== 'rgb2x2' || !info.replicated) return { ...rgbShiftTest(frame, { R: null, B: null }), pattern: pat };
  return { ...rgbShiftTest(frame, pylonSites(pat, info.phaseR, info.phaseB)), pattern: pat };
}

/** Small RGB thumbnail with gamma 2,2 for viewing only, each channel over the ceiling. */
export function thumbnail(f: Frame, ceil12: number, maxW = 160): { w: number; h: number; rgba: Uint8ClampedArray } {
  const step = Math.max(1, Math.ceil(f.w / maxW));
  const w = Math.floor(f.w / step), h = Math.floor(f.h / step);
  const out = new Uint8ClampedArray(w * h * 4);
  const s = Number.isFinite(ceil12) && ceil12 > 0 ? ceil12 : 1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const p = (y * step * f.w + x * step) * 3, o = (y * w + x) * 4;
    for (let c = 0; c < 3; c++) out[o + c] = 255 * Math.pow(Math.min(1, Math.max(0, f.dn[p + c] / s)), 1 / 2.2);
    out[o + 3] = 255;
  }
  return { w, h, rgba: out };
}

/** Channel planes of a frame over a scale, downsampled to at most maxW columns, for the canvas previews. */
export function previewPlanes(f: Frame, norm: number, maxW = 480, flags?: Uint8Array): { w: number; h: number; step: number; v: Float32Array; flag: Uint8Array } {
  const step = Math.max(1, Math.ceil(f.w / maxW));
  const w = Math.floor(f.w / step), h = Math.floor(f.h / step);
  const v = new Float32Array(w * h * 3), flag = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let fl = 0;
    const acc = [0, 0, 0];
    for (let dy = 0; dy < step; dy++) for (let dx = 0; dx < step; dx++) {
      const p = (y * step + dy) * f.w + x * step + dx;
      for (let c = 0; c < 3; c++) acc[c] += f.dn[3 * p + c];
      if (f.sat[3 * p] || f.sat[3 * p + 1] || f.sat[3 * p + 2]) fl |= 1;
      if (flags) fl |= flags[p];
    }
    for (let c = 0; c < 3; c++) v[(y * w + x) * 3 + c] = acc[c] / (step * step) / norm;
    flag[y * w + x] = fl;
  }
  return { w, h, step, v, flag };
}

// ------------------------------------------------------------------------------------- oven
export interface SeriesInput { tC: number; frames: { id: string; texpUs: number; frame: Frame }[] }

/** Hot pixels per series (WEB_SPEC 1a.4). With dark frames, dark above median + kσ, else the two shortest exposures. */
export function seriesHot(frames: Frame[], darks: Frame[], ceil12: number, kDark = 10): Uint8Array {
  if (darks.length) {
    const d = darks[0];
    const m = new Uint8Array(d.w * d.h * 3);
    for (const dk of darks) {
      const r = hotPixelsFromDark({ width: dk.w, height: dk.h, nc: 3, data: dk.dn }, ceil12, kDark);
      for (let i = 0; i < m.length; i++) if (r.mask[i]) m[i] = 1;
    }
    return m;
  }
  // shortest, middle and longest exposure (frames sorted by t_exp), a pixel must stand out in the three
  const pickIdx = [...new Set([0, Math.floor((frames.length - 1) / 2), frames.length - 1])];
  const use = pickIdx.map((i) => frames[i]).filter(Boolean).map((f) => ({ width: f.w, height: f.h, nc: 3, data: f.dn }));
  if (!use.length) return new Uint8Array(0);
  return hotPixelsNoDark(use, 5);
}

/**
 * Oven frame for the core. Saturated pixels (dilated) are flagged and leave their channel. Hot pixels are
 * replaced by the median of their neighbours of the same channel, so they leave the plateau median without
 * counting as saturation (a flagged pixel in the plateau would drop the whole channel of that image).
 */
export function ovenFrameOf(tC: number, texpUs: number, f: Frame, hot: Uint8Array | null): OvenFrame {
  let dn: Float32Array = f.dn;
  if (hot && hot.length === f.dn.length && hot.some((v) => v)) {
    dn = f.dn.slice();
    replaceHot({ width: f.w, height: f.h, nc: 3, data: dn }, hot);
  }
  return { tC, texpUs, width: f.w, height: f.h, dn, flag: f.sat };
}

export interface SeriesAnalysis {
  tC: number; aperture: Aperture; plateau: Plateau; previewId: string; texpUs: number[];
  hotCount: number[]; zeroFrac: number[]; strayFrac: number[]; narrow: boolean;
}

/** Aperture and plateau proposal of one series on its brightest clean frame. */
export function analyseSeries(of: OvenFrame): { aperture: Aperture; plateau: Plateau } {
  const aperture = findAperture(of);
  const plateau = proposePlateau(of, aperture);
  return { aperture, plateau };
}

/** Profile and flatness of a plateau moved by hand. */
export function plateauAt(of: OvenFrame, ap: Aperture, cy: number, cx: number, radius: number) {
  const prof = plateauProfile(of, cy, cx, 1.2 * ap.r, 8);
  const rho = prof.r.map((v) => v / ap.r);
  const k = {
    R: quadFitRho2(rho, prof.I.R, prof.n), G: quadFitRho2(rho, prof.I.G, prof.n), B: quadFitRho2(rho, prof.I.B, prof.n),
    RG: quadFitRho2(rho, prof.RG, prof.n), BG: quadFitRho2(rho, prof.BG, prof.n),
  };
  const x = (radius / ap.r) ** 2;
  return { profile: prof, shape: k, flat: { R: Math.abs(k.R) * x, G: Math.abs(k.G) * x, B: Math.abs(k.B) * x, RG: Math.abs(k.RG) * x, BG: Math.abs(k.BG) * x } };
}

export interface OvenFitInput {
  signals: FrameSignal[]; band: Band; bandEnergy: Band | null; eps: number; sigmaT: number; ceil12: number;
  darkMode: DarkMode; darkLine?: Rgb<{ a: number; b: number }>; perT: 'slope' | 'band'; calibrateAt: 'highest' | 'all';
  /** ψ_j on its file grid for the IR tail calibration (tailcal.ts), null to skip it. */
  tailGrid?: TailGrid | null;
  /** Disc signals of the tail read as calibration.py (ovenread.ts). Without them the plateau signals are used. */
  tailSignals?: TailSignal[] | null;
  tailRead?: TailRead | null;
}

/** How the tail points were read, the geometry of calibration.py per temperature and the common radius (px of the sensor grid). */
export interface TailRead { how: 'calibration.py' | 'plateau'; R: number; scale: number; geo: (ReadGeometry & { tC: number; refTexpUs: number })[] }

export interface OvenResult {
  fit: OvenFit; fitEnergy: OvenFit | null;
  tails: { tC: number; R: number; G: number; B: number }[];
  ratioWorst: number;
  /** IR tail calibration of ψ_R with the oven, computed whenever the grid is given (ok false under three temperatures). */
  tail: TailCal | null;
  tailRead: TailRead | null;
}

export function runOvenFit(x: OvenFitInput): OvenResult {
  const opts = { band: x.band, eps: x.eps, sigmaT: x.sigmaT, ceilingDn: x.ceil12, darkMode: x.darkMode, darkLine: x.darkLine, perT: x.perT, calibrateAt: x.calibrateAt };
  const fit = fitOven(x.signals, opts);
  const fitEnergy = x.bandEnergy ? fitOven(x.signals, { ...opts, band: x.bandEnergy }) : null;
  const temps = [...new Set(x.signals.map((s) => s.tC))].sort((a, b) => a - b);
  const tails = temps.map((tC) => ({ tC, R: tailFraction(x.band, 'R', tC + T0C), G: tailFraction(x.band, 'G', tC + T0C), B: tailFraction(x.band, 'B', tC + T0C) }));
  let worst = 0;
  for (const r of fit.ratioTestTop) worst = Math.max(worst, Math.abs(r.dT));
  const ts = x.tailSignals?.length ? x.tailSignals : x.signals;
  const tail = x.tailGrid ? tailCalibration(tailPoints(ts, { ceilingDn: x.ceil12, darkMode: x.darkMode, darkLine: x.darkLine }), x.tailGrid, { eps: x.eps, sigmaT: x.sigmaT }) : null;
  const tailRead = x.tailGrid ? (x.tailSignals?.length && x.tailRead ? x.tailRead : { how: 'plateau' as const, R: NaN, scale: 1, geo: [] }) : null;
  return { fit, fitEnergy, tails, ratioWorst: worst, tail, tailRead };
}

/**
 * Tail signals read as calibration.py (ovenread.ts) on frames at the sensor grid. `series` gives per temperature
 * the exposure times, the max per channel of each frame (12 bit DN) and a loader of frame i at the sensor grid
 * (pylon RGB not grouped). Two passes, the reference frame of each temperature for the geometry, then every frame.
 */
export async function readTailSignals(series: { tC: number; texpUs: number[]; max12: number[][]; load: (i: number) => Promise<{ frame: Frame; info: FrameInfo }> }[], ceil12: number, strips?: Map<string, Rgb<number>>, progress?: () => void): Promise<{ signals: TailSignal[]; read: TailRead }> {
  const geo: TailRead['geo'] = [];
  let scale = 1;
  for (const s of series) {
    const k = referenceIndex(s.texpUs, s.max12, ceil12);
    const { frame, info } = await s.load(k);
    scale = info.scale;
    geo.push({ ...readGeometry(frame, info.scale), tC: s.tC, refTexpUs: s.texpUs[k] });
    progress?.();
  }
  const R = commonRadius(geo);
  const signals: TailSignal[] = [];
  for (let j = 0; j < series.length; j++) {
    const s = series[j], g = geo[j];
    for (let i = 0; i < s.texpUs.length; i++) {
      const { frame } = await s.load(i);
      const r = readSignal(frame, s.tC, s.texpUs[i], g.cy, g.cx, R, ceil12);
      signals.push({ tC: r.tC, texpUs: r.texpUs, S: r.S, nSat: r.nSat, strip: strips?.get(`${s.tC}|${s.texpUs[i]}`) });
      progress?.();
    }
  }
  return { signals, read: { how: 'calibration.py', R: R * scale, scale, geo } };
}
export { tailGrid };

export { frameSignal, fitDarkLine };

/** Median level per channel of a dark frame and its fraction of zeros. */
export function darkLevel(f: Frame): { level: Rgb<number>; zeroFrac: number } {
  const n = f.w * f.h;
  const lv: number[] = [];
  let z = 0;
  for (let c = 0; c < 3; c++) {
    const v = new Float64Array(n);
    for (let p = 0; p < n; p++) { v[p] = f.dn[3 * p + c]; if (v[p] === 0) z++; }
    lv.push(median(v));
  }
  return { level: { R: lv[0], G: lv[1], B: lv[2] }, zeroFrac: z / (3 * n) };
}

// ------------------------------------------------------------------------------------- flat field and vignette
export interface Vignette { block: number; bw: number; bh: number; R: Float32Array; G: Float32Array; B: Float32Array }

/**
 * Vignette map from a flat-field mean minus its dark (WEB_SPEC 1a.8), on blocks of 8 × 8 working pixels,
 * normalised to the block at (cx, cy) (the plateau centre), the radial fit a + bρ² + cρ⁴ outside the
 * lit zone (blocks under half the maximum).
 */
export function buildVignette(flat: Float32Array, w: number, h: number, cx: number, cy: number, block = 8): Vignette {
  const bw = Math.floor(w / block), bh = Math.floor(h / block);
  const planes: Float64Array[] = [new Float64Array(bw * bh), new Float64Array(bw * bh), new Float64Array(bw * bh)];
  for (let by = 0; by < bh; by++) for (let bx = 0; bx < bw; bx++) for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let y = 0; y < block; y++) for (let x = 0; x < block; x++) s += flat[((by * block + y) * w + bx * block + x) * 3 + c];
    planes[c][by * bw + bx] = s / (block * block);
  }
  const ix = Math.min(bw - 1, Math.max(0, Math.floor(cx / block))), iy = Math.min(bh - 1, Math.max(0, Math.floor(cy / block)));
  const out = [0, 1, 2].map((c) => {
    const P = planes[c];
    let mx = 0;
    for (const v of P) mx = Math.max(mx, v);
    const lit = new Uint8Array(P.length);
    for (let i = 0; i < P.length; i++) lit[i] = P[i] > 0.5 * mx ? 1 : 0;
    const filled = fillOutside(P, bw, bh, lit, 'radial', { cx: ix, cy: iy });
    const c0 = filled[iy * bw + ix] || 1;
    return Float32Array.from(filled, (v) => v / c0);
  });
  return { block, bw, bh, R: out[0], G: out[1], B: out[2] };
}

/** Vignette value at a working pixel of the calibration grid, bilinear on the block centres, edge values outside. */
export function vignetteAt(v: Vignette, c: number, x: number, y: number): number {
  const P = c === 0 ? v.R : c === 1 ? v.G : v.B;
  const u = Math.min(v.bw - 1, Math.max(0, (x + 0.5) / v.block - 0.5)), t = Math.min(v.bh - 1, Math.max(0, (y + 0.5) / v.block - 0.5));
  const x0 = Math.floor(u), y0 = Math.floor(t), x1 = Math.min(v.bw - 1, x0 + 1), y1 = Math.min(v.bh - 1, y0 + 1);
  const a = u - x0, b = t - y0;
  return (1 - b) * ((1 - a) * P[y0 * v.bw + x0] + a * P[y0 * v.bw + x1]) + b * ((1 - a) * P[y1 * v.bw + x0] + a * P[y1 * v.bw + x1]);
}

// ------------------------------------------------------------------------------------- sphere
export interface SphereInput {
  mean: Frame; dark: Frame | null; ceil12: number; tSphS: number; spectrum: string; valUnit: SpectrumUnit; lamUnit: 'nm' | 'um' | 'A';
  nd: { kind: 'none' } | { kind: 'od'; od: number } | { kind: 'curve'; text: string };
  fill: 'median' | 'radial'; lamNm: Float64Array; psi: Rgb<Float64Array>;
}

export function runSphere(x: SphereInput) {
  const img = { width: x.mean.w, height: x.mean.h, data: x.mean.dn };
  const disc = findSphereDisc(img);
  const dark = x.dark ? x.dark.dn : null;
  const m = sphereMask(img, dark, disc, x.mean.sat, 200 / 16);
  const sp = loadSphereSpectrum(x.spectrum, { valUnit: x.valUnit, lamUnit: x.lamUnit });
  const L = resample(x.lamNm, sp.lamNm, sp.L, 0);
  let tauNd: Float64Array;
  if (x.nd.kind === 'curve') {
    const c = loadSphereSpectrum(x.nd.text, {});
    tauNd = ndTransmission(x.lamNm, { kind: 'curve', lamNm: c.lamNm, tau: c.L });
  } else tauNd = ndTransmission(x.lamNm, x.nd.kind === 'od' ? { kind: 'od', od: x.nd.od } : { kind: 'none' });
  const tauBar = effectiveNdTransmission(x.lamNm, x.psi, L, tauNd);
  const cal = calibrationConstants(img, dark, x.tSphS, x.lamNm, x.psi, L, { tauBar, mask: m.mask, fill: x.fill, disc: { cx: disc.cx, cy: disc.cy } });
  const near = fractionNearCeiling(img, disc.mask, x.ceil12, 0.9);
  const Tlamp = fitColourTemperature(sp.lamNm, sp.L);
  const lamMax = sp.lamNm[sp.lamNm.length - 1], lamMin = sp.lamNm[0];
  const short: Ch[] = [];
  for (const c of CH) {
    let hi = 0;
    for (let k = 0; k < x.lamNm.length; k++) if (x.psi[c][k] > 1e-3 * Math.max(...x.psi[c])) hi = x.lamNm[k];
    if (lamMax < hi || lamMin > 420) short.push(c);
  }
  // vignette map from the per pixel K over its disc median, on 8 × 8 blocks
  const w = x.mean.w, h = x.mean.h;
  const kimg = new Float32Array(w * h * 3);
  for (let p = 0; p < w * h; p++) for (let c = 0; c < 3; c++) kimg[3 * p + c] = cal.K[CH[c]][p] / cal.KDiscMedian[CH[c]];
  const vig = blockMap(kimg, w, h, 8);
  return { disc: { cx: disc.cx, cy: disc.cy, rFit: disc.rFit, rUse: disc.rUse }, mask: { nDisc: m.nDisc, nKept: m.nKept, lostSaturated: m.lostSaturated, lostDark: m.lostDark, medianDn: m.medianDn }, K: cal.KDiscMedian, rsd: cal.KDiscRsd, tauBar, near, Tlamp, short, vignette: vig, band: cal.bandIntegral, spectrum: { lamNm: sp.lamNm, L: sp.L } };
}

/** Mean of 8 × 8 blocks of a 3 channel map, as a Vignette (values already relative). */
export function blockMap(img: Float32Array, w: number, h: number, block: number): Vignette {
  const bw = Math.floor(w / block), bh = Math.floor(h / block);
  const P = [new Float32Array(bw * bh), new Float32Array(bw * bh), new Float32Array(bw * bh)];
  for (let by = 0; by < bh; by++) for (let bx = 0; bx < bw; bx++) for (let c = 0; c < 3; c++) {
    let s = 0, n = 0;
    for (let y = 0; y < block; y++) for (let x = 0; x < block; x++) { const v = img[((by * block + y) * w + bx * block + x) * 3 + c]; if (Number.isFinite(v)) { s += v; n++; } }
    P[c][by * bw + bx] = n ? s / n : 1;
  }
  return { block, bw, bh, R: P[0], G: P[1], B: P[2] };
}

// ------------------------------------------------------------------------------------- calibration as used by the flame
export interface CalUse {
  K: [number, number, number];
  vignette: Vignette | null;
  /** Origin of the calibration working grid on the sensor, raw px, and its scale. */
  ox: number; oy: number; scale: number; gx0: number; gy0: number;
  hot: Int32Array | null; calW: number; calH: number;
}

/** Working pixel (x, y) of a frame to the working pixel of the calibration grid, through sensor coordinates. */
function toCal(cu: CalUse, x: number, y: number, fx0: number, fy0: number, fscale: number, fgx0: number, fgy0: number): [number, number] {
  const sx = fx0 + fgx0 + fscale * x + (fscale - 1) / 2, sy = fy0 + fgy0 + fscale * y + (fscale - 1) / 2;
  return [(sx - cu.ox - cu.gx0 - (cu.scale - 1) / 2) / cu.scale, (sy - cu.oy - cu.gy0 - (cu.scale - 1) / 2) / cu.scale];
}

// ------------------------------------------------------------------------------------- orientation
/** Quarter turns clockwise and a left-right mirror, exact, on (H, W, nc). */
export function orient<T extends Float32Array | Uint8Array>(src: T, w: number, h: number, nc: number, quarter: number, mirror: boolean): { data: T; w: number; h: number } {
  const q = ((quarter % 4) + 4) % 4;
  const W = q % 2 ? h : w, H = q % 2 ? w : h;
  const out = new (src.constructor as { new (n: number): T })(src.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const xx = mirror ? W - 1 - x : x;
    let sx: number, sy: number;
    if (q === 0) { sx = xx; sy = y; } else if (q === 1) { sx = y; sy = h - 1 - xx; } else if (q === 2) { sx = w - 1 - xx; sy = h - 1 - y; } else { sx = w - 1 - y; sy = xx; }
    for (let c = 0; c < nc; c++) out[(y * W + x) * nc + c] = src[(sy * w + sx) * nc + c];
  }
  return { data: out, w: W, h: H };
}

/** Fine rotation by deg about the centre, bilinear on values and nearest on flags (with a warning in the page). */
export function rotateFine(v: Float32Array, s: Uint8Array, w: number, h: number, deg: number): { v: Float32Array; s: Uint8Array } {
  const a = (deg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
  const cx = (w - 1) / 2, cy = (h - 1) / 2;
  const ov = new Float32Array(v.length), os = new Uint8Array(s.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = x - cx, dy = y - cy;
    const sx = ca * dx + sa * dy + cx, sy = -sa * dx + ca * dy + cy;
    const o = (y * w + x) * 3;
    if (sx < 0 || sy < 0 || sx > w - 1 || sy > h - 1) { for (let c = 0; c < 3; c++) { ov[o + c] = 0; os[o + c] = 0; } continue; }
    const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1), fx = sx - x0, fy = sy - y0;
    const ni = (Math.round(sy) * w + Math.round(sx)) * 3;
    for (let c = 0; c < 3; c++) {
      ov[o + c] = (1 - fy) * ((1 - fx) * v[(y0 * w + x0) * 3 + c] + fx * v[(y0 * w + x1) * 3 + c]) + fy * ((1 - fx) * v[(y1 * w + x0) * 3 + c] + fx * v[(y1 * w + x1) * 3 + c]);
      os[o + c] = s[ni + c];
    }
  }
  return { v: ov, s: os };
}

// ------------------------------------------------------------------------------------- flame
export interface FlameFrameInfo { id: string; axis4: number | null; asym: number; nSat: number }

/** Asymmetry of one frame on a 4 × 4 copy (as _asym_curve, ANNAbel thresholds asym_warn 0,20 and asym_bad 0,40). */
export function frameAsym(f: Frame, quarter: number, mirror: boolean): { axis: number | null; asym: number } {
  const b = 4, W = Math.floor(f.w / b), H = Math.floor(f.h / b);
  const small = new Float32Array(W * H * 3);
  for (let y = 0; y < H * b; y++) for (let x = 0; x < W * b; x++) for (let c = 0; c < 3; c++) small[(Math.floor(y / b) * W + Math.floor(x / b)) * 3 + c] += f.dn[(y * f.w + x) * 3 + c] / (b * b);
  const o = orient(small, W, H, 3, quarter, mirror);
  const hw = Math.max(2, Math.floor(o.w * 0.3));
  try {
    const r = findAxis(o.data, o.w, o.h, { halfWidth: hw });
    return { axis: r.axis === null ? null : r.axis * b, asym: r.asym };
  } catch { return { axis: null, asym: NaN }; }
}

export interface GeometryOpts {
  sensorQuarter: number; sensorMirror: boolean;
  quarter: number; mirror: boolean; fineDeg: number;
  axisManual: number | null; halfWidth: number | null; nr: number | null;
  rowTop: number | null; rowBottom: number | null; rowsPerHeight: number;
  mmPerPxSensor: number; burnerRow: number | null; zBurnerMm: number;
  cropX0: number; cropY0: number;
}

export interface Prepared {
  nz: number; nr: number; drMm: number; zMm: Float64Array; rMm: Float64Array;
  /** right[c][k] and left[c][k] (mirrored) of length nr, k = 0 the lowest height. */
  right: Float64Array[][]; left: Float64Array[][];
  satRows: number[]; axis: number; axisInt: number | null; axisCodes: string[]; axisSnr: number;
  edgeRatio: number; halvesRms: number[]; sigma: number[][];
  oriented: { w: number; h: number }; rows: [number, number];
  preview: { w: number; h: number; step: number; v: Float32Array };
  unitAbs: boolean;
}

/**
 * Stage 3 from the mean of the accepted frames minus the background, in DN (H, W, 3) of the working
 * grid, to the two halves of the projections in W m⁻² sr⁻¹ (or relative), per channel and height.
 */
export function prepareFlame(mean: Float32Array, sat: Uint8Array, w0: number, h0: number, texpUs: number, cal: CalUse | null, frameGeom: { scale: number; gx0: number; gy0: number }, g: GeometryOpts): Prepared {
  // back to sensor coordinates (files rotated by another program)
  const img = orient(mean, w0, h0, 3, g.sensorQuarter, g.sensorMirror);
  const sm = orient(sat, w0, h0, 3, g.sensorQuarter, g.sensorMirror);
  let w = img.w, h = img.h;
  // step 2, divide by t_exp and K_j(x, y) in sensor coordinates
  const rad = new Float32Array(img.data.length);
  const tS = texpUs * 1e-6;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let cx = 0, cy = 0;
    if (cal?.vignette) [cx, cy] = toCal(cal, x, y, g.cropX0, g.cropY0, frameGeom.scale, frameGeom.gx0, frameGeom.gy0);
    for (let c = 0; c < 3; c++) {
      const K = cal ? cal.K[c] * (cal.vignette ? vignetteAt(cal.vignette, c, cx, cy) : 1) : 1;
      rad[(y * w + x) * 3 + c] = img.data[(y * w + x) * 3 + c] / (tS * K);
    }
  }
  // step 3, orientation for the analysis
  const o = orient(rad, w, h, 3, g.quarter, g.mirror);
  const os = orient(sm.data, w, h, 3, g.quarter, g.mirror);
  w = o.w; h = o.h;
  let v: Float32Array = o.data, s: Uint8Array = os.data;
  if (g.fineDeg) { const r = rotateFine(v, s, w, h, g.fineDeg); v = r.v; s = r.s; }
  // axis
  const hw = g.halfWidth ?? Math.max(4, Math.floor(0.3 * w));
  const ax = findAxis(v, w, h, { halfWidth: Math.min(hw, Math.floor(w / 2) - 1) });
  const axis = g.axisManual ?? ax.refined ?? ax.subpixel ?? ax.axis ?? w / 2;
  const nrMax = Math.floor(Math.min(axis, w - axis) + 0.05);
  const nr = Math.max(4, Math.min(g.nr ?? nrMax, nrMax));
  const r0 = Math.max(0, g.rowTop ?? 0), r1 = Math.min(h - 1, g.rowBottom ?? h - 1);
  const nb = Math.max(1, Math.round(g.rowsPerHeight));
  const nzTop = Math.floor((r1 - r0 + 1) / nb);
  const drMm = g.mmPerPxSensor * frameGeom.scale;
  const burner = g.burnerRow ?? h - 1;
  const rightTB: Float64Array[][] = [[], [], []], leftTB: Float64Array[][] = [[], [], []];
  const zTB: number[] = [], satTB: boolean[] = [];
  const xlo = Math.max(0, Math.floor(axis - nr - 1)), xhi = Math.min(w, Math.ceil(axis + nr + 1));
  for (let k = 0; k < nzTop; k++) {
    const ya = r0 + k * nb;
    let anySat = false;
    for (let c = 0; c < 3; c++) {
      const R = new Float64Array(nr), Lh = new Float64Array(nr);
      const row = new Float64Array(w);
      for (let y = ya; y < ya + nb; y++) {
        for (let x = 0; x < w; x++) row[x] = v[(y * w + x) * 3 + c];
        for (let i = 0; i < nr; i++) {
          R[i] += sampleAtLocal(row, axis + i + 0.5) / nb;
          Lh[i] += sampleAtLocal(row, axis - i - 0.5) / nb;
        }
        if (!anySat) for (let x = xlo; x < xhi; x++) if (s[(y * w + x) * 3 + c]) { anySat = true; break; }
      }
      rightTB[c].push(R); leftTB[c].push(Lh);
    }
    zTB.push(g.zBurnerMm + (burner - (ya + (nb - 1) / 2)) * drMm);
    satTB.push(anySat);
  }
  // heights from the lowest up
  const order = zTB.map((_, k) => k).sort((a, b) => zTB[a] - zTB[b]);
  const right = [0, 1, 2].map((c) => order.map((k) => rightTB[c][k]));
  const left = [0, 1, 2].map((c) => order.map((k) => leftTB[c][k]));
  const zMm = Float64Array.from(order, (k) => zTB[k]);
  const satRows = order.map((k, i) => (satTB[k] ? i : -1)).filter((i) => i >= 0);
  const rMm = Float64Array.from({ length: nr }, (_, i) => (i + 0.5) * drMm);
  let peak = 0, edge = 0;
  for (let k = 0; k < zMm.length; k++) { const a = right[1][k], b = left[1][k]; peak = Math.max(peak, a[0], b[0]); edge = Math.max(edge, 0.5 * (a[nr - 1] + b[nr - 1])); }
  const halvesRms: number[] = [], sigma: number[][] = [[], [], []];
  for (let k = 0; k < zMm.length; k++) {
    let d = 0, m = 0;
    for (let i = 0; i < nr; i++) { d += (right[1][k][i] - left[1][k][i]) ** 2; m = Math.max(m, right[1][k][i], left[1][k][i]); }
    halvesRms.push(m > 0 ? Math.sqrt(d / nr) / m : NaN);
    for (let c = 0; c < 3; c++) sigma[c].push(halvesSigma(right[c][k], left[c][k]));
  }
  // preview of the oriented image, channel sum
  const step = Math.max(1, Math.ceil(w / 360));
  const pw = Math.floor(w / step), ph = Math.floor(h / step);
  const pv = new Float32Array(pw * ph);
  let pmax = 0;
  for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) {
    const q = (y * step * w + x * step) * 3;
    const val = v[q] + v[q + 1] + v[q + 2];
    pv[y * pw + x] = val; if (val > pmax) pmax = val;
  }
  for (let i = 0; i < pv.length; i++) pv[i] /= pmax || 1;
  return {
    nz: zMm.length, nr, drMm, zMm, rMm, right, left, satRows, axis, axisInt: ax.axis, axisCodes: ax.codes, axisSnr: ax.snr,
    edgeRatio: peak > 0 ? edge / peak : NaN, halvesRms, sigma, oriented: { w, h }, rows: [r0, r0 + nzTop * nb - 1],
    preview: { w: pw, h: ph, step, v: pv }, unitAbs: !!cal,
  };
}

function sampleAtLocal(row: ArrayLike<number>, x: number): number {
  const u = x - 0.5, n = row.length;
  if (u <= 0) return row[0];
  if (u >= n - 1) return row[n - 1];
  const k = Math.floor(u), t = u - k;
  return row[k] * (1 - t) + row[k + 1] * t;
}

/** Hot pixels of the calibration replaced by the median of their neighbours in a working frame of the flame. */
export function applyHot(f: Frame, cal: CalUse | null, crop: { x0: number; y0: number }, geom: { scale: number; gx0: number; gy0: number }): number {
  if (!cal?.hot || !cal.hot.length) return 0;
  const mask = new Uint8Array(f.w * f.h * 3);
  let any = 0;
  // the hot list is in calibration working pixels (index·3 + channel); map each to this frame
  for (let q = 0; q < cal.hot.length; q++) {
    const idx = cal.hot[q], c = idx % 3, p = (idx - c) / 3;
    const xc = p % cal.calW, yc = Math.floor(p / cal.calW);
    const sx = cal.ox + cal.gx0 + cal.scale * xc, sy = cal.oy + cal.gy0 + cal.scale * yc;
    const x = Math.round((sx - crop.x0 - geom.gx0) / geom.scale), y = Math.round((sy - crop.y0 - geom.gy0) / geom.scale);
    if (x < 0 || y < 0 || x >= f.w || y >= f.h) continue;
    mask[(y * f.w + x) * 3 + c] = 1; any++;
  }
  if (!any) return 0;
  return replaceHot({ width: f.w, height: f.h, nc: 3, data: f.dn }, mask);
}

// ------------------------------------------------------------------------------------- Abel
export type Sym = 'average' | 'right' | 'left' | 'both';
export interface AbelRun { method: AbelMethod | 'annabel'; sym: Sym; lambdaRel: number; segments: number; satLambda: number }
export interface AbelOut { J: Float64Array[][]; reproj: Float64Array[][]; resid: Float64Array[][]; Pused: Float64Array[][]; overReg: number; overRows?: number[]; excluded: number[]; lambdaRel: number }

export function profilesFor(p: Prepared, sym: Sym): Float64Array[][] {
  return [0, 1, 2].map((c) => p.right[c].map((R, k) => {
    const Lh = p.left[c][k];
    return sym === 'right' ? Float64Array.from(R) : sym === 'left' ? Float64Array.from(Lh) : Float64Array.from(R, (v, i) => 0.5 * (v + Lh[i]));
  }));
}

export function runAbel(p: Prepared, run: AbelRun): AbelOut {
  const drM = p.drMm * 1e-3;
  const sym = run.method === 'sat' || run.method === 'annabel' ? run.sym : run.sym === 'both' ? 'average' : run.sym;
  const P = profilesFor(p, sym === 'both' ? 'average' : sym);
  let J: Float64Array[][], reproj: Float64Array[][], resid: Float64Array[][];
  if (run.method === 'sat' && sym === 'both') {
    const n = p.nr;
    const L = chordMatrixHalf(n, drM);
    const L2: Mat = { n: 2 * n, m: n, a: new Float64Array(2 * n * n) };
    L2.a.set(L.a, 0); L2.a.set(L.a, n * n);
    const B = splineBasis(n, run.segments);
    const F = satFactor(L2, B, run.satLambda);
    const ex = new Set(p.satRows);
    J = [0, 1, 2].map((c) => p.right[c].map((R, k) => {
      if (ex.has(k)) return new Float64Array(n).fill(NaN);
      const g = new Float64Array(2 * n); g.set(R, 0); g.set(p.left[c][k], n);
      return satApply(F, g);
    }));
    reproj = J.map((ch) => ch.map((f) => (Number.isNaN(f[0]) ? new Float64Array(n).fill(NaN) : matvec(L, f))));
    resid = reproj.map((ch, c) => ch.map((q, k) => q.map((v, i) => P[c][k][i] - v)));
  } else {
    const m = run.method === 'annabel' ? 'tikhonov' : run.method;
    const lr = Math.min(LAMBDA_REL_MAX, Math.max(LAMBDA_REL_MIN, run.lambdaRel));
    const inv = invertAll(P, drM, m, { lambdaRel: lr, segments: run.segments, satLambda: run.satLambda, excludeRows: p.satRows });
    J = inv.J; reproj = inv.reproj; resid = inv.resid;
  }
  const overRows: number[] = [];
  for (let k = 0; k < p.nz; k++) {
    const sig = p.sigma[1][k] / Math.SQRT2;
    if (Number.isFinite(sig) && sig > 0 && overRegularised(resid[1][k], sig)) overRows.push(k);
  }
  return { J, reproj, resid, Pused: P, overReg: overRows.length, overRows, excluded: p.satRows, lambdaRel: run.lambdaRel };
}

/** L-curve on the given heights with the reference channel (G or the sum of the three). */
export function runLcurve(p: Prepared, rows: number[], ref: 'G' | 'sum', sym: Sym) {
  const P = profilesFor(p, sym === 'both' ? 'average' : sym);
  const use = rows.filter((k) => !p.satRows.includes(k));
  const prof = use.map((k) => ref === 'G' ? P[1][k] : Float64Array.from(P[1][k], (v, i) => v + P[0][k][i] + P[2][k][i]));
  const L = chordMatrixHalf(p.nr, p.drMm * 1e-3);
  const lambdas = logspace(-6, 3, 25);
  const r = lcurveCorner(L, prof, lambdas);
  return { lambdas, ...r };
}

// ------------------------------------------------------------------------------------- T and f_s
export interface TfsOpts {
  path: 'ratio' | 'fit3'; pair: 'RG' | 'BG' | 'BR'; emLaw: EmLaw; sigFrac: number;
  zCutMm: number | null; chemThr: number; fsChannel: FsMode; refine: boolean; weights: [number, number, number];
  /** Loose islands. A region of the signal mask (8-connected on the r, z half plane) smaller than this fraction of the largest region is masked as low signal. 0 keeps every region. Default 0.05. */
  islandFrac?: number;
  /** Mark chemiluminescence from B/G against the T of R/G. Off without a calibration, where the nominal curves mark the whole field (GB3r2-adv-05). Default true. */
  chemBG?: boolean;
}

/**
 * Channels for f_s in the ratio path (G4 r2, director). 'RG' is the default, least squares of red and green
 * at the T of the ratio, as in the presentation and the synthetic chain. 'RGB' is the advanced option with blue,
 * which needs a valid K_B (short-pass filter or sphere); blue gets the weight of the calibration and 0 in the
 * chemiluminescence zone. 'R', 'G' and 'B' are one channel alone.
 */
export type FsMode = 'RG' | 'RGB' | Ch;
export const FS_MODES: readonly FsMode[] = ['RG', 'R', 'G', 'RGB', 'B'];
export function fsMode(v: unknown): FsMode { return FS_MODES.includes(v as FsMode) ? (v as FsMode) : 'RG'; }

export interface TfsOut {
  T: Float64Array; fs: Float64Array; chi2rel: Float64Array; mask: Uint8Array; chem: Uint8Array; flags: Uint8Array;
  Tlos: Float64Array; ambiguous: number; outside: number; zCutSuggest: number | null;
  /** Loose islands removed from the signal mask, how many, how many pixels, and the criterion used. */
  islands: { n: number; px: number; frac: number; largest: number };
  /** G_j(T) of the table as well, the kernels of the RGB-ANNAbel network (stage 5) */
  table: { T: Float64Array; ratio: Float64Array; dln: Float64Array; G?: Rgb<Float64Array> };
}

/** Mask bits in TfsOut.mask, 1 used, 2 low signal, 4 chemiluminescence, 8 saturated height, 16 ratio outside the table. */
export const M_USED = 1, M_LOW = 2, M_CHEM = 4, M_SAT = 8, M_OUT = 16;

export function bandFor(lamNm: Float64Array, psi: Rgb<Float64Array>, law: EmLaw): Band { return makeBand(lamNm, psi, 'gl', law); }

/**
 * The inverse of a T ↔ ratio table built once and applied to every pixel, with the semantics of
 * invertRatio in the core (linear in ln r, NaN outside, a non-monotonic table keeps the branch that
 * holds the high-T end and gives NaN where the discarded branch also reaches the value). Calling
 * invertRatio per pixel copies the table every time, which is too slow for the uncertainty band.
 */
export function makeInverter(Tgrid: ArrayLike<number>, table: ArrayLike<number>): (r: number, high?: boolean) => number {
  const n = table.length;
  let up = true, down = true;
  for (let i = 1; i < n; i++) { if (!(table[i] > table[i - 1])) up = false; if (!(table[i] < table[i - 1])) down = false; }
  let i0 = 0, lo = Infinity, hi = -Infinity;
  if (!(up || down)) {
    const dirEnd = Math.sign(table[n - 1] - table[n - 2]);
    i0 = n - 2;
    while (i0 > 0 && Math.sign(table[i0] - table[i0 - 1]) === dirEnd) i0--;
    for (let i = 0; i <= i0; i++) { lo = Math.min(lo, table[i]); hi = Math.max(hi, table[i]); }
  }
  let lr: number[] = [], Ts: number[] = [];
  for (let i = i0; i < n; i++) { lr.push(Math.log(table[i])); Ts.push(Tgrid[i]); }
  if (lr[0] > lr[lr.length - 1]) { lr = lr.reverse(); Ts = Ts.reverse(); }
  const m = lr.length;
  return (r: number, high = false) => {
    if (!(r > 0)) return NaN;
    if (!high && r >= lo && r <= hi) return NaN;
    const x = Math.log(r);
    if (x < lr[0] || x > lr[m - 1]) return NaN;
    let a = 0, b = m - 1;
    while (b - a > 1) { const c = (a + b) >> 1; if (lr[c] <= x) a = c; else b = c; }
    const t = lr[b] === lr[a] ? 0 : (x - lr[a]) / (lr[b] - lr[a]);
    return Ts[a] + t * (Ts[b] - Ts[a]);
  };
}

/**
 * Loose islands under the signal mask (GB3 r1, director). Labels the 8-connected regions of `m` (nz × nr,
 * row-major, i = 0 on the axis) and clears every region smaller than `frac` times the largest one. The
 * half plane is enough, the mirrored half is the same field. Returns what was removed.
 */
export function dropIslands(m: Uint8Array, nz: number, nr: number, frac: number): { n: number; px: number; frac: number; largest: number } {
  const lab = new Int32Array(nz * nr).fill(-1);
  const sizes: number[] = [];
  const stack: number[] = [];
  for (let q0 = 0; q0 < nz * nr; q0++) {
    if (!m[q0] || lab[q0] >= 0) continue;
    const id = sizes.length; let n = 0;
    lab[q0] = id; stack.push(q0);
    while (stack.length) {
      const q = stack.pop()!; n++;
      const k = (q / nr) | 0, i = q - k * nr;
      for (let dk = -1; dk <= 1; dk++) for (let di = -1; di <= 1; di++) {
        const kk = k + dk, ii = i + di;
        if (kk < 0 || kk >= nz || ii < 0 || ii >= nr) continue;
        const qq = kk * nr + ii;
        if (m[qq] && lab[qq] < 0) { lab[qq] = id; stack.push(qq); }
      }
    }
    sizes.push(n);
  }
  const largest = sizes.length ? Math.max(...sizes) : 0;
  if (!(frac > 0) || sizes.length < 2) return { n: 0, px: 0, frac, largest };
  const drop = sizes.map((n) => n < frac * largest);
  let nDrop = 0, px = 0;
  drop.forEach((d, id) => { if (d) { nDrop++; px += sizes[id]; } });
  for (let q = 0; q < nz * nr; q++) if (lab[q] >= 0 && drop[lab[q]]) m[q] = 0;
  return { n: nDrop, px, frac, largest };
}

/**
 * `fixed` gives the zone and the chemiluminescence marks instead of finding them from this J, so the same path runs on a
 * fine grid with the masks of the base grid. Without it the function is what it always was.
 */
export function runTfs(J: Float64Array[][], p: Prepared, band: Band, o: TfsOpts, losP?: Float64Array[][], tableIn?: RatioTable, fixed?: { use: ArrayLike<number>; chem: ArrayLike<number> }): TfsOut {
  const nz = p.nz, nr = p.nr, N = nz * nr;
  const table: RatioTable = tableIn ?? ratioTable(band);
  const invRG = makeInverter(table.T, table.RG);
  const j = new Float64Array(N * 3);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) for (let c = 0; c < 3; c++) j[(k * nr + i) * 3 + c] = J[c][k][i];
  const finite = new Uint8Array(N);
  for (let q = 0; q < N; q++) finite[q] = Number.isFinite(j[3 * q]) && Number.isFinite(j[3 * q + 1]) && Number.isFinite(j[3 * q + 2]) ? 1 : 0;
  const jz = Float64Array.from(j, (v) => (Number.isFinite(v) ? v : 0));
  const sig = fixed ? Uint8Array.from(fixed.use) : signalMask(jz, o.sigFrac, true);
  const islands = fixed ? { n: 0, px: 0, frac: 0, largest: 0 } : dropIslands(sig, nz, nr, o.islandFrac ?? 0.05);
  const mask = new Uint8Array(N);
  const sat = new Set(p.satRows);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
    const q = k * nr + i;
    if (sat.has(k) || !finite[q]) mask[q] |= M_SAT;
    else if (!sig[q]) mask[q] |= M_LOW;
  }
  // T from R/G for the chemiluminescence test
  const Trg = new Float64Array(N).fill(NaN);
  if (!fixed) for (let q = 0; q < N; q++) if (!mask[q]) Trg[q] = invRG(j[3 * q] / j[3 * q + 1]);
  const chem = new Uint8Array(N);
  const frac = new Float64Array(nz);
  if (fixed) { for (let q = 0; q < N; q++) if (!mask[q] && fixed.chem[q]) chem[q] = 1; }
  else for (let k = 0; k < nz; k++) {
    let n = 0, m = 0;
    for (let i = 0; i < nr; i++) {
      const q = k * nr + i;
      if (mask[q]) continue;
      n++;
      const pred = Number.isFinite(Trg[q]) ? interp1Safe(Trg[q], table.T, table.BG) : NaN;
      const meas = j[3 * q + 2] / j[3 * q + 1];
      if (o.chemBG !== false && Number.isFinite(pred) && meas > pred * (1 + o.chemThr)) { chem[q] = 1; m++; }
      if (o.zCutMm !== null && p.zMm[k] < o.zCutMm) chem[q] = 1;
    }
    frac[k] = n ? m / n : 0;
  }
  let zs: number | null = null;
  for (let k = 0; k < nz; k++) { if (frac[k] > 0.5) zs = p.zMm[k]; else if (zs !== null) break; }
  const T = new Float64Array(N).fill(NaN), fs = new Float64Array(N).fill(NaN), chi2rel = new Float64Array(N).fill(NaN);
  let ambiguous = 0, outside = 0;
  const flags = new Uint8Array(N);
  if (o.path === 'ratio') {
    const tab = o.pair === 'RG' ? table.RG : o.pair === 'BG' ? table.BG : table.BR;
    const inv = o.pair === 'RG' ? invRG : makeInverter(table.T, tab);
    const [a, b] = o.pair === 'RG' ? [0, 1] : o.pair === 'BG' ? [2, 1] : [2, 0];
    for (let q = 0; q < N; q++) {
      if (mask[q] || (o.pair !== 'RG' && chem[q])) continue;
      const r = j[3 * q + a] / j[3 * q + b];
      const t = inv(r);
      if (!Number.isFinite(t)) { const tu = inv(r, true); if (Number.isFinite(tu)) ambiguous++; else outside++; mask[q] |= M_OUT; continue; }
      T[q] = t;
    }
    const mode = fsMode(o.fsChannel);
    let f: Float64Array;
    if (mode === 'RG') f = fsChannels(j, T, table.T, table.G, [1, 1, 0]);
    else if (mode === 'RGB') f = fsChannels(j, T, table.T, table.G, [o.weights[0], o.weights[1], o.weights[2]], chem);
    else {
      const ci = CH.indexOf(mode);
      const jc = Float64Array.from({ length: N }, (_, q) => j[3 * q + ci]);
      f = fsFromChannel(jc, T, table.T, table.G[mode]);
    }
    for (let q = 0; q < N; q++) fs[q] = f[q];
  } else {
    const wAll = o.weights, wNoB: [number, number, number] = [o.weights[0], o.weights[1], 0];
    const mA = new Uint8Array(N), mB = new Uint8Array(N);
    for (let q = 0; q < N; q++) if (!mask[q]) { if (chem[q] || wAll[2] === 0) mB[q] = 1; else mA[q] = 1; }
    for (const [mm, w] of [[mA, wAll], [mB, wNoB]] as [Uint8Array, [number, number, number]][]) {
      if (!mm.some((x) => x)) continue;
      let res = solveTfs(jz, table.G, table.T, w, mm);
      if (o.refine) res = refineTfs(jz, table.G, table.T, res, w);
      const mk = maskTfs(res, table.T);
      for (let q = 0; q < N; q++) if (mm[q]) {
        T[q] = mk.res.T[q]; fs[q] = mk.res.fs[q]; flags[q] = mk.flags[q];
        const A = w[0] * jz[3 * q] ** 2 + w[1] * jz[3 * q + 1] ** 2 + w[2] * jz[3 * q + 2] ** 2;
        chi2rel[q] = A > 0 ? mk.res.chi2[q] / A : NaN;
      }
    }
  }
  for (let q = 0; q < N; q++) {
    if (chem[q]) mask[q] |= M_CHEM;
    if (Number.isFinite(T[q])) mask[q] |= M_USED;
  }
  // line of sight comparison, ratio of the projections themselves (rotulada "promedio en la línea de visión")
  const Tlos = new Float64Array(N).fill(NaN);
  if (losP) {
    for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
      const q = k * nr + i;
      if (mask[q] & (M_SAT | M_LOW)) continue;
      Tlos[q] = invRG(losP[0][k][i] / losP[1][k][i]);
    }
  }
  const tabR = o.pair === 'BG' ? table.BG : o.pair === 'BR' ? table.BR : table.RG;
  const dln = new Float64Array(table.T.length);
  for (let i = 0; i < dln.length; i++) {
    const a = Math.max(0, i - 1), b = Math.min(dln.length - 1, i + 1);
    dln[i] = (Math.log(tabR[b]) - Math.log(tabR[a])) / (table.T[b] - table.T[a]);
  }
  return { T, fs, chi2rel, mask, chem, flags, Tlos, ambiguous, outside, zCutSuggest: zs, islands, table: { T: table.T, ratio: tabR, dln, G: table.G } };
}

// ------------------------------------------------------------------------------------- uncertainty band
export interface BandOpts {
  n: number; seed: number; sigmaT: number; Tcal: number; eps: number; run: AbelRun; tfs: TfsOpts; absolute: boolean;
  /** Relative statistical σ of K_R, K_G, K_B, one independent factor per channel and repetition (0 when unknown). */
  sKrel?: [number, number, number];
  /** Relative σ of the scale in mm per pixel. f_s goes as 1/scale, T does not move. */
  scaleRel?: number;
}

/**
 * Band "ruido y calibración" (WEB_SPEC etapa 6). n repetitions, each with Gaussian noise on the
 * projections from the difference between halves (σ of the mean of the halves = σ_ray/√2) and the
 * oven error as one ΔT common to the three K_j (K_j' = K_j B_j(T_cal)/B_j(T_cal + ΔT)), the statistical
 * σ of each K_j as an independent factor per channel, and the scale uncertainty as a factor on f_s
 * (GB3r1-fisico-03 and 04). Returns the 16th and 84th percentiles of T and f_s per pixel.
 */
export function runBand(p: Prepared, band: Band, o: BandOpts) {
  const g = rng(o.seed);
  const N = p.nz * p.nr;
  const Ts: Float64Array[] = [], Fs: Float64Array[] = [];
  const base = profilesFor(p, 'average');
  const table = ratioTable(band);
  for (let r = 0; r < o.n; r++) {
    const dT = Number.isFinite(o.Tcal) ? g.normal() * o.sigmaT : 0;
    const sk = o.sKrel ?? [0, 0, 0];
    const fac = CH.map((c, k) => (Number.isFinite(o.Tcal) ? bandX(band, c, o.Tcal + dT, o.eps) / bandX(band, c, o.Tcal, o.eps) : 1) * (sk[k] > 0 ? 1 / (1 + sk[k] * g.normal()) : 1));
    const fsScale = o.scaleRel && o.scaleRel > 0 ? 1 / (1 + o.scaleRel * g.normal()) : 1;
    const q: Prepared = { ...p, right: [], left: [] };
    for (let c = 0; c < 3; c++) {
      const rr: Float64Array[] = [], ll: Float64Array[] = [];
      for (let k = 0; k < p.nz; k++) {
        const s = p.sigma[c][k] / Math.SQRT2;
        const row = Float64Array.from(base[c][k], (v) => (v + (Number.isFinite(s) ? s * g.normal() : 0)) * fac[c]);
        rr.push(row); ll.push(row);
      }
      q.right.push(rr); q.left.push(ll);
    }
    // the band is always on the mean of the halves. SAT with both halves in the fit gives that same T, so SAT keeps its own band. The network for the emission cannot be trained 30 times.
    const run: AbelRun = { ...o.run, method: o.run.method === 'annabel' ? 'tikhonov' : o.run.method, sym: 'average' };
    const ab = runAbel(q, run);
    const t = runTfs(ab.J, q, band, { ...o.tfs, refine: false }, undefined, table);
    Ts.push(t.T); Fs.push(fsScale === 1 ? t.fs : t.fs.map((v) => v * fsScale));
  }
  const pct = (arr: Float64Array[], qq: number) => {
    const out = new Float64Array(N).fill(NaN);
    const v: number[] = [];
    for (let i = 0; i < N; i++) {
      v.length = 0;
      for (const a of arr) if (Number.isFinite(a[i])) v.push(a[i]);
      if (v.length >= Math.max(3, arr.length / 2)) out[i] = percentile(v, qq);
    }
    return out;
  };
  return { Tlo: pct(Ts, 16), Thi: pct(Ts, 84), fslo: pct(Fs, 16), fshi: pct(Fs, 84) };
}

// ------------------------------------------------------------------------------------- more points with SAT or the networks
/** What stage 1 needs to be evaluated again, kept from training. */
export interface EmissionNet { params: Float32Array; norm: CoordNorm; outScale: number; bScale: number }
/** What stage 2 needs to be evaluated again, kept from training. */
export interface TfsNet { params: Float32Array; norm: CoordNorm; sizes: number[]; Tlo: number; Thi: number; fsRef: number }
export type RefineKind = 'sat' | 'annabel' | 'net';
export interface RefineIn {
  kind: RefineKind; factor: number; run: AbelRun; tfs: TfsOpts;
  /** Results of the base grid, read and never written. */
  base: { T: Float64Array; fs: Float64Array; mask: Uint8Array; chem: Uint8Array; J: Float64Array[][] };
  emission?: EmissionNet | null; net?: TfsNet | null;
}
export interface RefineCheck { nBase: number; dTK: number; dFsRel: number; dJRel: number; mismatch: number }
export interface RefineOut {
  kind: RefineKind; factor: number; fr: number; fz: number; nr: number; nz: number; rMm: Float64Array; zMm: Float64Array; axisNode: boolean; inner: number;
  /** nz × nr, NaN outside the zone of the base grid. fs is the volume fraction as in the base results. */
  T: Float64Array; fs: Float64Array;
  /** Interleaved (nz × nr × 3) when this view has the emission on its own grid, else null. */
  J: Float64Array | null;
  /** Bits of TfsOut.mask per fine point. */
  mask: Uint8Array;
  /** 1 on the points of the base grid. */
  onBase: Uint8Array;
  check: RefineCheck; ms: number;
}

/** J (r, z) of the stage 1 network on the fine grid, each channel as nz rows of nr, the arithmetic of finishStage1. */
function emissionOnFine(e: EmissionNet, g: FineGrid, progress?: (d: number, t: number) => void): Float64Array[][] {
  const X = coordFeatures(e.norm, g.rMm, g.zMm, [0]);
  const rowsPerBlock = 16, net = new Net(rowsPerBlock * g.nr);
  const J: Float64Array[][] = [[], [], []];
  const f = Math.fround;
  for (let c = 0; c < 3; c++) for (let l = 0; l < g.nz; l++) J[c].push(new Float64Array(g.nr));
  for (let r0 = 0; r0 < g.nz; r0 += rowsPerBlock) {
    const r1 = Math.min(g.nz, r0 + rowsPerBlock);
    net.forward(e.params, X, r0 * g.nr, (r1 - r0) * g.nr, e.outScale);
    for (let l = r0; l < r1; l++) for (let s = 0; s < g.nr; s++) for (let c = 0; c < 3; c++) {
      J[c][l][s] = f(f(net.Jn[((l - r0) * g.nr + s) * 3 + c] * e.bScale) * PER_MM_TO_PER_M);
    }
    progress?.(r1, g.nz);
  }
  return J;
}

/** J of SAT on the fine radii, with the coefficients of the same fit as runAbel, one per height. */
function satOnFine(p: Prepared, run: AbelRun, g: FineGrid, progress?: (d: number, t: number) => void): Float64Array[][] {
  const n = p.nr, drM = p.drMm * 1e-3;
  const L = chordMatrixHalf(n, drM);
  const B = splineBasis(n, run.segments);
  const both = run.sym === 'both';
  let A: Mat = L;
  if (both) { A = { n: 2 * n, m: n, a: new Float64Array(2 * n * n) }; A.a.set(L.a, 0); A.a.set(L.a, n * n); }
  const F = satFactor(A, B, run.satLambda);
  const Bf = splineBasisAt(splineX(g, n), run.segments);
  const ex = new Set(p.satRows);
  const P = both ? null : profilesFor(p, run.sym);
  const J: Float64Array[][] = [[], [], []];
  for (let c = 0; c < 3; c++) for (let k = 0; k < p.nz; k++) {
    if (ex.has(k)) { J[c].push(new Float64Array(g.nr).fill(NaN)); continue; }
    let rhs: ArrayLike<number>;
    if (both) { const v = new Float64Array(2 * n); v.set(p.right[c][k], 0); v.set(p.left[c][k], n); rhs = v; } else rhs = P![c][k];
    J[c].push(satEval(Bf, satCoef(F, rhs)));
    if (k % 32 === 0) progress?.(c * p.nz + k, 3 * p.nz);
  }
  return J;
}

/**
 * The same function on more points, in the Worker. SAT is one fit per height, so it refines the radius. The networks are
 * functions of (r, z), so they refine both. T and f_s come from the network of T and f_s, or else from the same direct path
 * as the base results on the fine emission, with the masks of the base grid. Base results are not touched and no fit runs again.
 */
export function runRefine(p: Prepared, band: Band, x: RefineIn, progress?: (d: number, t: number) => void): RefineOut {
  const t0 = typeof performance !== 'undefined' ? performance.now() : 0;
  const factor = refineFactor(x.factor);
  const fz = x.kind === 'sat' ? 1 : factor;
  const g = fineGrid(p.rMm, p.zMm, factor, fz);
  const N = g.nz * g.nr, baseN = p.nz * p.nr;
  const { base } = x;
  const norm = x.kind === 'net' ? x.net?.norm : x.kind === 'annabel' ? x.emission?.norm : null;
  if (x.kind === 'net' && !x.net) throw new Error('no_emission');
  if (x.kind === 'annabel' && !x.emission) throw new Error('no_emission');
  // the shown zone and the chemiluminescence marks come from the base cells nearest to each fine point
  const use = new Uint8Array(N), chem = new Uint8Array(N), near = new Uint8Array(N);
  forNearest(g, p.nr, (q, cells) => {
    let ok = 1, ch = 0, bits = 0;
    for (const c of cells) { if (!Number.isFinite(base.T[c])) ok = 0; if (base.chem[c]) ch = 1; bits |= base.mask[c]; }
    // saturated rows at an end leave the training range of the network, and it is not evaluated past that range
    if (norm) { const l = Math.floor(q / g.nr), z = g.zMm[l]; if (g.onZ[l] < 0 && (z < norm.z_min || z > norm.z_max)) ok = 0; }
    use[q] = ok; chem[q] = ch; near[q] = bits;
  });
  let T: Float64Array, fs: Float64Array, Jr: Float64Array[][] | null = null;
  if (x.kind === 'net') {
    const X = coordFeatures(x.net!.norm, g.rMm, g.zMm, [0]);
    T = new Float64Array(N).fill(NaN); fs = new Float64Array(N).fill(NaN);
    const rows = 64;
    for (let r0 = 0; r0 < g.nz; r0 += rows) {
      const r1 = Math.min(g.nz, r0 + rows);
      const r = predictTfs(x.net!.params, x.net!, X.subarray(r0 * g.nr * 3, r1 * g.nr * 3));
      for (let q = 0; q < r.T.length; q++) { const o = r0 * g.nr + q; if (use[o]) { T[o] = r.T[q]; fs[o] = r.fs[q]; } }
      progress?.(r1, g.nz);
    }
    if (x.emission) Jr = emissionOnFine(x.emission, g);
  } else {
    Jr = x.kind === 'sat' ? satOnFine(p, x.run, g, progress) : emissionOnFine(x.emission!, g, progress);
    const gridP = { ...p, nz: g.nz, nr: g.nr, zMm: g.zMm, rMm: g.rMm, satRows: [] as number[] };
    const r = runTfs(Jr, gridP, band, x.tfs, undefined, undefined, { use, chem });
    T = r.T; fs = r.fs;
    for (let q = 0; q < N; q++) if (!use[q]) { T[q] = NaN; fs[q] = NaN; }
  }
  let J: Float64Array | null = null;
  if (Jr) {
    J = new Float64Array(N * 3);
    for (let l = 0; l < g.nz; l++) for (let s = 0; s < g.nr; s++) for (let c = 0; c < 3; c++) J[(l * g.nr + s) * 3 + c] = Jr[c][l][s];
  }
  const mask = new Uint8Array(N);
  for (let q = 0; q < N; q++) {
    if (Number.isFinite(T[q])) mask[q] = M_USED | (chem[q] ? M_CHEM : 0);
    else if (use[q]) mask[q] = M_OUT;
    else mask[q] = near[q] & ~M_USED || M_LOW;
  }
  // the points shared with the base grid must give the base values
  const onBase = new Uint8Array(N);
  const idx = basePointsOnFine(g, p.nz, p.nr);
  const check: RefineCheck = { nBase: baseN, dTK: 0, dFsRel: 0, dJRel: 0, mismatch: 0 };
  const cmp = (a: number, b: number) => (Number.isFinite(a) !== Number.isFinite(b) ? -1 : Number.isFinite(a) ? Math.abs(a - b) : 0);
  for (let q = 0; q < baseN; q++) {
    const o = idx[q];
    onBase[o] = 1;
    mask[o] = base.mask[q];
    const dT = cmp(T[o], base.T[q]), dF = cmp(fs[o], base.fs[q]);
    if (dT < 0 || dF < 0) { check.mismatch++; continue; }
    check.dTK = Math.max(check.dTK, dT);
    if (dF > 0) check.dFsRel = Math.max(check.dFsRel, dF / Math.abs(base.fs[q]));
    if (J) for (let c = 0; c < 3; c++) {
      const k = Math.floor(q / p.nr), i = q - k * p.nr;
      const dJ = cmp(J[o * 3 + c], base.J[c][k][i]);
      if (dJ > 0) check.dJRel = Math.max(check.dJRel, dJ / Math.abs(base.J[c][k][i]));
    }
  }
  return { kind: x.kind, factor, fr: g.fr, fz: g.fz, nr: g.nr, nz: g.nz, rMm: g.rMm, zMm: g.zMm, axisNode: g.axisNode, inner: g.inner, T, fs, J, mask, onBase, check, ms: (typeof performance !== 'undefined' ? performance.now() : 0) - t0 };
}

// ------------------------------------------------------------------------------------- calibration files
/** G_j(T) recomputed from the ψ_j of a file and compared with its G_tab (WEB_SPEC 1c). */
export function checkGtab(lamNm: ArrayLike<number>, psi: Rgb<ArrayLike<number>>, Tgrid: ArrayLike<number>, Gtab: Rgb<ArrayLike<number>>, law: EmLaw): number {
  const G = kernelTableTrapz(lamNm, psi, Tgrid, law);
  let worst = 0;
  for (const c of CH) for (let i = 0; i < Tgrid.length; i++) worst = Math.max(worst, Math.abs(G[c][i] / Gtab[c][i] - 1));
  return worst;
}

export { bandX, T0C };
