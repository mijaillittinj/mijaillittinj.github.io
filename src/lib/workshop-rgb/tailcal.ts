/**
 * IR tail calibration of ψ_R with the oven (decision of the professor after G5). One parameter s_R scales
 * ψ_R above 700 nm (s = 1 is the nominal curve), ψ_G and ψ_B stay nominal. It is fitted on the R/G ratio of
 * every oven image with R and G between 5 and 90 % of the scale,
 *
 *   ln(y_R/y_G) = ln c + ln H_R(T; s) − ln H_G(T),   H_j = ∫ ψ_j I_bλ dλ,   c = K_R/K_G,
 *
 * with weight 1/√n_T so that every temperature counts the same. Then K_j per temperature with the corrected
 * ψ_R, the drift of K_R/K_G, one temperature left out at a time (the acceptance criterion, K_R/K_G measured
 * at the temperature left out against the fit of the others, under 1 %), and the ratio test carried to a
 * flame at 1800 K before and after the correction.
 *
 * The same algorithm as tail_calibration in manim/tools/calibration.py, with the same integration rule
 * (trapezoid on the 1 nm grid of the response file, λ ≥ 700 nm is the tail) and the same selection of
 * points (k_fit). tests/workshop-rgb-tailcal.test.ts compares both on the points of oven_fit_points.csv
 * (tools/make_web_fixtures.py, tailcal_ref.json) at 1e-6. The uncertainty differs on purpose. calibration.py
 * draws a bootstrap and a thermocouple Monte Carlo, here it is deterministic and linear (covariance of the
 * fit, ±σ_T at one temperature at a time, +σ_T common to all, in quadrature), written by its own numpy code
 * in the same fixture.
 *
 * Units. λ in m inside (lamNm in nm), T in K inside and tC in °C, y in DN s⁻¹.
 */
import { planck, emValue, EM_GROUP, CH, type Ch, type Rgb, type EmLaw } from './radiometry.ts';
import { linfit } from './stats.ts';
import type { FrameSignal } from './oven.ts';

export const T0C = 273.15;
export const TAIL_CUT_NM = 700;
export const T_FLAME = 1800;
/** Acceptance criterion, |K_R/K_G at the temperature left out over the fit of the others − 1| below this. */
export const TAIL_CRITERION = 0.01;

export interface TailPoint { tC: number; texpUs: number; ch: Ch; y: number; inLin: boolean; inFit: boolean; saturated: boolean; dark: boolean }

/** ψ_j on its own grid with the soot factor, for trapezoid integrals as calibration.py. */
export interface TailGrid { lamNm: Float64Array; lam: Float64Array; psi: Rgb<Float64Array>; soot: Float64Array; hi: Uint8Array }

export function tailGrid(lamNm: ArrayLike<number>, psi: Rgb<ArrayLike<number>>, law: EmLaw = EM_GROUP, cutNm = TAIL_CUT_NM): TailGrid {
  const n = lamNm.length;
  const lam = new Float64Array(n), soot = new Float64Array(n), hi = new Uint8Array(n);
  const cut = cutNm * 1e-9;
  for (let i = 0; i < n; i++) {
    lam[i] = lamNm[i] * 1e-9;
    soot[i] = (6 * Math.PI * emValue(law, lamNm[i])) / lam[i];
    hi[i] = lam[i] >= cut ? 1 : 0;
  }
  return { lamNm: Float64Array.from(lamNm), lam, psi: { R: Float64Array.from(psi.R), G: Float64Array.from(psi.G), B: Float64Array.from(psi.B) }, soot, hi };
}

/** ∫ f dλ by trapezoid with the integrand masked to below (part 'lo') or above (part 'hi') the cut, as np.where then trapezoid. */
function trap(g: TailGrid, f: (i: number) => number, part: 'all' | 'lo' | 'hi' = 'all'): number {
  let s = 0, prev = NaN;
  for (let i = 0; i < g.lam.length; i++) {
    const v = part === 'all' ? f(i) : (part === 'hi') === (g.hi[i] === 1) ? f(i) : 0;
    if (i > 0) s += 0.5 * (prev + v) * (g.lam[i] - g.lam[i - 1]);
    prev = v;
  }
  return s;
}

/** ψ_c with the tail above the cut scaled by s (s applies to R only). */
const psiAt = (g: TailGrid, c: Ch, s: number, i: number) => (c === 'R' && g.hi[i] ? s : 1) * g.psi[c][i];

/** ε ∫ ψ_c I_bλ dλ with the tail of R scaled by s. */
export function tailBand(g: TailGrid, c: Ch, TK: number, s = 1, eps = 1): number {
  return eps * trap(g, (i) => psiAt(g, c, s, i) * planck(g.lam[i], TK));
}

/** G_c(T) = ∫ ψ_c (6π E/λ) I_bλ dλ with the tail of R scaled by s. */
export function tailG(g: TailGrid, c: Ch, TK: number, s = 1): number {
  return trap(g, (i) => psiAt(g, c, s, i) * g.soot[i] * planck(g.lam[i], TK));
}

/**
 * Points of the tail fit from the disc signals, as calibration.py. The tool reads them with ovenread.ts (the
 * reader of calibration.py: mean over a disc common to every temperature, saturation per channel), not with
 * the plateau median of 1a.3, so the tail figures on the real oven are those of the talk (parity after G5).
 * D_j(T) the intercept of S against t_exp over the frames between 5 and 90 % of the scale (every unsaturated
 * frame if fewer than 3), or the strip or the dark files when chosen. y = (S − D)/t_exp.
 */
export type TailSignal = Pick<FrameSignal, 'tC' | 'texpUs' | 'S' | 'nSat'> & { strip?: FrameSignal['strip'] };

export function tailPoints(signals: TailSignal[], opts: { ceilingDn?: number; darkMode?: 'intercept' | 'strip' | 'files'; darkLine?: Rgb<{ a: number; b: number }>; minTexpUs?: number } = {}): TailPoint[] {
  const ceil = opts.ceilingDn ?? 4095, mode = opts.darkMode ?? 'intercept', minT = opts.minTexpUs ?? 100;
  const temps = [...new Set(signals.map((s) => s.tC))].sort((a, b) => a - b);
  const out: TailPoint[] = [];
  for (const tC of temps) for (const c of CH) {
    const all = signals.filter((s) => s.tC === tC && s.texpUs >= minT && Number.isFinite(s.S[c]));
    const frac = (s: TailSignal) => s.S[c] / ceil;
    const unsat = all.filter((s) => s.nSat[c] === 0);
    let icpt = 0;
    if (mode === 'intercept') {
      let li = unsat.filter((s) => frac(s) >= 0.05 && frac(s) <= 0.9);
      if (li.length < 3) li = unsat;
      icpt = li.length >= 2 ? linfit(li.map((s) => s.texpUs * 1e-6), li.map((s) => s.S[c])).a : 0;
    }
    for (const s of all) {
      const D = mode === 'intercept' ? icpt : mode === 'strip' ? (s.strip ? s.strip[c] : NaN) : opts.darkLine ? opts.darkLine[c].a + opts.darkLine[c].b * s.texpUs * 1e-6 : 0;
      const sat = s.nSat[c] > 0, f = frac(s);
      out.push({ tC, texpUs: s.texpUs, ch: c, y: (s.S[c] - D) / (s.texpUs * 1e-6), saturated: sat, dark: f < 0.05,
        inLin: !sat && f >= 0.05 && f <= 0.9, inFit: !sat && f >= 0.5 && f <= 0.7 });
    }
  }
  return out;
}

export interface KPerT { K: number; n: number; band: 'inFit' | 'inLin' | 'noSat'; x: number }
export interface KFit { K: number; spreadRel: number; perT: Record<number, KPerT> }

/** K_j per temperature (mean of y over the band, over ε ∫ ψ_j I_bλ dλ) and their geometric mean, as k_fit of calibration.py. */
export function kFit(points: TailPoint[], g: TailGrid, c: Ch, temps: number[], band: 'inFit' | 'inLin', s = 1, eps = 1, dT = 0): KFit {
  const perT: Record<number, KPerT> = {};
  for (const tC of temps) {
    const mine = points.filter((p) => p.tC === tC && p.ch === c);
    let pts = mine.filter((p) => p[band]), used: KPerT['band'] = band;
    if (!pts.length) { pts = mine.filter((p) => p.inLin); used = 'inLin'; }
    if (!pts.length) { pts = mine.filter((p) => !p.saturated && !p.dark); used = 'noSat'; }
    if (!pts.length) continue;
    const x = tailBand(g, c, tC + T0C + dT, c === 'R' ? s : 1, eps);
    const m = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    perT[tC] = { K: m / x, n: pts.length, band: used, x };
  }
  const lk = Object.values(perT).map((v) => Math.log(v.K));
  const mu = lk.reduce((a, b) => a + b, 0) / lk.length;
  const sd = lk.length > 1 ? Math.sqrt(lk.reduce((a, b) => a + (b - mu) ** 2, 0) / (lk.length - 1)) : 0;
  return { K: Math.exp(mu), spreadRel: sd, perT };
}

/** Root of f on [a, b] to xtol, bisection then secant safeguarded (brentq of scipy gives the same root to xtol). */
function root(f: (x: number) => number, a: number, b: number, xtol = 1e-10): number {
  let fa = f(a), fb = f(b);
  if (!(fa * fb <= 0)) return NaN;
  for (let k = 0; k < 300 && b - a > xtol; k++) {
    let m = b - (fb * (b - a)) / (fb - fa);
    if (!(m > a + 0.1 * (b - a) && m < b - 0.1 * (b - a))) m = 0.5 * (a + b);
    const fm = f(m);
    if (fm === 0) return m;
    if (fa * fm < 0) { b = m; fb = fm; } else { a = m; fa = fm; }
  }
  return 0.5 * (a + b);
}

export interface TailLoo {
  tC: number; s: number; KRKG: number;
  /** R/G of the temperature left out against the prediction of the fit of the others, %, corrected and nominal. */
  residPct: number; residNominalPct: number;
  /** K_R/K_G measured at the temperature left out (with ψ_R of the other three) over the fit of the others − 1, %. */
  KRKGrelPct: number; KRKGrelNominalPct: number; KRKGout: number;
  /** T at 1800 K with the fit of the others, and with K_R/K_G of the temperature left out (the ratio test after), K. */
  dTflameK: number; dToutK: number;
}

export interface TailRatioRow { tC: number; KRKGrelTop: number; KRKGrelTopNominal: number; dTflameK: number; dTflameNominalK: number }

export interface TailCal {
  temps: number[]; nPairs: number; nPerT: Record<number, number>; ok: boolean; why?: string;
  s: number; lnc: number; KRKG: number; rmsPct: number;
  u: { s: number; sStat: number; sT: number; sCommon: number; lnc: number; KRKG: number; flameK: number; dTdsK: number; dTdlncK: number };
  /** K_j per temperature with the corrected ψ_R, band 5 to 90 % (inLin), and with the nominal ψ. */
  Kcorr: Rgb<KFit>; Knom: Rgb<KFit>;
  /** K_j geometric mean over the temperatures, band 50 to 70 % (inFit), corrected ψ_R. */
  Kfit: Rgb<KFit>;
  KRKGperT: Record<number, number>; KRKGperTNominal: Record<number, number>;
  driftPct: number; driftNominalPct: number; spreadPct: number; driftNominalByChannelPct: Rgb<number>;
  loo: TailLoo[];
  ratio: TailRatioRow[];
  /** Acceptance criterion. With four or more temperatures the worst over every temperature left out, with three only the coldest. */
  criterion: { worstPct: number; nominalPct: number; from: number[]; pass: boolean; strongT: number; strongPct: number; limitPct: number };
  /** Ratio test carried to the flame, worst |ΔT| out of sample after, and with the nominal ψ before (K_R/K_G of the highest T). */
  ratioTest: { afterK: number; beforeK: number };
  tail700FlamePct: number;
}

/**
 * The tail calibration. Needs three or more temperatures with R/G pairs in 5 to 90 % (`ok` false otherwise,
 * with `why`). `sigmaT` is the thermocouple uncertainty in K for the deterministic uncertainty.
 */
export function tailCalibration(points: TailPoint[], g: TailGrid, opts: { eps?: number; sigmaT?: number; Tflame?: number } = {}): TailCal {
  const eps = opts.eps ?? 1, sT = opts.sigmaT ?? 5, TF = opts.Tflame ?? T_FLAME;
  const key = (p: TailPoint) => `${p.tC}|${p.texpUs}`;
  const R = new Map<string, TailPoint>(), G = new Map<string, TailPoint>();
  for (const p of points) { if (p.ch === 'R') R.set(key(p), p); else if (p.ch === 'G') G.set(key(p), p); }
  const pairs: { tC: number; texpUs: number; lr: number }[] = [];
  for (const [k, r] of R) { const q = G.get(k); if (r.inLin && q && q.inLin) pairs.push({ tC: r.tC, texpUs: r.texpUs, lr: Math.log(r.y / q.y) }); }
  pairs.sort((a, b) => a.tC - b.tC || a.texpUs - b.texpUs);
  const temps = [...new Set(pairs.map((p) => p.tC))].sort((a, b) => a - b);
  const nPerT: Record<number, number> = {};
  for (const p of pairs) nPerT[p.tC] = (nPerT[p.tC] ?? 0) + 1;

  // band pieces of R and G below and above the cut, cached per T
  const cache = new Map<string, [number, number, number, number]>();
  const AB = (TK: number) => {
    const k = TK.toFixed(6);
    let v = cache.get(k);
    if (!v) {
      const wR = (i: number) => g.psi.R[i] * planck(g.lam[i], TK), wG = (i: number) => g.psi.G[i] * planck(g.lam[i], TK);
      v = [trap(g, wR, 'lo'), trap(g, wR, 'hi'), trap(g, wG, 'lo'), trap(g, wG, 'hi')];
      cache.set(k, v);
    }
    return v;
  };
  const lnH = (TK: number, s: number) => { const [aR, bR, aG, bG] = AB(TK); return Math.log(Math.max(aR + s * bR, 1e-300) / Math.max(aG + bG, 1e-300)); };

  /** Weighted least squares of (ln c, s), Gauss-Newton with step halving. fixS fits ln c alone. */
  const fit = (P: typeof pairs, off: Record<number, number> = {}, fixS?: number): [number, number] => {
    const n: Record<number, number> = {};
    for (const p of P) n[p.tC] = (n[p.tC] ?? 0) + 1;
    const TK = P.map((p) => p.tC + T0C + (off[p.tC] ?? 0)), w = P.map((p) => 1 / Math.sqrt(n[p.tC]));
    const resid = (lc: number, s: number) => P.map((p, i) => (p.lr - lc - lnH(TK[i], s)) * w[i]);
    const cost = (r: number[]) => r.reduce((a, b) => a + b * b, 0);
    if (fixS !== undefined) {
      let num = 0, den = 0;
      P.forEach((p, i) => { num += w[i] * w[i] * (p.lr - lnH(TK[i], fixS)); den += w[i] * w[i]; });
      return [num / den, fixS];
    }
    let lc = 0.04, s = 0.12, r = resid(lc, s), c0 = cost(r);
    for (let it = 0; it < 200; it++) {
      let a11 = 0, a12 = 0, a22 = 0, g1 = 0, g2 = 0;
      P.forEach((_, i) => {
        const [aR, bR] = AB(TK[i]);
        const j1 = -w[i], j2 = (-w[i] * bR) / (aR + s * bR);
        a11 += j1 * j1; a12 += j1 * j2; a22 += j2 * j2; g1 += j1 * r[i]; g2 += j2 * r[i];
      });
      const det = a11 * a22 - a12 * a12;
      if (!(det > 0)) break;
      let d1 = -(a22 * g1 - a12 * g2) / det, d2 = -(a11 * g2 - a12 * g1) / det;
      let t = 1, done = false;
      for (let h = 0; h < 40; h++) {
        const r2 = resid(lc + t * d1, s + t * d2), c2 = cost(r2);
        if (c2 <= c0) { lc += t * d1; s += t * d2; r = r2; done = Math.abs(t * d2) < 1e-15 * (1 + Math.abs(s)) && Math.abs(t * d1) < 1e-15 * (1 + Math.abs(lc)); c0 = c2; break; }
        t /= 2;
        if (h === 39) { done = true; d1 = 0; d2 = 0; }
      }
      if (done) break;
    }
    return [lc, s];
  };

  const empty: Rgb<KFit> = { R: { K: NaN, spreadRel: NaN, perT: {} }, G: { K: NaN, spreadRel: NaN, perT: {} }, B: { K: NaN, spreadRel: NaN, perT: {} } };
  const nan = NaN;
  if (temps.length < 3) {
    return { temps, nPairs: pairs.length, nPerT, ok: false, why: 'tail_few_temps', s: nan, lnc: nan, KRKG: nan, rmsPct: nan,
      u: { s: nan, sStat: nan, sT: nan, sCommon: nan, lnc: nan, KRKG: nan, flameK: nan, dTdsK: nan, dTdlncK: nan },
      Kcorr: empty, Knom: empty, Kfit: empty, KRKGperT: {}, KRKGperTNominal: {}, driftPct: nan, driftNominalPct: nan, spreadPct: nan,
      driftNominalByChannelPct: { R: nan, G: nan, B: nan }, loo: [], ratio: [],
      criterion: { worstPct: nan, nominalPct: nan, from: [], pass: false, strongT: nan, strongPct: nan, limitPct: 100 * TAIL_CRITERION },
      ratioTest: { afterK: nan, beforeK: nan }, tail700FlamePct: nan };
  }
  const [lc0, s0] = fit(pairs);
  const rms = Math.sqrt(pairs.reduce((a, p) => a + (p.lr - lc0 - lnH(p.tC + T0C, s0)) ** 2, 0) / pairs.length);

  // deterministic uncertainty: covariance of the fit, ±σ_T at one temperature at a time, +σ_T common
  const nT = (tC: number) => nPerT[tC];
  let a11 = 0, a12 = 0, a22 = 0, chi = 0;
  for (const p of pairs) {
    const [aR, bR] = AB(p.tC + T0C);
    const w = 1 / Math.sqrt(nT(p.tC)), j1 = -w, j2 = (-w * bR) / (aR + s0 * bR);
    a11 += j1 * j1; a12 += j1 * j2; a22 += j2 * j2;
    chi += ((p.lr - lc0 - lnH(p.tC + T0C, s0)) * w) ** 2;
  }
  const det = a11 * a22 - a12 * a12, f2 = chi / Math.max(pairs.length - 2, 1);
  const cov = [[(a22 / det) * f2, (-a12 / det) * f2], [(-a12 / det) * f2, (a11 / det) * f2]];
  const dxT = temps.map((tC) => { const p = fit(pairs, { [tC]: sT }), m = fit(pairs, { [tC]: -sT }); return [(p[0] - m[0]) / 2, (p[1] - m[1]) / 2]; });
  const fc = fit(pairs, Object.fromEntries(temps.map((t) => [t, sT])));
  const dxc = [fc[0] - lc0, fc[1] - s0];
  const sq = (k: number) => cov[k][k] + dxT.reduce((a, v) => a + v[k] ** 2, 0) + dxc[k] ** 2;

  // K_j with the corrected and the nominal ψ_R
  const Kcorr = { R: kFit(points, g, 'R', temps, 'inLin', s0, eps), G: kFit(points, g, 'G', temps, 'inLin', 1, eps), B: kFit(points, g, 'B', temps, 'inLin', 1, eps) };
  const Knom = { R: kFit(points, g, 'R', temps, 'inLin', 1, eps), G: Kcorr.G, B: Kcorr.B };
  const Kfit = { R: kFit(points, g, 'R', temps, 'inFit', s0, eps), G: kFit(points, g, 'G', temps, 'inFit', 1, eps), B: kFit(points, g, 'B', temps, 'inFit', 1, eps) };
  const rr: Record<number, number> = {}, rn: Record<number, number> = {};
  for (const t of temps) { rr[t] = Kcorr.R.perT[t].K / Kcorr.G.perT[t].K; rn[t] = Knom.R.perT[t].K / Knom.G.perT[t].K; }
  const t0 = temps[0], tt = temps[temps.length - 1];
  const gm = Math.exp(temps.reduce((a, t) => a + Math.log(rr[t]), 0) / temps.length);
  const spread = Math.max(...temps.map((t) => Math.abs(rr[t] / gm - 1)));

  // flame, G_j with the corrected ψ_R
  const Gr = (TK: number, s: number) => Math.log(tailG(g, 'R', TK, s) / tailG(g, 'G', TK, 1));
  const flameT = (lnr: number, lc: number, s: number) => root((T) => Gr(T, s) + lc - lnr, 1300, 3500);
  const lnr0 = lc0 + Gr(TF, s0);
  const h = [1e-5, 1e-4];
  const gr = [(flameT(lnr0, lc0 + h[0], s0) - flameT(lnr0, lc0 - h[0], s0)) / (2 * h[0]), (flameT(lnr0, lc0, s0 + h[1]) - flameT(lnr0, lc0, s0 - h[1])) / (2 * h[1])];
  const gv = (v: number[]) => gr[0] * v[0] + gr[1] * v[1];
  const uFlame = Math.sqrt(gr[0] * (cov[0][0] * gr[0] + cov[0][1] * gr[1]) + gr[1] * (cov[1][0] * gr[0] + cov[1][1] * gr[1]) + dxT.reduce((a, v) => a + gv(v) ** 2, 0) + gv(dxc) ** 2);

  // one temperature out at a time
  const lm: Record<number, number> = {};
  for (const t of temps) { const q = pairs.filter((p) => p.tC === t); lm[t] = q.reduce((a, p) => a + p.lr, 0) / q.length; }
  const loo: TailLoo[] = temps.map((tC) => {
    const P = pairs.filter((p) => p.tC !== tC);
    const [lc, s] = fit(P), [lcn] = fit(P, {}, 1);
    const pred = lc + lnH(tC + T0C, s), predn = lcn + lnH(tC + T0C, 1);
    const kRo = kFit(points, g, 'R', [tC], 'inLin', s, eps).perT[tC].K, kGo = Knom.G.perT[tC].K, kRn = Knom.R.perT[tC].K;
    return { tC, s, KRKG: Math.exp(lc), residPct: 100 * (Math.exp(lm[tC] - pred) - 1), residNominalPct: 100 * (Math.exp(lm[tC] - predn) - 1),
      KRKGout: kRo / kGo, KRKGrelPct: 100 * (kRo / kGo / Math.exp(lc) - 1), KRKGrelNominalPct: 100 * (kRn / kGo / Math.exp(lcn) - 1),
      dTflameK: flameT(lnr0, lc, s) - TF, dToutK: flameT(lnr0, Math.log(kRo / kGo), s) - TF };
  });
  // ratio test carried to the flame, K_R/K_G of each temperature against the highest, before and after
  const Gn = (TK: number) => Math.log(tailG(g, 'R', TK, 1) / tailG(g, 'G', TK, 1));
  const lnrn = Math.log(rn[tt]) + Gn(TF);
  const ratio: TailRatioRow[] = temps.map((tC) => ({ tC, KRKGrelTop: rr[tC] / rr[tt], KRKGrelTopNominal: rn[tC] / rn[tt],
    dTflameK: flameT(lnr0, Math.log(rr[tC]) - Math.log(rr[tt]) + lc0, s0) - TF,
    dTflameNominalK: root((T) => Gn(T) + Math.log(rn[tC]) - lnrn, 900, 6000) - TF }));
  const from = temps.length >= 4 ? temps : [t0];
  const worst = Math.max(...loo.filter((l) => from.includes(l.tC)).map((l) => Math.abs(l.KRKGrelPct)));
  const worstN = Math.max(...loo.filter((l) => from.includes(l.tC)).map((l) => Math.abs(l.KRKGrelNominalPct)));
  const strong = loo.find((l) => l.tC === t0)!;
  // fraction of G_R above the cut in the flame with the corrected tail
  const fr = trap(g, (i) => psiAt(g, 'R', s0, i) * g.soot[i] * planck(g.lam[i], TF), 'hi') / tailG(g, 'R', TF, s0);
  return {
    temps, nPairs: pairs.length, nPerT, ok: true, s: s0, lnc: lc0, KRKG: Math.exp(lc0), rmsPct: 100 * rms,
    u: { s: Math.sqrt(sq(1)), sStat: Math.sqrt(cov[1][1]), sT: Math.sqrt(dxT.reduce((a, v) => a + v[1] ** 2, 0)), sCommon: dxc[1], lnc: Math.sqrt(sq(0)), KRKG: Math.exp(lc0) * Math.sqrt(sq(0)), flameK: uFlame, dTdsK: gr[1], dTdlncK: gr[0] },
    Kcorr, Knom, Kfit, KRKGperT: rr, KRKGperTNominal: rn,
    driftPct: 100 * (rr[tt] / rr[t0] - 1), driftNominalPct: 100 * (rn[tt] / rn[t0] - 1), spreadPct: 100 * spread,
    driftNominalByChannelPct: { R: 100 * (Knom.R.perT[tt].K / Knom.R.perT[t0].K - 1), G: 100 * (Knom.G.perT[tt].K / Knom.G.perT[t0].K - 1), B: 100 * (Knom.B.perT[tt].K / Knom.B.perT[t0].K - 1) },
    loo, ratio,
    criterion: { worstPct: worst, nominalPct: worstN, from, pass: worst < 100 * TAIL_CRITERION, strongT: t0, strongPct: strong.KRKGrelPct, limitPct: 100 * TAIL_CRITERION },
    ratioTest: { afterK: Math.max(...loo.filter((l) => from.includes(l.tC)).map((l) => Math.abs(l.dToutK))), beforeK: Math.max(...ratio.map((r) => Math.abs(r.dTflameNominalK))) },
    tail700FlamePct: 100 * fr,
  };
}

/** ψ_R with its tail above the cut scaled by s, on the grid of the response (for the flame stages). */
export function scaleTailR(lamNm: ArrayLike<number>, psi: Rgb<ArrayLike<number>>, s: number, cutNm = TAIL_CUT_NM): Rgb<Float64Array> {
  const cut = cutNm * 1e-9;
  return { R: Float64Array.from(psi.R, (v, i) => (lamNm[i] * 1e-9 >= cut ? s * v : v)), G: Float64Array.from(psi.G), B: Float64Array.from(psi.B) };
}
