/**
 * Radiometry of the workshop tool. Planck, the channel responses ψ_j, E(m), band integrals of the
 * oven B_j(T) = ε ∫ ψ_j I_bλ dλ and of soot G_j(T) = ∫ ψ_j (6π E(m_λ)/λ) I_bλ dλ, the T ↔ ratio
 * table and its inverse, solve_T_fs, the signal mask and f_s.
 *
 * Two integration rules, each matching its Python source.
 *   makeBand          Gauss-Legendre with 6 nodes on every 1 nm piece of the linear interpolant of ψ_j,
 *                     as Band in manim/tools/prepare_data.py (the presentation numbers), or the left
 *                     Riemann sum of that file ('riemann')
 *   kernelTableTrapz  trapezoid on the ψ grid with the per-nm Planck function, as kernel_table in
 *                     fuentes/03_esfera_integradora/radiometric_calibration.py
 * Units. λ in nm in names ending in Nm, metres otherwise. I_bλ in W m⁻² sr⁻¹ m⁻¹, B_j in W m⁻² sr⁻¹,
 * G_j in W m⁻³ sr⁻¹ per unit f_s.
 */
import { interp1, interp, trapz } from './stats.ts';

export const H_PLANCK = 6.62607015e-34;
export const C_LIGHT = 299792458;
export const K_BOLTZ = 1.380649e-23;
/** 2hc², W m² sr⁻¹ (intensity). */
export const C1 = 2 * H_PLANCK * C_LIGHT ** 2;
/** hc/k, m K. */
export const C2 = (H_PLANCK * C_LIGHT) / K_BOLTZ;
export const LAM_REF_NM = 550;
export const CH = ['R', 'G', 'B'] as const;
export type Ch = 'R' | 'G' | 'B';
export type Rgb<T> = { R: T; G: T; B: T };

/** Blackbody spectral intensity I_bλ(λ, T) in W m⁻² sr⁻¹ m⁻¹, λ in m. */
export function planck(lamM: number, T: number): number {
  return C1 / (lamM ** 5 * Math.expm1(C2 / (lamM * T)));
}

/** ∂I_bλ/∂T. */
export function dplanckDT(lamM: number, T: number): number {
  const x = C2 / (lamM * T);
  return (planck(lamM, T) * x * Math.exp(x)) / Math.expm1(x) / T;
}

/** Per-nm Planck exactly as radiometric_calibration.planck_radiance_nm (exp − 1, W m⁻² sr⁻¹ nm⁻¹). */
export function planckNmRC(lamNm: number, T: number): number {
  const l = lamNm * 1e-9;
  return (C1 / l ** 5 / (Math.exp(C2 / (l * T)) - 1.0)) * 1e-9;
}

// ------------------------------------------------------------------------------------- E(m)
/**
 * Law of E(m_λ), editable in the tool. Default the group's, E = 0,284 + 0,272 ln²λ + 0,057 lnλ/λ with
 * λ in µm (annabel_app/core/calibration.py:458, attributed to Yon et al. 2011).
 */
export type EmLaw = { kind: 'group' } | { kind: 'coeffs'; a: number; b: number; c: number } | { kind: 'constant'; E: number };
export const EM_GROUP: EmLaw = { kind: 'group' };

export function emValue(law: EmLaw, lamNm: number): number {
  if (law.kind === 'constant') return law.E;
  const u = lamNm * 1e-3;
  const a = law.kind === 'group' ? 0.284 : law.a, b = law.kind === 'group' ? 0.272 : law.b, c = law.kind === 'group' ? 0.057 : law.c;
  const L = Math.log(u);
  return a + b * L ** 2 + (c * L) / u;
}

// ------------------------------------------------------------------------------------- ψ_j
export interface Response { lamNm: Float64Array; eta: Rgb<Float64Array>; ranges?: Rgb<[number, number]>; format: 'columns' | 'pairs'; name?: string }

/** Sigmoid IR cut 1/(1 + e^{(λ − cut)/width}), as ir_cut_filter. */
export function irCut(lamNm: number, cutNm: number, widthNm = 12): number {
  return 1 / (1 + Math.exp((lamNm - cutNm) / widthNm));
}

export interface IrCut { cutNm: number; widthNm: number }

/**
 * Short-pass filter in front of the lens, τ_sp(λ) = (1 − floor)/(1 + e^{(λ − cut)/width}) + floor, the same
 * model as `shortpass` in prepare_data.py (G2 r2). It multiplies τ(λ) on top of any IR cut.
 */
export interface ShortPass { cutNm: number; widthNm: number; floor: number }

/** Preset "pasa-cortos 700 nm" (decision after G4 r3), edge of 5 nm and floor OD 3 (transmission 10⁻³). */
export const SP700: ShortPass = { cutNm: 700, widthNm: 5, floor: 1e-3 };

export function shortPass(lamNm: number, sp: ShortPass): number {
  return (1 - sp.floor) / (1 + Math.exp((lamNm - sp.cutNm) / sp.widthNm)) + sp.floor;
}

/**
 * ψ_j(λ) = (λ/λ_ref) η_j τ for a per-photon curve (QE), η_j τ for a per-energy curve (A/W).
 * τ is the optional IR cut (none by default for the IMX392, whose curve already includes the mount filter) times
 * the optional short-pass filter declared in the tool (SP700).
 */
export function buildPsi(lamNm: ArrayLike<number>, eta: Rgb<ArrayLike<number>>, opts: { basis?: 'photon' | 'energy'; lamRefNm?: number; irCut?: IrCut | null; shortPass?: ShortPass | null } = {}): Rgb<Float64Array> {
  const basis = opts.basis ?? 'photon', lr = opts.lamRefNm ?? LAM_REF_NM, ir = opts.irCut ?? null, sp = opts.shortPass ?? null;
  const mk = (e: ArrayLike<number>) => {
    const out = new Float64Array(lamNm.length);
    for (let i = 0; i < out.length; i++) {
      const tau = (ir ? irCut(lamNm[i], ir.cutNm, ir.widthNm) : 1) * (sp ? shortPass(lamNm[i], sp) : 1);
      out[i] = basis === 'photon' ? (lamNm[i] / lr) * e[i] * tau : e[i] * tau;
    }
    return out;
  };
  return { R: mk(eta.R), G: mk(eta.G), B: mk(eta.B) };
}

function splitCsvLine(line: string): string[] {
  const d = line.includes(';') && !line.includes(',') ? ';' : line.includes('\t') && !line.includes(',') ? '\t' : ',';
  return line.split(d).map((s) => s.trim());
}

/**
 * Read a response CSV in either format of the tool.
 *   columns  `lambda_nm,R,G,B` (data/imx392_response.csv), lines starting with # ignored, η as written
 *   pairs    the digitiser export of responsivity_basler.csv, read as load_responsivity does
 *            (pairs of columns by header name, sorted, duplicates averaged, 1 nm grid over the union of
 *            ranges, zero outside each curve, divided by the common maximum)
 */
export function parseResponseCsv(text: string, name?: string): Response {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== '' && !l.trim().startsWith('#'));
  const head = splitCsvLine(lines[0]);
  const low = head.map((h) => h.toLowerCase());
  const iR = low.findIndex((h) => h === 'r' || h === 'red');
  const iG = low.findIndex((h) => h === 'g' || h === 'green');
  const iB = low.findIndex((h) => h === 'b' || h === 'blue');
  const pairs = low[1] === '' || head.length >= 6 && low.some((h) => h.startsWith('red')) && low.some((h) => h === '');
  if (!pairs) {
    const lam: number[] = [], R: number[] = [], G: number[] = [], B: number[] = [];
    for (const l of lines.slice(1)) {
      const t = splitCsvLine(l).map(Number);
      if (!t.every((v, k) => k > Math.max(iR, iG, iB) || Number.isFinite(v))) continue;
      lam.push(t[0]); R.push(t[iR]); G.push(t[iG]); B.push(t[iB]);
    }
    return { lamNm: Float64Array.from(lam), eta: { R: Float64Array.from(R), G: Float64Array.from(G), B: Float64Array.from(B) }, format: 'columns', name };
  }
  const col = (n: string) => {
    const i = low.findIndex((h) => h.startsWith(n));
    if (i < 0 || i + 1 >= head.length) throw new Error(`response_csv_no_${n}`);
    return i;
  };
  const raw: Partial<Rgb<{ x: Float64Array; y: Float64Array }>> = {};
  const ranges: Partial<Rgb<[number, number]>> = {};
  for (const [n, c] of [['red', 'R'], ['green', 'G'], ['blue', 'B']] as const) {
    const i = col(n);
    const xs: number[] = [], ys: number[] = [];
    for (const l of lines.slice(1)) {
      const t = l.split(',');
      const x = parseFloat(t[i]), y = parseFloat(t[i + 1]);
      if (t[i] === undefined || t[i + 1] === undefined || t[i].trim() === '' || t[i + 1].trim() === '') continue;
      if (Number.isFinite(x) && Number.isFinite(y) && /^\s*[-+]?[\d.]/.test(t[i]) && /^\s*[-+]?[\d.]/.test(t[i + 1])) { xs.push(x); ys.push(y); }
    }
    const ord = xs.map((_, k) => k).sort((a, b) => xs[a] - xs[b]);
    const ux: number[] = [], uy: number[] = [], cnt: number[] = [];
    for (const k of ord) {
      if (ux.length && ux[ux.length - 1] === xs[k]) { uy[uy.length - 1] += ys[k]; cnt[cnt.length - 1]++; }
      else { ux.push(xs[k]); uy.push(ys[k]); cnt.push(1); }
    }
    raw[c] = { x: Float64Array.from(ux), y: Float64Array.from(uy, (v, k) => v / cnt[k]) };
    ranges[c] = [ux[0], ux[ux.length - 1]];
  }
  const lo = Math.min(ranges.R![0], ranges.G![0], ranges.B![0]);
  const hi = Math.max(ranges.R![1], ranges.G![1], ranges.B![1]);
  const start = Math.floor(lo), stop = Math.ceil(hi) + 1;
  const n = Math.ceil(stop - start);
  const lamNm = Float64Array.from({ length: n }, (_, k) => start + k);
  const eta = {} as Rgb<Float64Array>;
  let gmax = 0;
  for (const c of CH) {
    const e = interp(lamNm, raw[c]!.x, raw[c]!.y, 0, 0);
    for (let k = 0; k < e.length; k++) { if (e[k] < 0) e[k] = 0; if (e[k] > gmax) gmax = e[k]; }
    eta[c] = e;
  }
  for (const c of CH) for (let k = 0; k < n; k++) eta[c][k] /= gmax;
  return { lamNm, eta, ranges: ranges as Rgb<[number, number]>, format: 'pairs', name };
}

// ------------------------------------------------------------------------------------- bands
export interface Band { lamNm: Float64Array; lam: Float64Array; w: Float64Array; psi: Rgb<Float64Array>; E: Float64Array; law: EmLaw }

let GL6: { x: Float64Array; w: Float64Array } | null = null;
/** Gauss-Legendre nodes and weights on [−1, 1] by Newton on P_n (same values as numpy leggauss to 1e-15). */
export function gaussLegendre(n: number): { x: Float64Array; w: Float64Array } {
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let dp = 0;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) { const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
      dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x[n - 1 - i] = z;
    w[n - 1 - i] = 2 / ((1 - z * z) * dp * dp);
  }
  return { x, w };
}

/**
 * Band integration nodes of a ψ_j given on lamNm (ascending), as prepare_data.Band.
 *  'gl'       6 Gauss-Legendre nodes per piece, ψ linearly interpolated at the nodes
 *  'riemann'  left Riemann sum on the file grid
 */
export function makeBand(lamNm: ArrayLike<number>, psi: Rgb<ArrayLike<number>>, mode: 'gl' | 'riemann' = 'gl', law: EmLaw = EM_GROUP): Band {
  const nl = lamNm.length;
  let nodes: Float64Array, wNm: Float64Array, P: Rgb<Float64Array>;
  if (mode === 'gl') {
    if (!GL6) GL6 = gaussLegendre(6);
    const { x, w } = GL6, q = x.length;
    nodes = new Float64Array((nl - 1) * q); wNm = new Float64Array((nl - 1) * q);
    for (let s = 0; s < nl - 1; s++) {
      const a = lamNm[s], b = lamNm[s + 1];
      for (let k = 0; k < q; k++) { nodes[s * q + k] = 0.5 * (b - a) * x[k] + 0.5 * (a + b); wNm[s * q + k] = 0.5 * (b - a) * w[k]; }
    }
    P = { R: interp(nodes, lamNm, psi.R), G: interp(nodes, lamNm, psi.G), B: interp(nodes, lamNm, psi.B) };
  } else {
    nodes = Float64Array.from({ length: nl - 1 }, (_, k) => lamNm[k]);
    wNm = Float64Array.from({ length: nl - 1 }, (_, k) => lamNm[k + 1] - lamNm[k]);
    const cut = (a: ArrayLike<number>) => Float64Array.from({ length: nl - 1 }, (_, k) => a[k]);
    P = { R: cut(psi.R), G: cut(psi.G), B: cut(psi.B) };
  }
  return { lamNm: nodes, lam: nodes.map((v) => v * 1e-9), w: wNm.map((v) => v * 1e-9), psi: P, E: nodes.map((v) => emValue(law, v)), law };
}

/** The same band with another E(m) law. */
export function withEmLaw(b: Band, law: EmLaw): Band {
  return { ...b, E: b.lamNm.map((v) => emValue(law, v)), law };
}

/** The same band with ψ_j multiplied by `factor` above cutNm (model uncertainty of the IR tail). */
export function scaleTail(b: Band, factor: number, cutNm = 700): Band {
  const f = (p: Float64Array) => p.map((v, k) => (b.lamNm[k] > cutNm ? v * factor : v));
  return { ...b, psi: { R: f(b.psi.R), G: f(b.psi.G), B: f(b.psi.B) } };
}

/** Oven signal B_j(T) = ε ∫ ψ_j I_bλ dλ in W m⁻² sr⁻¹. */
export function bandX(b: Band, c: Ch, T: number, eps = 1): number {
  const p = b.psi[c];
  let s = 0;
  for (let k = 0; k < p.length; k++) s += planck(b.lam[k], T) * p[k] * b.w[k];
  return eps * s;
}

/** Soot kernel G_j(T) = ∫ ψ_j (6π E/λ) I_bλ dλ in W m⁻³ sr⁻¹ per unit f_s (or its T derivative). */
export function bandG(b: Band, c: Ch, T: number, deriv = false): number {
  const p = b.psi[c];
  let s = 0;
  for (let k = 0; k < p.length; k++) {
    const f = deriv ? dplanckDT(b.lam[k], T) : planck(b.lam[k], T);
    s += f * p[k] * ((6 * Math.PI * b.E[k]) / b.lam[k]) * b.w[k];
  }
  return s;
}

/** Fraction of B_j(T) (kind 'X') or G_j(T) (kind 'G') that comes from λ > cutNm (ir_tail warning over 10 %). */
export function tailFraction(b: Band, c: Ch, T: number, cutNm = 700, kind: 'X' | 'G' = 'X'): number {
  const p = b.psi[c];
  let tot = 0, hi = 0;
  for (let k = 0; k < p.length; k++) {
    let v = planck(b.lam[k], T) * p[k] * b.w[k];
    if (kind === 'G') v *= (6 * Math.PI * b.E[k]) / b.lam[k];
    tot += v;
    if (b.lamNm[k] > cutNm) hi += v;
  }
  return hi / tot;
}

export interface RatioTable { T: Float64Array; G: Rgb<Float64Array>; RG: Float64Array; BG: Float64Array; BR: Float64Array; dlnRG: Float64Array }

/** T ↔ ratio table with G_j(T), R/G, B/G, B/R and d ln(R/G)/dT. Default 1000 to 2600 K every 1 K. */
export function ratioTable(b: Band, Tgrid: ArrayLike<number> = Float64Array.from({ length: 1601 }, (_, k) => 1000 + k)): RatioTable {
  const n = Tgrid.length;
  const T = Float64Array.from(Tgrid);
  const G = { R: new Float64Array(n), G: new Float64Array(n), B: new Float64Array(n) };
  const RG = new Float64Array(n), BG = new Float64Array(n), BR = new Float64Array(n), dlnRG = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (const c of CH) G[c][i] = bandG(b, c, T[i]);
    RG[i] = G.R[i] / G.G[i]; BG[i] = G.B[i] / G.G[i]; BR[i] = G.B[i] / G.R[i];
    dlnRG[i] = bandG(b, 'R', T[i], true) / G.R[i] - bandG(b, 'G', T[i], true) / G.G[i];
  }
  return { T, G, RG, BG, BR, dlnRG };
}

/** Is the table strictly monotonic? */
export function isMonotonic(r: ArrayLike<number>): boolean {
  let up = true, down = true;
  for (let i = 1; i < r.length; i++) { if (!(r[i] > r[i - 1])) up = false; if (!(r[i] < r[i - 1])) down = false; }
  return up || down;
}

/**
 * T with table(T) = r by linear interpolation in ln r, NaN outside the table (as invert_monotone in
 * prepare_data.py). A table that is not monotonic (B/G with the IMX392 has a minimum near 1530 K) has two
 * branches. By default a value that both branches reach is ambiguous and gives NaN (code
 * `ratio_ambiguous`, GB2r1-numerico-04). `branch: 'high'` uses only the branch that holds the high-T end,
 * which is right only when T is known to be above the turning point (for example from R/G), and
 * `range` limits the table to [lo, hi] K before inverting.
 */
export function invertRatio(Tgrid: ArrayLike<number>, table: ArrayLike<number>, r: number, opts: { branch?: 'unique' | 'high'; range?: [number, number] } = {}): number {
  let Tg = Array.from(Tgrid), tb = Array.from(table);
  if (opts.range) {
    const keep = Tg.map((_, i) => i).filter((i) => Tg[i] >= opts.range![0] && Tg[i] <= opts.range![1]);
    Tg = keep.map((i) => Tg[i]); tb = keep.map((i) => tb[i]);
  }
  const n = tb.length;
  if (n < 2) return NaN;
  let i0 = 0;
  if (!isMonotonic(tb)) {
    const dirEnd = Math.sign(tb[n - 1] - tb[n - 2]);
    i0 = n - 2;
    while (i0 > 0 && Math.sign(tb[i0] - tb[i0 - 1]) === dirEnd) i0--;
    if ((opts.branch ?? 'unique') === 'unique' && ratioAmbiguous(tb.slice(0, i0 + 1), r)) return NaN;
  }
  const lr: number[] = [], Ts: number[] = [];
  for (let i = i0; i < n; i++) { lr.push(Math.log(tb[i])); Ts.push(Tg[i]); }
  if (lr[0] > lr[lr.length - 1]) { lr.reverse(); Ts.reverse(); }
  return interp1(Math.log(r), lr, Ts, NaN, NaN);
}

/** Does the discarded (low-T) branch of a table also reach the value r? */
export function ratioAmbiguous(lowBranch: ArrayLike<number>, r: number): boolean {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < lowBranch.length; i++) { lo = Math.min(lo, lowBranch[i]); hi = Math.max(hi, lowBranch[i]); }
  return r >= lo && r <= hi;
}

/**
 * Kernel table exactly as radiometric_calibration.kernel_table, trapezoid in nm with the per-nm Planck
 * function, ψ on lamNm. Returns G[c][iT] in W m⁻³ sr⁻¹ for f_s = 1.
 */
export function kernelTableTrapz(lamNm: ArrayLike<number>, psi: Rgb<ArrayLike<number>>, Tgrid: ArrayLike<number>, law: EmLaw = EM_GROUP): Rgb<Float64Array> {
  const nl = lamNm.length;
  const Em = Float64Array.from({ length: nl }, (_, k) => emValue(law, lamNm[k]));
  const out = { R: new Float64Array(Tgrid.length), G: new Float64Array(Tgrid.length), B: new Float64Array(Tgrid.length) };
  const y = new Float64Array(nl);
  for (let t = 0; t < Tgrid.length; t++) {
    const ju = new Float64Array(nl);
    for (let k = 0; k < nl; k++) ju[k] = ((6 * Math.PI * Em[k]) / (lamNm[k] * 1e-9)) * planckNmRC(lamNm[k], Tgrid[t]);
    for (const c of CH) {
      for (let k = 0; k < nl; k++) y[k] = psi[c][k] * ju[k];
      out[c][t] = trapz(y, lamNm);
    }
  }
  return out;
}

/** ∫ ψ_c L dλ by the trapezoid in nm, as band_integral (L per nm on the same grid). */
export function bandIntegral(lamNm: ArrayLike<number>, psiC: ArrayLike<number>, L: ArrayLike<number>): number {
  const y = new Float64Array(lamNm.length);
  for (let k = 0; k < y.length; k++) y[k] = psiC[k] * L[k];
  return trapz(y, lamNm);
}

// ------------------------------------------------------------------------------------- T and f_s
export interface TfsResult { T: Float64Array; fs: Float64Array; chi2: Float64Array }

/**
 * Port of solve_T_fs (radiometric_calibration.py). j is interleaved (N × 3) in W m⁻³ sr⁻¹, G the kernel
 * table on Tgrid. f_s*(T) = Σ w j G / Σ w G², χ²(T) = Σ w j² − (Σ w j G)²/Σ w G², T at the grid point of
 * minimum χ² (first one on ties). Masked-out pixels are NaN.
 */
export function solveTfs(j: ArrayLike<number>, G: Rgb<ArrayLike<number>>, Tgrid: ArrayLike<number>, weights: [number, number, number] = [1, 1, 1], mask?: ArrayLike<number> | null): TfsResult {
  const npx = j.length / 3, NT = Tgrid.length;
  const wG = [new Float64Array(NT), new Float64Array(NT), new Float64Array(NT)];
  const Gs = [G.R, G.G, G.B];
  const denom = new Float64Array(NT);
  for (let t = 0; t < NT; t++) {
    for (let c = 0; c < 3; c++) wG[c][t] = weights[c] * Gs[c][t];
    denom[t] = wG[0][t] * Gs[0][t] + wG[1][t] * Gs[1][t] + wG[2][t] * Gs[2][t];
    if (!(denom[t] > 0)) throw new Error('kernel_nonpositive');
  }
  const T = new Float64Array(npx).fill(NaN), fs = new Float64Array(npx).fill(NaN), chi2 = new Float64Array(npx).fill(NaN);
  for (let p = 0; p < npx; p++) {
    if (mask && !mask[p]) continue;
    const j0 = j[3 * p], j1 = j[3 * p + 1], j2 = j[3 * p + 2];
    const A = weights[0] * j0 * j0 + weights[1] * j1 * j1 + weights[2] * j2 * j2;
    let best = -Infinity, k = 0, numk = 0;
    for (let t = 0; t < NT; t++) {
      const num = j0 * wG[0][t] + j1 * wG[1][t] + j2 * wG[2][t];
      const q = (num * num) / denom[t];
      if (q > best) { best = q; k = t; numk = num; }
    }
    T[p] = Tgrid[k];
    fs[p] = numk / denom[k];
    chi2[p] = A - best;
  }
  return { T, fs, chi2 };
}

/** Flags of solveTfs (GB2r1-numerico-10). Bit 1 f_s ≤ 0 (`fs_negative`), bit 2 T on the first or last grid point (`T_at_grid_edge`). */
export const TFS_FS_NEGATIVE = 1, TFS_T_EDGE = 2;
export function tfsFlags(res: TfsResult, Tgrid: ArrayLike<number>): Uint8Array {
  const n = res.T.length, out = new Uint8Array(n);
  const lo = Tgrid[0], hi = Tgrid[Tgrid.length - 1];
  for (let p = 0; p < n; p++) {
    if (Number.isNaN(res.T[p])) continue;
    if (!(res.fs[p] > 0)) out[p] |= TFS_FS_NEGATIVE;
    if (res.T[p] === lo || res.T[p] === hi) out[p] |= TFS_T_EDGE;
  }
  return out;
}

/** solveTfs result with the flagged pixels set to NaN (T, f_s and χ²), for the maps of the tool. */
export function maskTfs(res: TfsResult, Tgrid: ArrayLike<number>): { res: TfsResult; flags: Uint8Array } {
  const flags = tfsFlags(res, Tgrid);
  const T = Float64Array.from(res.T), fs = Float64Array.from(res.fs), chi2 = Float64Array.from(res.chi2);
  for (let p = 0; p < flags.length; p++) if (flags[p]) { T[p] = NaN; fs[p] = NaN; chi2[p] = NaN; }
  return { res: { T, fs, chi2 }, flags };
}

/**
 * Off-grid refinement of solveTfs (GB2r1-numerico-11). χ²(T) of the three grid points around the minimum,
 * a parabola for T, G_j at that T by the quadratic through the same three points, and f_s* again. The grid
 * result (exact port, kept for parity) is the input. Pixels at the grid edge keep the grid value.
 */
export function refineTfs(j: ArrayLike<number>, G: Rgb<ArrayLike<number>>, Tgrid: ArrayLike<number>, res: TfsResult, weights: [number, number, number] = [1, 1, 1]): TfsResult {
  const n = res.T.length, NT = Tgrid.length;
  const T = Float64Array.from(res.T), fs = Float64Array.from(res.fs), chi2 = Float64Array.from(res.chi2);
  const Gs = [G.R, G.G, G.B];
  const fit = (p: number, g: number[]) => {
    let num = 0, den = 0, A = 0;
    for (let c = 0; c < 3; c++) { const jc = j[3 * p + c]; num += weights[c] * jc * g[c]; den += weights[c] * g[c] * g[c]; A += weights[c] * jc * jc; }
    return { fs: num / den, chi2: A - (num * num) / den };
  };
  const lag = (k: number, x: number, c: number) => {
    const x0 = Tgrid[k - 1], x1 = Tgrid[k], x2 = Tgrid[k + 1], g = Gs[c];
    return g[k - 1] * ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2)) + g[k] * ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2)) + g[k + 1] * ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1));
  };
  for (let p = 0; p < n; p++) {
    if (Number.isNaN(res.T[p])) continue;
    let k = -1;
    for (let t = 0; t < NT; t++) if (Tgrid[t] === res.T[p]) { k = t; break; }
    if (k <= 0 || k >= NT - 1) continue;
    const c = [-1, 0, 1].map((d) => fit(p, [0, 1, 2].map((ch) => Gs[ch][k + d])).chi2);
    const den = c[0] - 2 * c[1] + c[2];
    if (!(den > 0)) continue;
    const u = Math.min(1, Math.max(-1, (0.5 * (c[0] - c[2])) / den));
    const Tr = Tgrid[k] + u * (u >= 0 ? Tgrid[k + 1] - Tgrid[k] : Tgrid[k] - Tgrid[k - 1]);
    const r = fit(p, [0, 1, 2].map((ch) => lag(k, Tr, ch)));
    T[p] = Tr; fs[p] = r.fs; chi2[p] = r.chi2;
  }
  return { T, fs, chi2 };
}

/** Pixels worth fitting, Σ_j j_j > frac × max (signal_mask), or ≥ with `inclusive` (prepare_data chain). */
export function signalMask(j: ArrayLike<number>, frac = 0.02, inclusive = false): Uint8Array {
  const n = j.length / 3;
  const tot = new Float64Array(n);
  let mx = -Infinity;
  for (let p = 0; p < n; p++) { tot[p] = j[3 * p] + j[3 * p + 1] + j[3 * p + 2]; if (tot[p] > mx) mx = tot[p]; }
  const m = new Uint8Array(n);
  for (let p = 0; p < n; p++) m[p] = (inclusive ? tot[p] >= frac * mx : tot[p] > frac * mx) ? 1 : 0;
  return m;
}

/** T from the ratio j_a/j_b of every pixel with the table (NaN outside it). */
export function ratioTemperature(ja: ArrayLike<number>, jb: ArrayLike<number>, Tgrid: ArrayLike<number>, table: ArrayLike<number>, opts: { branch?: 'unique' | 'high'; range?: [number, number] } = {}): Float64Array {
  const out = new Float64Array(ja.length);
  for (let i = 0; i < out.length; i++) out[i] = invertRatio(Tgrid, table, ja[i] / jb[i], opts);
  return out;
}

/** f_s = j_c / G_c(T) from one channel, G interpolated linearly on its table (NaN where T is NaN). */
export function fsFromChannel(jc: ArrayLike<number>, T: ArrayLike<number>, Tgrid: ArrayLike<number>, Gc: ArrayLike<number>): Float64Array {
  const out = new Float64Array(jc.length);
  for (let i = 0; i < out.length; i++) out[i] = Number.isFinite(T[i]) ? jc[i] / interp1(T[i], Tgrid, Gc) : NaN;
  return out;
}

/**
 * f_s by weighted least squares of the channels at a given T, Σ w_c j_c G_c / Σ w_c G_c² (fs_3ch of
 * prepare_data.py with its chans). Weights [1, 1, 0] are R and G, the default of the tool and of the
 * presentation since G4 r2 (blue out, its oven K_B does not hold in the flame). A per-pixel `noBlue`
 * mask sets the blue weight to 0 there (chemiluminescence). NaN where T is NaN.
 */
export function fsChannels(j: ArrayLike<number>, T: ArrayLike<number>, Tgrid: ArrayLike<number>, G: Rgb<ArrayLike<number>>, weights: [number, number, number] = [1, 1, 0], noBlue?: ArrayLike<number> | null): Float64Array {
  const n = T.length;
  const out = new Float64Array(n);
  for (let p = 0; p < n; p++) {
    if (!Number.isFinite(T[p])) { out[p] = NaN; continue; }
    const w2 = noBlue && noBlue[p] ? 0 : weights[2];
    const g0 = interp1(T[p], Tgrid, G.R), g1 = interp1(T[p], Tgrid, G.G), g2 = w2 ? interp1(T[p], Tgrid, G.B) : 0;
    const den = weights[0] * g0 * g0 + weights[1] * g1 * g1 + w2 * g2 * g2;
    out[p] = den > 0 ? (weights[0] * j[3 * p] * g0 + weights[1] * j[3 * p + 1] * g1 + w2 * j[3 * p + 2] * g2) / den : NaN;
  }
  return out;
}

/** f_s by least squares of the three channels at a given T, weights 1 (advanced option, needs a valid K_B). */
export function fsThreeChannels(j: ArrayLike<number>, T: ArrayLike<number>, Tgrid: ArrayLike<number>, G: Rgb<ArrayLike<number>>): Float64Array {
  return fsChannels(j, T, Tgrid, G, [1, 1, 1]);
}

/** ANNAbel delivers J per millimetre (dr in mm). Per metre is × 1e3. */
export const PER_MM_TO_PER_M = 1e3;

/**
 * Greybody colour temperature of a spectrum by shape only (fit_colour_temperature), on
 * linspace(1000, 6000, 2501) by default, λ range 420 to 750 nm.
 */
export function fitColourTemperature(lamNm: ArrayLike<number>, L: ArrayLike<number>, range: [number, number] = [420, 750], Tgrid?: ArrayLike<number>): number {
  const idx: number[] = [];
  for (let k = 0; k < lamNm.length; k++) if (lamNm[k] >= range[0] && lamNm[k] <= range[1] && L[k] > 0) idx.push(k);
  if (idx.length < 5) return NaN;
  const Tg = Tgrid ?? Float64Array.from({ length: 2501 }, (_, k) => 1000 + (5000 * k) / 2500);
  const y = idx.map((k) => Math.log(L[k]));
  let best = Infinity, bestT = NaN;
  const r = new Float64Array(idx.length);
  for (let t = 0; t < Tg.length; t++) {
    let m = 0;
    for (let q = 0; q < idx.length; q++) { r[q] = y[q] - Math.log(planckNmRC(lamNm[idx[q]], Tg[t])); m += r[q]; }
    m /= idx.length;
    let s = 0;
    for (let q = 0; q < idx.length; q++) s += (r[q] - m) ** 2;
    if (s < best) { best = s; bestT = Tg[t]; }
  }
  return bestT;
}
