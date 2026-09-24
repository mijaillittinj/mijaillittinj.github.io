/**
 * Textbook models behind the interactive teaching examples (ILN221, ILN222, ILN223).
 * Pure functions, SI units unless stated otherwise; checked in tests/teaching.test.ts.
 */

export const G = 9.81;               // m/s²
export const SIGMA = 5.670374419e-8; // W/(m² K⁴)
const H_PLANCK = 6.62607015e-34, C_LIGHT = 2.99792458e8, K_B = 1.380649e-23;
export const WIEN_B = 2897.771955;   // µm K

/* ------------------------------------------------------------ Pipe flow */

export interface Fluid { key: string; rho: number; mu: number }
/** Property values at about 20 °C (oil: a light lubricating oil, order of magnitude only). */
export const fluids: Fluid[] = [
  { key: 'water', rho: 998, mu: 1.0e-3 },
  { key: 'air', rho: 1.204, mu: 1.825e-5 },
  { key: 'oil', rho: 880, mu: 0.1 },
];

export const reynolds = (rho: number, V: number, D: number, mu: number) => (rho * V * D) / mu;

export const RE_LAM = 2300, RE_TURB = 4000;
export type Regime = 'laminar' | 'transitional' | 'turbulent';
export const regime = (Re: number): Regime => (Re < RE_LAM ? 'laminar' : Re < RE_TURB ? 'transitional' : 'turbulent');

export const fLaminar = (Re: number) => 64 / Re;

/** Colebrook–White, solved by fixed-point iteration on x = 1/√f. relRough = ε/D. */
export function fColebrook(Re: number, relRough: number): number {
  let x = 1 / Math.sqrt(fHaaland(Re, relRough));
  for (let k = 0; k < 50; k++) {
    const xn = -2 * Math.log10(relRough / 3.7 + (2.51 * x) / Re);
    if (Math.abs(xn - x) < 1e-12) { x = xn; break; }
    x = xn;
  }
  return 1 / (x * x);
}

/** Haaland's explicit approximation of Colebrook. */
export function fHaaland(Re: number, relRough: number): number {
  const x = -1.8 * Math.log10((relRough / 3.7) ** 1.11 + 6.9 / Re);
  return 1 / (x * x);
}

/** Darcy friction factor: 64/Re below 2300, Colebrook above (transitional values are uncertain). */
export const frictionFactor = (Re: number, relRough: number) =>
  Re < RE_LAM ? fLaminar(Re) : fColebrook(Re, relRough);

/** Darcy–Weisbach pressure drop per unit length, Pa/m. */
export const dpPerLength = (f: number, rho: number, V: number, D: number) => (f * rho * V * V) / (2 * D);

/** Velocity profile u/V (V = mean velocity) at s = r/R. Laminar: Hagen–Poiseuille; turbulent: 1/n power law. */
export function uLaminar(s: number) { return 2 * (1 - s * s); }
export function uPowerLaw(s: number, n = 7) {
  const umaxOverV = ((n + 1) * (2 * n + 1)) / (2 * n * n);
  return umaxOverV * Math.max(0, 1 - s) ** (1 / n);
}

/* ------------------------------------------------------------ Tank draining (Torricelli) */

/** h(t) for a cylindrical tank of diameter Dt draining through an orifice of diameter d. */
export function tankHeight(t: number, h0: number, Dt: number, d: number, Cd: number): number {
  const k = Cd * (d / Dt) ** 2 * Math.sqrt(G / 2);
  const s = Math.sqrt(h0) - k * t;
  return s > 0 ? s * s : 0;
}
export function drainTime(h0: number, Dt: number, d: number, Cd: number): number {
  return (Dt / d) ** 2 * Math.sqrt(2 * h0 / G) / Cd;
}

/* ------------------------------------------------------------ Transient conduction, plane wall */

/**
 * Non-dimensional plane wall of half-thickness L, initially θ = 1, convective
 * boundary at ξ = 1 (Biot number Bi), symmetry at ξ = 0:
 *   ∂θ/∂τ = ∂²θ/∂ξ²,  ∂θ/∂ξ(0) = 0,  −∂θ/∂ξ(1) = Bi θ(1),  τ = Fo = αt/L².
 * Implicit (backward Euler) finite volumes, unconditionally stable. Returns the
 * profile at each requested Fourier number (ascending), on nodes ξ_i = i/(n−1).
 */
export function wallTransient(Bi: number, foTimes: number[], n = 41, dtau = 2e-3): Float64Array[] {
  const dx = 1 / (n - 1);
  const cp = new Float64Array(n), dp = new Float64Array(n);
  // one backward-Euler step of size h (half control volumes at both ends)
  const step = (th: Float64Array, h: number) => {
    const r = h / (dx * dx);
    const A = (i: number) => (i === 0 ? 0 : i === n - 1 ? -2 * r : -r);
    const B = (i: number) => 1 + 2 * r + (i === n - 1 ? 2 * r * dx * Bi : 0);
    const C = (i: number) => (i === 0 ? -2 * r : i === n - 1 ? 0 : -r);
    cp[0] = C(0) / B(0); dp[0] = th[0] / B(0);
    for (let i = 1; i < n; i++) {
      const m = B(i) - A(i) * cp[i - 1];
      cp[i] = C(i) / m;
      dp[i] = (th[i] - A(i) * dp[i - 1]) / m;
    }
    const nt = new Float64Array(n);
    nt[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) nt[i] = dp[i] - cp[i] * nt[i + 1];
    return nt;
  };
  let th = new Float64Array(n).fill(1);
  let tau = 0;
  const out: Float64Array[] = [];
  for (const target of foTimes) {
    while (tau < target - 1e-12) {
      const h = Math.min(dtau, target - tau);
      th = step(th, h);
      tau += h;
    }
    out.push(th.slice());
  }
  return out;
}

/** Mean of a nodal profile on a uniform grid (trapezoid rule). */
export function profileMean(th: ArrayLike<number>): number {
  const n = th.length;
  let s = 0;
  for (let i = 0; i < n - 1; i++) s += 0.5 * (th[i] + th[i + 1]);
  return s / (n - 1);
}

export const lumped = (Bi: number, Fo: number) => Math.exp(-Bi * Fo);

/** First eigenvalue of ζ tan ζ = Bi and the one-term centre solution (valid for Fo ≳ 0.2). */
export function oneTermCentre(Bi: number, Fo: number): number {
  let lo = 1e-9, hi = Math.PI / 2 - 1e-12;
  for (let k = 0; k < 200; k++) { const m = 0.5 * (lo + hi); if (m * Math.tan(m) < Bi) lo = m; else hi = m; }
  const z = 0.5 * (lo + hi);
  const C1 = (4 * Math.sin(z)) / (2 * z + Math.sin(2 * z));
  return C1 * Math.exp(-z * z * Fo);
}

/* ------------------------------------------------------------ Critical radius of insulation */

/** Heat loss per metre of an insulated pipe (pipe-wall resistance neglected), W/m. */
export function heatLossPerLength(r: number, ri: number, k: number, h: number, dT: number): number {
  return (2 * Math.PI * dT) / (Math.log(r / ri) / k + 1 / (h * r));
}
export const criticalRadius = (k: number, h: number) => k / h;

/* ------------------------------------------------------------ Blackbody radiation */

/** Planck spectral emissive power E_bλ in W/(m² µm), λ in µm. */
export function planck(lamUm: number, T: number): number {
  const lam = lamUm * 1e-6;
  const x = (H_PLANCK * C_LIGHT) / (lam * K_B * T);
  if (x > 700) return 0;
  return (2 * Math.PI * H_PLANCK * C_LIGHT * C_LIGHT) / (lam ** 5 * Math.expm1(x)) * 1e-6;
}
export const wienPeak = (T: number) => WIEN_B / T; // µm

/** ∫ E_bλ dλ between two wavelengths (µm), log-spaced trapezoid. */
export function bandPower(l0: number, l1: number, T: number, n = 2000): number {
  const a = Math.log(l0), b = Math.log(l1);
  let s = 0, prevL = l0, prevE = planck(l0, T);
  for (let i = 1; i <= n; i++) {
    const l = Math.exp(a + ((b - a) * i) / n);
    const e = planck(l, T);
    s += 0.5 * (e + prevE) * (l - prevL);
    prevL = l; prevE = e;
  }
  return s;
}
export const bandFraction = (l0: number, l1: number, T: number) => bandPower(l0, l1, T) / (SIGMA * T ** 4);

/* ------------------------------------------------------------ Heat exchangers, ε–NTU */

export function effCounter(NTU: number, Cr: number): number {
  if (Math.abs(1 - Cr) < 1e-9) return NTU / (1 + NTU);
  const e = Math.exp(-NTU * (1 - Cr));
  return (1 - e) / (1 - Cr * e);
}
export function effParallel(NTU: number, Cr: number): number {
  return (1 - Math.exp(-NTU * (1 + Cr))) / (1 + Cr);
}
