/**
 * Finance tab: European call options under Black–Scholes.
 * Forward: σ, r, K, T → price. Inverse: noisy synthetic market quotes → implied
 * volatility per quote and one least-squares σ. With a volatility smile in the market,
 * a single constant σ cannot fit every quote: model error, not noise.
 */
import { rng } from '../linalg.ts';

/** erf with |error| < 1.2e-7 (Numerical Recipes erfc Chebyshev fit). */
export function erf(x: number): number {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}
export const Phi = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
const phi = (x: number) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

/** Black–Scholes call price; tau = time to maturity (years). */
export function bsCall(S: number, K: number, r: number, sigma: number, tau: number): number {
  if (tau <= 0 || sigma <= 0) return Math.max(S - K * Math.exp(-r * Math.max(tau, 0)), 0);
  if (S <= 0) return 0;
  const sq = sigma * Math.sqrt(tau);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / sq, d2 = d1 - sq;
  return S * Phi(d1) - K * Math.exp(-r * tau) * Phi(d2);
}
export function bsDelta(S: number, K: number, r: number, sigma: number, tau: number): number {
  if (tau <= 0) return S > K ? 1 : 0;
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * Math.sqrt(tau));
  return Phi(d1);
}
export function bsVega(S: number, K: number, r: number, sigma: number, tau: number): number {
  if (tau <= 0) return 0;
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * Math.sqrt(tau));
  return S * phi(d1) * Math.sqrt(tau);
}

/** Implied volatility by safeguarded Newton (bisection fallback). NaN if outside no-arbitrage bounds. */
export function impliedVol(price: number, S: number, K: number, r: number, tau: number): number {
  const lo0 = Math.max(S - K * Math.exp(-r * tau), 0);
  if (!(price > lo0 + 1e-10) || price >= S) return NaN;
  let lo = 1e-4, hi = 4, s = 0.3;
  for (let it = 0; it < 100; it++) {
    const f = bsCall(S, K, r, s, tau) - price;
    if (Math.abs(f) < 1e-10) return s;
    if (f > 0) hi = s; else lo = s;
    const v = bsVega(S, K, r, s, tau);
    let n = v > 1e-12 ? s - f / v : NaN;
    if (!(n > lo && n < hi)) n = 0.5 * (lo + hi);
    s = n;
  }
  return s;
}

export interface Quote { K: number; tau: number; price: number; sigmaTrue: number }
export interface Market { S: number; r: number; sigma0: number; smile: boolean; quotes: Quote[] }

/** True volatility of the synthetic market: constant, or a skewed smile in log-moneyness. */
export function sigmaOf(m: Pick<Market, 'S' | 'sigma0' | 'smile'>, K: number, tau: number): number {
  if (!m.smile) return m.sigma0;
  const k = Math.log(K / m.S);
  return m.sigma0 - 0.18 * k + 0.9 * k * k / Math.sqrt(tau / 0.5);
}

export const STRIKES = [80, 85, 90, 95, 100, 105, 110, 115, 120];
export const MATURITIES = [0.25, 1];

/** Synthetic market at spot 100: quotes over strikes and two maturities, noise in % of price (+ small tick). */
export function makeMarket(seed: number, noisePct: number, smile: boolean): Market {
  const R = rng(seed);
  const sigma0 = 0.15 + 0.25 * R.uni();
  const m: Market = { S: 100, r: 0.03, sigma0, smile, quotes: [] };
  for (const tau of MATURITIES) for (const K of STRIKES) {
    const st = sigmaOf(m, K, tau);
    const p = bsCall(m.S, K, m.r, st, tau);
    const noisy = p * (1 + (noisePct / 100) * R.normal()) + 0.01 * R.normal();
    const floor = Math.max(m.S - K * Math.exp(-m.r * tau), 0) + 0.005;
    m.quotes.push({ K, tau, price: Math.max(noisy, floor), sigmaTrue: st });
  }
  return m;
}

/** Vega-weighted-free least squares in price: σ̂ = argmin Σ (C_BS(σ) − price)². Golden section on [0.02, 1.2]. */
export function fitSigma(m: Market): { sigma: number; rms: number } {
  const f = (s: number) => m.quotes.reduce((a, q) => a + (bsCall(m.S, q.K, m.r, s, q.tau) - q.price) ** 2, 0);
  let a = 0.02, b = 1.2;
  const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  for (let it = 0; it < 80; it++) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
  }
  const s = 0.5 * (a + b);
  return { sigma: s, rms: Math.sqrt(f(s) / m.quotes.length) };
}

/** PINN data set: one strike, quotes recorded on several days (different spot levels and times to maturity). */
export interface PinnQuote { S: number; tau: number; price: number }
export function makePinnData(seed: number, sigma: number, r: number, K: number, noisePct: number): PinnQuote[] {
  const R = rng(seed + 101), out: PinnQuote[] = [];
  for (const tau of [0.25, 0.5, 0.75, 1]) for (const S of [80, 90, 100, 110, 120]) {
    const p = bsCall(S, K, r, sigma, tau);
    out.push({ S, tau, price: Math.max(p * (1 + (noisePct / 100) * R.normal()), 0) });
  }
  return out;
}
