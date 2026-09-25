/**
 * Utility-scale PV plant in a desert climate: soiling and degradation (electrical tab).
 *
 * Forward model, daily resolution over two years (t in days, t = 0 is a cleaning day):
 *   H(t)        plane-of-array irradiation, kWh/m²/day (seasonal, clear sky, a few low days)
 *   Ta(t)       daytime ambient temperature, °C
 *   Tc = Ta + k_T H                      cell temperature (NOCT-type, daily)
 *   E_clean = P · H · PR0 · (1 + γ (Tc − 25))            expected energy of a clean plant, MWh
 *   SR(t) = exp(−r (t − t_k))            soiling ratio, reset to 1 at each cleaning t_k
 *   D(t)  = 1 − R_d t / 365               degradation
 *   E = E_clean · SR · D · (1 + ε)        measured energy
 * The performance index PI = E / E_clean is what monitoring data give once irradiance and
 * temperature are accounted for.
 *
 * Classical inverse: detect cleanings as positive jumps of PI, then fit
 *   log PI = c0 − r (t − t_k) − d t/365      (linear least squares in c0, r, d)
 * with standard errors. Without cleanings t − t_k = t and r and d are confounded.
 */
import { rng } from '../linalg.ts';

export const DAYS = 730;
export const P_MWP = 100;
export const PR0 = 0.84;
export const GAMMA = -0.0035; // 1/°C
export const K_T = 2.4;       // °C per kWh/m²/day

export interface PlantTruth {
  r: number;         // soiling rate, fraction per day (e.g. 0.002 = 0.2 %/day)
  interval: number;  // days between cleanings; Infinity = no cleanings
  Rd: number;        // degradation, fraction per year
  noise: number;     // relative noise of daily energy
}

export interface PlantData {
  H: Float64Array; Ta: Float64Array; Tc: Float64Array;
  Eclean: Float64Array; SR: Float64Array; D: Float64Array; E: Float64Array; PI: Float64Array;
  cleanings: number[]; // cleaning days (from the maintenance log), including day 0
}

/** Weather does not depend on the truth, so changing sliders keeps the same two years. */
export function weather(seed: number) {
  const R = rng(seed);
  const H = new Float64Array(DAYS), Ta = new Float64Array(DAYS);
  for (let t = 0; t < DAYS; t++) {
    const season = Math.cos((2 * Math.PI * (t + 10)) / 365); // southern summer around 1 January
    let h = 8.4 + 1.6 * season + 0.15 * R.normal();
    if (R.uni() < 0.035) h *= 0.35 + 0.5 * R.uni(); // rare cloudy days
    H[t] = Math.max(0.5, h);
    Ta[t] = 22 + 5.5 * season + 1.5 * R.normal();
  }
  return { H, Ta };
}

export function cleaningDays(interval: number): number[] {
  const out = [0];
  if (!Number.isFinite(interval)) return out;
  for (let t = interval; t < DAYS; t += interval) out.push(Math.round(t));
  return out;
}

export function simulate(truth: PlantTruth, seed: number): PlantData {
  const { H, Ta } = weather(seed);
  const R = rng(seed + 1000);
  const cleanings = cleaningDays(truth.interval);
  const Tc = new Float64Array(DAYS), Eclean = new Float64Array(DAYS), SR = new Float64Array(DAYS);
  const D = new Float64Array(DAYS), E = new Float64Array(DAYS), PI = new Float64Array(DAYS);
  let k = 0;
  for (let t = 0; t < DAYS; t++) {
    while (k + 1 < cleanings.length && cleanings[k + 1] <= t) k++;
    Tc[t] = Ta[t] + K_T * H[t];
    Eclean[t] = P_MWP * H[t] * PR0 * (1 + GAMMA * (Tc[t] - 25));
    SR[t] = Math.exp(-truth.r * (t - cleanings[k]));
    D[t] = 1 - (truth.Rd * t) / 365;
    E[t] = Eclean[t] * SR[t] * D[t] * (1 + truth.noise * R.normal());
    PI[t] = E[t] / Eclean[t];
  }
  return { H, Ta, Tc, Eclean, SR, D, E, PI, cleanings };
}

/** Last cleaning day at or before each day. */
export function lastCleaning(cleanings: number[]): Int32Array {
  const out = new Int32Array(DAYS);
  let k = 0;
  for (let t = 0; t < DAYS; t++) {
    while (k + 1 < cleanings.length && cleanings[k + 1] <= t) k++;
    out[t] = cleanings[k];
  }
  return out;
}

/** Robust noise level of PI from day-to-day differences (MAD). */
export function noiseLevel(PI: ArrayLike<number>): number {
  const d: number[] = [];
  for (let t = 1; t < PI.length; t++) d.push(Math.abs(PI[t] - PI[t - 1]));
  d.sort((a, b) => a - b);
  return (1.4826 * d[Math.floor(d.length / 2)]) / Math.SQRT2;
}

/** Cleanings detected as positive jumps of PI (mean of 3 days after minus 3 days before). */
export function detectCleanings(PI: ArrayLike<number>): number[] {
  const s = noiseLevel(PI), thr = Math.max(0.004, 2.6 * s * Math.sqrt(2 / 3));
  const jump = new Float64Array(PI.length);
  for (let t = 3; t + 2 < PI.length; t++) {
    jump[t] = (PI[t] + PI[t + 1] + PI[t + 2]) / 3 - (PI[t - 1] + PI[t - 2] + PI[t - 3]) / 3;
  }
  const out = [0];
  for (let t = 3; t + 2 < PI.length; t++) {
    if (jump[t] < thr) continue;
    let best = true;
    for (let q = Math.max(3, t - 3); q <= Math.min(PI.length - 3, t + 3); q++) if (jump[q] > jump[t]) best = false;
    if (best && t - out[out.length - 1] >= 4) out.push(t);
  }
  return out;
}

export interface ClassicFit {
  c0: number; r: number; Rd: number; seR: number; seRd: number;
  confounded: boolean; sum: number; // r + Rd/365 when confounded
  cleanings: number[];
}

/** Least squares for log PI = c0 − r (t − t_k) − d t/365. */
export function fitSoiling(PI: ArrayLike<number>, cleanings: number[]): ClassicFit {
  const last = lastCleaning(cleanings);
  const X: [number, number, number][] = [], y: number[] = [];
  for (let t = 0; t < DAYS; t++) {
    if (!(PI[t] > 0)) continue;
    X.push([1, -(t - last[t]), -t / 365]);
    y.push(Math.log(PI[t]));
  }
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  for (let i = 0; i < X.length; i++) for (let p = 0; p < 3; p++) {
    b[p] += X[i][p] * y[i];
    for (let q = 0; q < 3; q++) A[p][q] += X[i][p] * X[i][q];
  }
  const inv = inv3(A);
  // confounding: correlation between the soiling and degradation columns (after centring)
  const n = X.length, m1 = A[0][1] / n, m2 = A[0][2] / n;
  const v1 = A[1][1] / n - m1 * m1, v2 = A[2][2] / n - m2 * m2, cv = A[1][2] / n - m1 * m2;
  const corr = Math.abs(cv) / Math.sqrt(Math.max(v1 * v2, 1e-300));
  const confounded = !inv || corr > 0.9995;
  if (confounded) {
    // fit log PI = c0 − s t (one combined decline rate per day)
    let st = 0, stt = 0, sy = 0, sty = 0;
    for (let i = 0; i < n; i++) { const t = -X[i][1]; st += t; stt += t * t; sy += y[i]; sty += t * y[i]; }
    const s = -(n * sty - st * sy) / (n * stt - st * st);
    return { c0: (sy + s * st) / n, r: NaN, Rd: NaN, seR: NaN, seRd: NaN, confounded: true, sum: s, cleanings };
  }
  const c = [0, 0, 0];
  for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) c[p] += inv![p][q] * b[q];
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (y[i] - (c[0] * X[i][0] + c[1] * X[i][1] + c[2] * X[i][2])) ** 2;
  const s2 = rss / Math.max(1, n - 3);
  return {
    c0: c[0], r: c[1], Rd: c[2], seR: Math.sqrt(s2 * inv![1][1]), seRd: Math.sqrt(s2 * inv![2][2]),
    confounded: false, sum: c[1] + c[2] / 365, cleanings,
  };
}

function inv3(A: number[][]): number[][] | null {
  const [a, b, c] = A[0], [d, e, f] = A[1], [g, h, i] = A[2];
  const C = [[e * i - f * h, c * h - b * i, b * f - c * e], [f * g - d * i, a * i - c * g, c * d - a * f], [d * h - e * g, b * g - a * h, a * e - b * d]];
  const det = a * C[0][0] + b * C[1][0] + c * C[2][0];
  const scale = Math.abs(a * e * i) + 1e-300;
  if (Math.abs(det) < 1e-12 * scale) return null;
  return C.map((row) => row.map((v) => v / det));
}

/** Soiling ratio implied by a rate and a cleaning log. */
export function soilingSeries(r: number, cleanings: number[]): Float64Array {
  const last = lastCleaning(cleanings), out = new Float64Array(DAYS);
  for (let t = 0; t < DAYS; t++) out[t] = Math.exp(-r * (t - last[t]));
  return out;
}

/** Energy lost to soiling (MWh) given clean energy, degradation and a soiling series. */
export function soilingLoss(Eclean: ArrayLike<number>, D: ArrayLike<number>, SR: ArrayLike<number>) {
  let lost = 0, total = 0;
  for (let t = 0; t < DAYS; t++) { const e = Eclean[t] * D[t]; total += e; lost += e * (1 - SR[t]); }
  return { lost, total, frac: lost / total };
}

/**
 * Annual cost of a cleaning policy with interval N days: cleanings plus lost revenue.
 * Mean soiling loss over an interval with exponential soiling: 1 − (1 − e^{−rN})/(rN).
 * cost in USD per cleaning, price in USD/MWh, annual energy in MWh.
 */
export function annualCost(N: number, r: number, costPerCleaning: number, price: number, annualMWh: number): number {
  const x = r * N;
  const loss = x < 1e-9 ? x / 2 : 1 - (1 - Math.exp(-x)) / x;
  return (365 / N) * costPerCleaning + price * annualMWh * loss;
}

export function optimalInterval(r: number, costPerCleaning: number, price: number, annualMWh: number): number {
  let best = 1, bestC = Infinity;
  for (let N = 1; N <= 365; N++) {
    const c = annualCost(N, r, costPerCleaning, price, annualMWh);
    if (c < bestC) { bestC = c; best = N; }
  }
  return best;
}
