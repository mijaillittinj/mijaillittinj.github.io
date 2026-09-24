/**
 * Non-intrusive load monitoring (energy disaggregation) for the electrical tab.
 *
 * Forward: each appliance i is on or off at every minute (s_i(t) ∈ {0, 1}) and draws a
 * constant power P_i when on; the smart meter records only
 *     P(t) = P_base + Σ_i P_i s_i(t) + ε(t).
 * Inverse: recover every s_i(t) from P(t). Each appliance is a two-state Markov chain whose
 * switching probabilities encode its typical on/off durations; the joint chain (2^N states)
 * is decoded with the Viterbi algorithm (a factorial hidden Markov model).
 * With identical duration priors, two appliances of equal power (kettle and oven, 2 kW) can
 * only be told apart by chance: power alone does not identify them.
 * Synthetic day, idealised constant-power signatures.
 */
import { rng } from '../linalg.ts';

export interface Appliance {
  id: string;
  power: number;      // W when on
  meanOn: number;     // typical on-duration (min)
  meanOff: number;    // typical off-duration (min)
  colour: string;     // fill in the stacked plots
}

export const MINUTES = 1440;
export const BASE_LOAD = 40; // W, always-on standby (known)

export const APPLIANCES: Appliance[] = [
  { id: 'fridge', power: 120, meanOn: 18, meanOff: 35, colour: '#5b8fb9' },
  { id: 'lights', power: 180, meanOn: 150, meanOff: 600, colour: '#e0b74a' },
  { id: 'tv', power: 100, meanOn: 120, meanOff: 700, colour: '#8e6bb3' },
  { id: 'washer', power: 500, meanOn: 80, meanOff: 1300, colour: '#4f9a82' },
  { id: 'kettle', power: 2000, meanOn: 3, meanOff: 240, colour: '#c8553d' },
  { id: 'oven', power: 2000, meanOn: 45, meanOff: 900, colour: '#8a5a3c' },
];

/** Minute-resolved schedule of a synthetic day. Returns states[i][t] ∈ {0,1}. */
export function syntheticDay(seed: number, present: boolean[] = APPLIANCES.map(() => true)): Uint8Array[] {
  const R = rng(seed);
  const jit = (m: number) => Math.round(R.normal() * m);
  const S = APPLIANCES.map(() => new Uint8Array(MINUTES));
  const on = (i: number, a: number, len: number) => {
    for (let t = Math.max(0, a); t < Math.min(MINUTES, a + len); t++) S[i][t] = 1;
  };
  // fridge: compressor cycles all day
  for (let t = Math.floor(R.uni() * 30); t < MINUTES;) {
    const l = 15 + Math.round(R.uni() * 7);
    on(0, t, l);
    t += l + 30 + Math.round(R.uni() * 12);
  }
  // lights: morning and evening
  on(1, 390 + jit(10), 55 + jit(8));
  on(1, 1110 + jit(15), 290 + jit(15));
  // TV: lunch and evening
  on(2, 780 + jit(10), 40 + jit(5));
  on(2, 1140 + jit(10), 200 + jit(15));
  // washing machine: one cycle in the morning
  on(3, 540 + jit(30), 85);
  // kettle: short boils
  for (const t0 of [425, 630, 960, 1260]) on(4, t0 + jit(15), 3 + Math.round(R.uni()));
  // oven: lunch and dinner
  on(5, 720 + jit(10), 50 + jit(5));
  on(5, 1170 + jit(10), 40 + jit(5));
  present.forEach((p, i) => { if (!p) S[i].fill(0); });
  return S;
}

/** Aggregate meter reading with Gaussian noise of standard deviation `sigma` (W). */
export function meter(states: Uint8Array[], sigma: number, seed: number): Float64Array {
  const R = rng(seed), P = new Float64Array(MINUTES);
  for (let t = 0; t < MINUTES; t++) {
    let s = BASE_LOAD;
    for (let i = 0; i < APPLIANCES.length; i++) s += APPLIANCES[i].power * states[i][t];
    P[t] = s + sigma * R.normal();
  }
  return P;
}

/**
 * Viterbi decoding of the factorial HMM. `present` restricts the search to appliances the
 * user has in the house. With `durationPriors = false` every appliance gets the same
 * switching probabilities (a plain switching penalty), so equal-power appliances cannot be
 * distinguished.
 */
export function disaggregate(P: ArrayLike<number>, sigma: number, present: boolean[], durationPriors = true): Uint8Array[] {
  const idx = APPLIANCES.map((_, i) => i).filter((i) => present[i]);
  const n = idx.length, K = 1 << n, T = P.length;
  const s2 = 2 * Math.max(sigma, 15) ** 2;
  const powerOf = new Float64Array(K);
  for (let k = 0; k < K; k++) for (let b = 0; b < n; b++) if (k & (1 << b)) powerOf[k] += APPLIANCES[idx[b]].power;
  // per-appliance transition costs (−log p)
  const cost = idx.map((i) => {
    const a = APPLIANCES[i];
    const pOff = 1 / (durationPriors ? a.meanOn : 30), pOn = 1 / (durationPriors ? a.meanOff : 300);
    return { stayOff: -Math.log(1 - pOn), turnOn: -Math.log(pOn), stayOn: -Math.log(1 - pOff), turnOff: -Math.log(pOff) };
  });
  const trans = new Float64Array(K * K); // trans[prev * K + next]
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) {
    let c = 0;
    for (let j = 0; j < n; j++) {
      const pa = (a >> j) & 1, pb = (b >> j) & 1, cj = cost[j];
      c += pa ? (pb ? cj.stayOn : cj.turnOff) : (pb ? cj.turnOn : cj.stayOff);
    }
    trans[a * K + b] = c;
  }
  const back = new Uint8Array(T * K);
  let prev = new Float64Array(K), cur = new Float64Array(K);
  for (let k = 0; k < K; k++) prev[k] = (P[0] - BASE_LOAD - powerOf[k]) ** 2 / s2 + (k ? 5 : 0);
  for (let t = 1; t < T; t++) {
    const y = P[t] - BASE_LOAD;
    for (let b = 0; b < K; b++) {
      let best = Infinity, arg = 0;
      for (let a = 0; a < K; a++) {
        const v = prev[a] + trans[a * K + b];
        if (v < best) { best = v; arg = a; }
      }
      cur[b] = best + (y - powerOf[b]) ** 2 / s2;
      back[t * K + b] = arg;
    }
    [prev, cur] = [cur, prev];
  }
  let k = 0;
  for (let b = 1; b < K; b++) if (prev[b] < prev[k]) k = b;
  const out = APPLIANCES.map(() => new Uint8Array(T));
  for (let t = T - 1; t >= 0; t--) {
    for (let j = 0; j < n; j++) out[idx[j]][t] = (k >> j) & 1;
    if (t > 0) k = back[t * K + k];
  }
  return out;
}

/** Energy in kWh of each appliance for a set of states. */
export function energies(states: Uint8Array[]): number[] {
  return states.map((s, i) => {
    let m = 0;
    for (const v of s) m += v;
    return (APPLIANCES[i].power * m) / 60 / 1000;
  });
}

/** Fraction of appliance-minutes classified correctly, over the appliances present. */
export function accuracy(truth: Uint8Array[], est: Uint8Array[], present: boolean[]): number {
  let ok = 0, n = 0;
  truth.forEach((s, i) => {
    if (!present[i]) return;
    for (let t = 0; t < s.length; t++) { ok += +(s[t] === est[i][t]); n++; }
  });
  return n ? ok / n : 1;
}

/** Share of the energy of appliances that are ON assigned to the right appliance (energy-weighted). */
export function energyAccuracy(truth: Uint8Array[], est: Uint8Array[]): number {
  let err = 0, tot = 0;
  for (let t = 0; t < MINUTES; t++) {
    for (let i = 0; i < APPLIANCES.length; i++) {
      const p = APPLIANCES[i].power;
      err += Math.abs(truth[i][t] - est[i][t]) * p;
      tot += truth[i][t] * p;
    }
  }
  return tot ? Math.max(0, 1 - err / (2 * tot)) : 1;
}
