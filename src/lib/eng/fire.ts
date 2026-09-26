/**
 * Forest fire vs firefighters (home-figure tab "Fire safety").
 *
 * Forward model: fire spread on a grid of 100 m cells. The time for the fire to pass from
 * a burning cell i to a neighbour j is d_ij / R_ij, with a rate of spread
 *     R_ij = R0(fuel_j) · h_j · exp(c1 V) · exp(c2 V (cos θ_ij − 1)),
 * where V is the wind speed (m/s), θ_ij the angle between the spread direction and the
 * wind, and h_j a random cell-to-cell variation of the fuel (seeded). The wind factor is the
 * form used in common cellular-automaton wildfire models (c1 = 0.045, c2 = 0.131 s/m).
 * Arrival times follow from a shortest-path computation (Dijkstra), and a cell burns if the
 * fire reaches it before the suppression horizon T. Water and firebreaks do not burn; roads
 * slow the fire down.
 *
 * The game: the fire (adversary) chooses the ignition point and the wind direction among a
 * set of plausible scenarios S to maximise the damage D; the firefighters choose where to
 * build a fixed length of firebreak (plan b) to minimise the worst case:
 *     min_b max_{s ∈ S} D(b, s).
 * It is solved by alternating best responses (a double-oracle loop): the fire finds its
 * worst scenario against the current plan; the firefighters improve their plan against all
 * scenarios found so far. The firefighters' move uses a surrogate model learned from
 * simulations (ridge regression of the damage on the firebreak segments chosen), and every
 * candidate plan is then checked with the full simulator.
 */
import { rng } from '../linalg.ts';

export const N = 48;              // grid size (cells), 100 m per cell → 4.8 km × 4.8 km
export const HOUSE_WEIGHT = 40;   // damage of one house, in hectare-equivalents

export const Fuel = { Bare: 0, Grass: 1, Forest: 2, Water: 3, House: 4 } as const;
/** Rate of spread without wind, in cells per hour. */
const R0 = [0.7, 6, 3, 0, 2.5];
const C1 = 0.045, C2 = 0.131;

export interface Landscape {
  fuel: Uint8Array;        // N*N fuel types
  hetero: Float32Array;    // cell-to-cell variation of the fuel (≈ 0.7–1.3)
  houses: number[];        // indices of house cells
  ignitions: number[];     // plausible ignition points (roadsides, edges)
}

const idx = (x: number, y: number) => y * N + x;

/** Synthetic landscape: forest with grassland patches, a river ending in a lake, a road and a village. */
export function makeLandscape(seed = 7): Landscape {
  const R = rng(seed);
  const fuel = new Uint8Array(N * N).fill(Fuel.Forest);
  // grassland patches: smooth random field from a few Gaussian bumps
  const bumps = Array.from({ length: 9 }, () => ({ x: R.uni() * N, y: R.uni() * N, s: 3 + R.uni() * 5, a: R.uni() < 0.5 ? 1 : -0.7 }));
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    let v = 0;
    for (const b of bumps) v += b.a * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / (2 * b.s * b.s));
    if (v > 0.35) fuel[idx(x, y)] = Fuel.Grass;
  }
  // river from the top edge down to a lake (a natural but incomplete barrier)
  for (let y = 0; y < 27; y++) {
    const cx = Math.round(27 + 3 * Math.sin(y / 4.5));
    for (let x = cx; x <= cx + (y % 7 === 0 ? 2 : 1); x++) fuel[idx(x, y)] = Fuel.Water;
  }
  for (let y = 25; y < 33; y++) for (let x = 22; x < 34; x++) {
    if (((x - 28) / 6) ** 2 + ((y - 29) / 3.6) ** 2 <= 1) fuel[idx(x, y)] = Fuel.Water;
  }
  // east–west road with a bridge over the river
  for (let x = 0; x < N; x++) fuel[idx(x, 11)] = Fuel.Bare;
  // village in the south-east: houses in gardens (grass)
  const houses: number[] = [];
  for (let y = 34; y < 45; y++) for (let x = 34; x < 46; x++) fuel[idx(x, y)] = Fuel.Grass;
  const hp = [[36, 36], [39, 36], [42, 36], [37, 39], [40, 39], [43, 39], [36, 42], [39, 42], [42, 42], [44, 37], [44, 42], [38, 44]];
  for (const [x, y] of hp) { fuel[idx(x, y)] = Fuel.House; houses.push(idx(x, y)); }
  // village access road
  for (let y = 11; y < 34; y++) fuel[idx(40, y)] = Fuel.Bare;
  const hetero = new Float32Array(N * N);
  for (let k = 0; k < N * N; k++) hetero[k] = 0.7 + 0.6 * R.uni();
  // plausible ignition points: along the road, near the west edge and in the grassland
  const ignitions = [idx(6, 12), idx(18, 10), idx(34, 12), idx(8, 30), idx(15, 42), idx(24, 40)];
  for (const i of ignitions) if (fuel[i] === Fuel.Water || fuel[i] === Fuel.Bare) fuel[i] = Fuel.Grass;
  return { fuel, hetero, houses, ignitions };
}

/** Wind "from" a compass bearing (degrees, 0 = north, 90 = east) → unit vector the wind blows towards (grid y points down). */
export function windVector(fromDeg: number): [number, number] {
  const a = ((fromDeg + 180) * Math.PI) / 180;
  return [Math.sin(a), -Math.cos(a)];
}

export interface Scenario { ignition: number; windFrom: number }

const NB: [number, number][] = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];

/**
 * Arrival time of the fire (hours) at every cell, from one or more sources with given start
 * times. `blocked` marks firebreak cells (do not burn). Cells never reached get Infinity.
 */
export function arrivalTimes(L: Landscape, blocked: Uint8Array | null, sources: { i: number; t: number }[], windSpeed: number, windFrom: number, tMax = Infinity): Float64Array {
  const t = new Float64Array(N * N).fill(Infinity);
  const [wx, wy] = windVector(windFrom);
  const base = Math.exp(C1 * windSpeed);
  const wf = NB.map(([dx, dy]) => { const d = Math.hypot(dx, dy); return base * Math.exp(C2 * windSpeed * ((dx * wx + dy * wy) / d - 1)); });
  const dist = NB.map(([dx, dy]) => Math.hypot(dx, dy));
  // binary heap of (time, index)
  const hk = new Float64Array(N * N * 8 + 16), hv = new Int32Array(N * N * 8 + 16);
  let hs = 0;
  const push = (k: number, v: number) => {
    let c = hs++;
    while (c > 0) { const p = (c - 1) >> 1; if (hk[p] <= k) break; hk[c] = hk[p]; hv[c] = hv[p]; c = p; }
    hk[c] = k; hv[c] = v;
  };
  const pop = () => {
    const v = hv[0], k = hk[0];
    const lk = hk[--hs], lv = hv[hs];
    let c = 0;
    for (;;) { let m = 2 * c + 1; if (m >= hs) break; if (m + 1 < hs && hk[m + 1] < hk[m]) m++; if (hk[m] >= lk) break; hk[c] = hk[m]; hv[c] = hv[m]; c = m; }
    hk[c] = lk; hv[c] = lv;
    return [k, v] as const;
  };
  for (const s of sources) if (s.t < t[s.i]) { t[s.i] = s.t; push(s.t, s.i); }
  while (hs > 0) {
    const [ti, i] = pop();
    if (ti > t[i] || ti > tMax) continue;
    const x = i % N, y = (i / N) | 0;
    for (let k = 0; k < 8; k++) {
      const nx = x + NB[k][0], ny = y + NB[k][1];
      if (nx < 0 || ny < 0 || nx >= N || ny >= N) continue;
      const j = ny * N + nx;
      if (blocked && blocked[j]) continue;
      const r = R0[L.fuel[j]];
      if (r <= 0) continue;
      const tj = ti + dist[k] / (r * L.hetero[j] * wf[k]);
      if (tj < t[j]) { t[j] = tj; push(tj, j); }
    }
  }
  return t;
}

export interface Damage { burned: number; houses: number; value: number }

/** Damage of a fire: burned hectares + HOUSE_WEIGHT × houses lost, for cells reached before the horizon. */
export function damage(L: Landscape, t: Float64Array, horizon: number): Damage {
  let burned = 0, houses = 0;
  for (let k = 0; k < t.length; k++) if (t[k] <= horizon) { burned++; if (L.fuel[k] === Fuel.House) houses++; }
  return { burned, houses, value: burned + HOUSE_WEIGHT * houses };
}

export interface Params { windSpeed: number; horizon: number }

export function scenarioDamage(L: Landscape, blocked: Uint8Array | null, s: Scenario, p: Params): Damage {
  if (blocked && blocked[s.ignition]) return { burned: 0, houses: 0, value: 0 };
  return damage(L, arrivalTimes(L, blocked, [{ i: s.ignition, t: 0 }], p.windSpeed, s.windFrom, p.horizon), p.horizon);
}

/** All scenarios: every plausible ignition point × eight wind directions. */
export function allScenarios(L: Landscape): Scenario[] {
  const out: Scenario[] = [];
  for (const ignition of L.ignitions) for (let w = 0; w < 8; w++) out.push({ ignition, windFrom: w * 45 });
  return out;
}

/* ------------------------------------------------------------------ plans */

/** A firebreak segment: SEG cells from (x, y) in direction (dx, dy). */
export interface Segment { x: number; y: number; dx: number; dy: number }
export const SEG = 5;

export function segmentCells(s: Segment): number[] {
  const out: number[] = [];
  for (let k = 0; k < SEG; k++) {
    const x = s.x + k * s.dx, y = s.y + k * s.dy;
    if (x >= 0 && y >= 0 && x < N && y < N) out.push(idx(x, y));
  }
  return out;
}

/** True if a firebreak may be built on cell c: not a house, and not within 200 m of an ignition point
 * (the firefighters know where fires can start, not which one will; blocking the source itself would make the game trivial). */
export function buildable(L: Landscape, c: number): boolean {
  if (L.fuel[c] === Fuel.House) return false;
  const x = c % N, y = (c / N) | 0;
  for (const i of L.ignitions) if (Math.max(Math.abs((i % N) - x), Math.abs(((i / N) | 0) - y)) <= 2) return false;
  return true;
}

export function planMask(L: Landscape, plan: Segment[]): Uint8Array {
  const m = new Uint8Array(N * N);
  for (const s of plan) for (const c of segmentCells(s)) if (buildable(L, c)) m[c] = 1;
  return m;
}

export function worstCase(L: Landscape, plan: Segment[], scenarios: Scenario[], p: Params) {
  const m = planMask(L, plan);
  let best = { s: scenarios[0], d: { burned: 0, houses: 0, value: -1 } as Damage };
  let sum = 0;
  for (const s of scenarios) {
    const d = scenarioDamage(L, m, s, p);
    sum += d.value;
    if (d.value > best.d.value) best = { s, d };
  }
  return { worst: best.s, damage: best.d, mean: sum / scenarios.length };
}

/** Naive plan: firebreak segments ringing the village (same budget). */
export function naivePlan(nSeg: number): Segment[] {
  const ring: Segment[] = [
    { x: 33, y: 33, dx: 1, dy: 0 }, { x: 38, y: 33, dx: 1, dy: 0 }, { x: 43, y: 33, dx: 1, dy: 0 },
    { x: 33, y: 34, dx: 0, dy: 1 }, { x: 33, y: 39, dx: 0, dy: 1 }, { x: 33, y: 44, dx: 0, dy: 1 },
    { x: 34, y: 45, dx: 1, dy: 0 }, { x: 46, y: 34, dx: 0, dy: 1 }, { x: 39, y: 45, dx: 1, dy: 0 }, { x: 46, y: 39, dx: 0, dy: 1 },
  ];
  return ring.slice(0, nSeg);
}

/* ------------------------------------------------------------------ the game */

const DIRS: [number, number][] = [[1, 0], [0, 1], [1, 1], [1, -1]];

/**
 * Candidate segments: centred on cells the fire reaches in the given scenarios, oriented
 * across the local direction of spread (from the arrival-time gradient) or at random.
 */
function candidates(L: Landscape, scen: Scenario[], p: Params, seed: number, n = 160): Segment[] {
  const R = rng(seed);
  const pts: { c: number; dir: [number, number] }[] = [];
  for (const s of scen) {
    const t = arrivalTimes(L, null, [{ i: s.ignition, t: 0 }], p.windSpeed, s.windFrom, p.horizon);
    for (let k = 0; k < t.length; k++) {
      if (!(t[k] <= p.horizon) || t[k] < 0.4 || L.fuel[k] === Fuel.House) continue;
      const x = k % N, y = (k / N) | 0;
      const g = (a: number, b: number) => (Number.isFinite(a) && Number.isFinite(b) ? a - b : 0);
      const gx = x > 0 && x < N - 1 ? g(t[k + 1], t[k - 1]) : 0;
      const gy = y > 0 && y < N - 1 ? g(t[k + N], t[k - N]) : 0;
      // segment direction perpendicular to the spread direction (gx, gy), snapped to 45°
      const a = Math.atan2(gx, -gy); // perpendicular angle
      const oct = ((Math.round(a / (Math.PI / 4)) % 4) + 4) % 4;
      pts.push({ c: k, dir: DIRS[[0, 2, 1, 3][oct]] });
    }
  }
  const out: Segment[] = [], seen = new Set<string>();
  for (let tries = 0; out.length < n && tries < n * 30 && pts.length; tries++) {
    const q = pts[Math.floor(R.uni() * pts.length)];
    const [dx, dy] = R.uni() < 0.75 ? q.dir : DIRS[Math.floor(R.uni() * 4)];
    const x0 = (q.c % N) - 2 * dx, y0 = ((q.c / N) | 0) - 2 * dy;
    const key = `${x0},${y0},${dx},${dy}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ x: x0, y: y0, dx, dy });
  }
  return out;
}

/** Ridge regression y ≈ w0 + X w (X binary, n × m), returns w (length m). */
function ridge(X: Uint8Array[], y: number[], m: number, lambda: number): Float64Array {
  const n = y.length;
  const mean = y.reduce((a, b) => a + b, 0) / n;
  // normal equations (m ≤ ~150): (XᵀX + λI) w = Xᵀ(y − ȳ), solved by Cholesky
  const A = new Float64Array(m * m), b = new Float64Array(m);
  for (let r = 0; r < n; r++) {
    const on: number[] = [];
    for (let c = 0; c < m; c++) if (X[r][c]) on.push(c);
    for (const c of on) { b[c] += y[r] - mean; for (const d of on) A[c * m + d] += 1; }
  }
  for (let c = 0; c < m; c++) A[c * m + c] += lambda;
  const Lc = new Float64Array(m * m);
  for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) {
    let s = A[i * m + j];
    for (let k = 0; k < j; k++) s -= Lc[i * m + k] * Lc[j * m + k];
    Lc[i * m + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / Lc[j * m + j];
  }
  const z = new Float64Array(m), w = new Float64Array(m);
  for (let i = 0; i < m; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= Lc[i * m + k] * z[k]; z[i] = s / Lc[i * m + i]; }
  for (let i = m - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < m; k++) s -= Lc[k * m + i] * w[k]; w[i] = s / Lc[i * m + i]; }
  return w;
}

export interface GameRound {
  round: number;
  plan: Segment[];          // best plan found so far
  worst: Scenario;          // fire's worst scenario against it
  worstDamage: Damage;      // worst case over ALL scenarios (upper value of the game for this plan)
  knownDamage: number;      // this round's plan against the scenarios found so far (what the firefighters optimised)
  trialWorst: number;       // this round's plan against ALL scenarios (the fire's best reply)
  known: Scenario[];
  done: boolean;
}

/**
 * Alternating best responses, written as a generator so the page can run it in short
 * slices: it yields `null` every few simulations and a GameRound at the end of each round.
 * It starts from the naive plan and keeps only improvements, so the worst-case damage of
 * the best plan never increases.
 */
export function* minimaxGame(L: Landscape, p: Params, nSeg: number, seed = 1, maxRounds = 8): Generator<GameRound | null> {
  const S = allScenarios(L);
  const key = (s: Scenario) => `${s.ignition}:${s.windFrom}`;
  let best = naivePlan(nSeg);
  let wc = worstCase(L, best, S, p);
  const known: Scenario[] = [wc.worst];
  let sims = 0;
  function* knownDamage(plan: Segment[], out: { v: number }) {
    const m = planMask(L, plan);
    let mx = 0;
    for (const s of known) { mx = Math.max(mx, scenarioDamage(L, m, s, p).value); if (++sims % 6 === 0) yield null; }
    out.v = mx;
  }
  const o = { v: 0 };
  yield { round: 0, plan: best, worst: wc.worst, worstDamage: wc.damage, knownDamage: 0, trialWorst: wc.damage.value, known: [...known], done: false };
  const R = rng(seed);
  let stale = 0;
  for (let round = 1; round <= maxRounds; round++) {
    // firefighters' move, step 1: a surrogate learned from simulated random plans
    const C = [...best, ...candidates(L, known, p, seed + round)];
    const m = C.length;
    const X: Uint8Array[] = [], y: number[] = [];
    for (let r = 0; r < 60; r++) {
      const x = new Uint8Array(m), plan: Segment[] = [];
      while (plan.length < Math.min(nSeg, m)) { const c = Math.floor(R.uni() * m); if (!x[c]) { x[c] = 1; plan.push(C[c]); } }
      yield* knownDamage(plan, o);
      X.push(x); y.push(o.v);
    }
    const w = ridge(X, y, m, 2);
    const order = [...w.keys()].sort((a, b) => w[a] - w[b]); // most protective first
    // step 2: greedy construction, screening the candidates the surrogate ranks best and
    // checking each with the full simulator
    let greedy: Segment[] = [];
    const used = new Set<number>();
    for (let k = 0; k < nSeg; k++) {
      let bestC = -1, bestV = Infinity, tested = 0;
      for (const c of order) {
        if (used.has(c)) continue;
        if (++tested > 40) break;
        yield* knownDamage([...greedy, C[c]], o);
        if (o.v < bestV) { bestV = o.v; bestC = c; }
      }
      if (bestC < 0) break;
      used.add(bestC); greedy.push(C[bestC]);
    }
    // step 3: local search from the greedy plan and from the best plan so far
    const pool = order.slice(0, 24);
    let plan: Segment[] = [], val = Infinity;
    for (const start of [greedy, best]) {
      let cur = start.slice();
      yield* knownDamage(cur, o);
      let cv = o.v;
      for (let pass = 0; pass < 2; pass++) for (let i = 0; i < cur.length; i++) for (const c of pool) {
        const trial = cur.slice(); trial[i] = C[c];
        yield* knownDamage(trial, o);
        if (o.v < cv) { cv = o.v; cur = trial; }
      }
      if (cv < val) { val = cv; plan = cur; }
    }
    // fire's move: its worst scenario against the new plan
    const wcNew = worstCase(L, plan, S, p);
    const improved = wcNew.damage.value < wc.damage.value;
    if (improved) { best = plan; wc = wcNew; }
    const isNew = !known.some((s) => key(s) === key(wcNew.worst));
    if (isNew) known.push(wcNew.worst);
    stale = isNew || improved ? 0 : stale + 1;
    const done = stale >= 2 || round === maxRounds;
    yield { round, plan: best, worst: wc.worst, worstDamage: wc.damage, knownDamage: val, trialWorst: wcNew.damage.value, known: [...known], done };
    if (done) return;
  }
}

/** Run the whole game synchronously (tests, Node). */
export function solveGame(L: Landscape, p: Params, nSeg: number, seed = 1, maxRounds = 8): GameRound[] {
  const rounds: GameRound[] = [];
  for (const r of minimaxGame(L, p, nSeg, seed, maxRounds)) if (r) rounds.push(r);
  return rounds;
}
