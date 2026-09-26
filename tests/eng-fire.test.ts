// Checks for the forest-fire tab: run with `node tests/eng-fire.test.ts`
import { N, Fuel, makeLandscape, arrivalTimes, allScenarios, worstCase, naivePlan, solveGame, planMask, buildable, type Landscape } from '../src/lib/eng/fire.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// a uniform forest with no random variation, for clean geometric checks
const uniform = (): Landscape => ({ fuel: new Uint8Array(N * N).fill(Fuel.Forest), hetero: new Float32Array(N * N).fill(1), houses: [], ignitions: [] });
const at = (x: number, y: number) => y * N + x;

// 1. Water stops the fire: a full-height river column is never crossed
{
  const L = uniform();
  for (let y = 0; y < N; y++) L.fuel[at(24, y)] = Fuel.Water;
  const t = arrivalTimes(L, null, [{ i: at(10, 24), t: 0 }], 10, 270);
  let crossed = 0;
  for (let y = 0; y < N; y++) for (let x = 25; x < N; x++) if (Number.isFinite(t[at(x, y)])) crossed++;
  check('fire does not cross water', crossed === 0, `(${crossed} cells reached beyond the river)`);
}

// 2. Firebreaks do not burn; roads slow the fire
{
  const L = uniform();
  for (let x = 0; x < N; x++) L.fuel[at(x, 30)] = Fuel.Bare;
  const t = arrivalTimes(L, null, [{ i: at(24, 10), t: 0 }], 0, 0);
  const tf = arrivalTimes(uniform(), null, [{ i: at(24, 10), t: 0 }], 0, 0);
  check('a road slows the fire down', t[at(24, 40)] > tf[at(24, 40)], `(${t[at(24, 40)].toFixed(2)} h vs ${tf[at(24, 40)].toFixed(2)} h in forest)`);
  const blocked = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) blocked[at(35, y)] = 1;
  const t2 = arrivalTimes(uniform(), blocked, [{ i: at(20, 20), t: 0 }], 5, 270);
  check('a firebreak line stops the fire', !Number.isFinite(t2[at(40, 20)]));
}

// 3. Downwind spreads faster than upwind (wind from the west blows towards the east)
{
  const L = uniform();
  const t = arrivalTimes(L, null, [{ i: at(24, 24), t: 0 }], 8, 270);
  const east = t[at(34, 24)], west = t[at(14, 24)];
  check('downwind arrival is earlier than upwind', east < west / 3, `(east ${east.toFixed(2)} h, west ${west.toFixed(2)} h)`);
  const t0 = arrivalTimes(L, null, [{ i: at(24, 24), t: 0 }], 0, 270);
  check('no wind: spread is symmetric', Math.abs(t0[at(34, 24)] - t0[at(14, 24)]) < 1e-12);
}

// 4. The minimax loop never increases the worst-case damage of the best plan found,
//    and the robust plan is at least as good as the naive ring around the village
{
  const L = makeLandscape(7);
  const p = { windSpeed: 8, horizon: 4 };
  const rounds = solveGame(L, p, 5, 1);
  let mono = true;
  for (let k = 1; k < rounds.length; k++) if (rounds[k].worstDamage.value > rounds[k - 1].worstDamage.value) mono = false;
  check('best worst-case damage is non-increasing over rounds', mono, `(${rounds.map((r) => r.worstDamage.value).join(' → ')})`);
  const S = allScenarios(L);
  const naive = worstCase(L, naivePlan(5), S, p), robust = worstCase(L, rounds[rounds.length - 1].plan, S, p);
  check('robust plan worst case ≤ naive plan worst case', robust.damage.value <= naive.damage.value, `(robust ${robust.damage.value}, naive ${naive.damage.value})`);
  check('robust plan strictly improves on the naive plan here', robust.damage.value < naive.damage.value);
  // firebreaks are never built on houses or next to ignition points
  const m = planMask(L, rounds[rounds.length - 1].plan);
  let bad = 0;
  for (let c = 0; c < m.length; c++) if (m[c] && !buildable(L, c)) bad++;
  check('firebreaks respect the building rules', bad === 0);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
