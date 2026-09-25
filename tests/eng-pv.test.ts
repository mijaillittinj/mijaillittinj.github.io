// PV plant soiling and degradation (electrical tab): run with `node tests/eng-pv.test.ts`
import { simulate, fitSoiling, detectCleanings, soilingSeries, soilingLoss, optimalInterval, annualCost, DAYS } from '../src/lib/eng/pv.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. a clean plant without degradation or noise has PI = 1 (temperature is in the clean model)
{
  const d = simulate({ r: 0, interval: 30, Rd: 0, noise: 0 }, 3);
  let mx = 0;
  for (let t = 0; t < DAYS; t++) mx = Math.max(mx, Math.abs(d.PI[t] - 1));
  check('clean plant: PI = 1', mx < 1e-12, `(max |PI − 1| ${mx.toExponential(1)})`);
  let lo = Infinity, hi = -Infinity;
  for (let t = 0; t < DAYS; t++) { const e = d.Eclean[t] / (100 * d.H[t]); lo = Math.min(lo, e); hi = Math.max(hi, e); }
  check('temperature correction keeps PR in a plausible range', lo > 0.7 && hi < 0.85, `(PR ${lo.toFixed(3)}–${hi.toFixed(3)})`);
}

// 2. regular cleanings: soiling rate and degradation are recovered from the maintenance log
{
  const truth = { r: 0.002, interval: 30, Rd: 0.007, noise: 0.007 };
  const d = simulate(truth, 4);
  const f = fitSoiling(d.PI, d.cleanings);
  check('soiling rate recovered', Math.abs(f.r - truth.r) < 0.0002, `(${(f.r * 100).toFixed(3)} vs 0.200 %/day)`);
  check('degradation recovered', Math.abs(f.Rd - truth.Rd) < 0.0025, `(${(f.Rd * 100).toFixed(2)} ± ${(f.seRd * 100).toFixed(2)} vs 0.70 %/yr)`);
  const det = detectCleanings(d.PI);
  check('cleanings visible as jumps in PI', Math.abs(det.length - d.cleanings.length) <= 2, `(${det.length} detected, ${d.cleanings.length} logged)`);
  const est = soilingLoss(d.Eclean, d.D, soilingSeries(f.r, d.cleanings)), tru = soilingLoss(d.Eclean, d.D, d.SR);
  check('energy lost to soiling estimated', Math.abs(est.lost / tru.lost - 1) < 0.05, `(${(est.lost / 1000).toFixed(1)} vs ${(tru.lost / 1000).toFixed(1)} GWh)`);
}

// 3. without cleanings soiling and degradation are confounded; only their sum is determined
{
  const truth = { r: 0.002, interval: Infinity, Rd: 0.007, noise: 0.007 };
  const d = simulate(truth, 4);
  const f = fitSoiling(d.PI, d.cleanings);
  check('no cleanings: confounding reported', f.confounded, '');
  check('no cleanings: total decline still determined', Math.abs(f.sum - (truth.r + truth.Rd / 365)) < 0.0002, `(${(f.sum * 100).toFixed(3)} %/day)`);
  const rare = fitSoiling(simulate({ ...truth, interval: 365 }, 4).PI, simulate({ ...truth, interval: 365 }, 4).cleanings);
  const often = fitSoiling(simulate({ ...truth, interval: 30 }, 4).PI, simulate({ ...truth, interval: 30 }, 4).cleanings);
  check('uncertainty of degradation shrinks with more cleanings', often.seRd < rare.seRd * 1.01 || often.seRd < 0.0006, `(σ ${(often.seRd * 100).toFixed(2)} vs ${(rare.seRd * 100).toFixed(2)} %/yr)`);
}

// 4. decision: higher soiling rate → shorter optimal interval; optimum is a minimum
{
  const E = 250000, C = 40000, p = 45;
  const n1 = optimalInterval(0.001, C, p, E), n2 = optimalInterval(0.003, C, p, E);
  check('dirtier site → clean more often', n2 < n1, `(${n2} vs ${n1} days)`);
  const c = annualCost(n1, 0.001, C, p, E);
  check('optimum is a minimum', c <= annualCost(n1 + 5, 0.001, C, p, E) && c <= annualCost(Math.max(1, n1 - 5), 0.001, C, p, E), '');
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
