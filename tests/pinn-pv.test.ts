// PINN for PV soiling and degradation: run with `node tests/pinn-pv.test.ts`
import { simulate, DAYS } from '../src/lib/eng/pv.ts';
import { initPvPinn, pvLossGrad, pvAdam, rateOf, degradationOf, pinnSoiling } from '../src/lib/eng/pinn-pv.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. gradient check
{
  const d = simulate({ r: 0.002, interval: 45, Rd: 0.007, noise: 0.007 }, 2);
  const p = initPvPinn(4, 3, 0.15, 0.5);
  for (let q = 0; q < p.th.length; q++) p.th[q] += 0.05 * Math.sin(3 * q + 1); // move away from the flat start
  const { grad } = pvLossGrad(p, d.PI, d.cleanings);
  let mx = 0;
  for (let q = 0; q < p.th.length; q++) {
    const h = 1e-6, v = p.th[q];
    p.th[q] = v + h; const lp = pvLossGrad(p, d.PI, d.cleanings).loss; p.th[q] = v - h; const lm = pvLossGrad(p, d.PI, d.cleanings).loss; p.th[q] = v;
    const fd = (lp - lm) / (2 * h);
    mx = Math.max(mx, Math.abs(fd - grad[q]) / Math.max(1e-6, Math.abs(fd)));
  }
  check('PINN gradient matches finite differences', mx < 1e-4, `(${mx.toExponential(1)})`);
}

// 2. recovery with regular cleanings (same schedule as the browser)
for (const seed of [3, 4]) {
  const truth = { r: 0.002, interval: 30, Rd: 0.007, noise: 0.007 };
  const d = simulate(truth, seed);
  const p = initPvPinn(16, 5), N = 2500;
  for (let k = 0; k < N; k++) pvAdam(p, pvLossGrad(p, d.PI, d.cleanings).grad, 0.02 * (0.1 + 0.45 * (1 + Math.cos((Math.PI * k) / N))));
  const sr = pinnSoiling(p, d.cleanings);
  let e = 0;
  for (let t = 0; t < DAYS; t++) e += (sr[t] - d.SR[t]) ** 2;
  check(`seed ${seed}: soiling rate recovered`, Math.abs(rateOf(p) - 0.2) < 0.01, `(${rateOf(p).toFixed(3)} %/day)`);
  check(`seed ${seed}: degradation recovered`, Math.abs(degradationOf(p) - 0.7) < 0.2, `(${degradationOf(p).toFixed(2)} %/yr)`);
  check(`seed ${seed}: soiling ratio recovered`, Math.sqrt(e / DAYS) < 0.005, `(RMS ${Math.sqrt(e / DAYS).toFixed(4)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
