// Checks for the chaos and data-assimilation module: run with `node tests/learn-lorenz.test.ts`
import { rk4, rhs, onAttractor, dist, twin, type Vec3 } from '../src/lib/lorenz.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. RK4 is fourth-order: halving dt reduces the error at t = 0.5 by ~16
{
  const s0: Vec3 = [1, 1, 20];
  const run = (dt: number) => { let s = s0; for (let i = 0; i < Math.round(0.5 / dt); i++) s = rk4(s, dt); return s; };
  const ref = run(1e-4), e1 = dist(run(0.01), ref), e2 = dist(run(0.005), ref);
  check('RK4 converges at fourth order', e1 / e2 > 12 && e1 / e2 < 20, `(ratio ${(e1 / e2).toFixed(1)})`);
}

// 2. Butterfly effect: a 1e-6 perturbation grows to O(1) within ~20 time units, rate ≈ λ ≈ 0.9
{
  const dt = 0.01;
  let a = onAttractor(1, dt), b: Vec3 = [a[0] + 1e-6, a[1], a[2]];
  let tHit = NaN;
  const logs: number[] = [];
  for (let k = 1; k <= 3000; k++) {
    a = rk4(a, dt); b = rk4(b, dt);
    const d = dist(a, b);
    if (k % 100 === 0) logs.push(Math.log(d));
    if (Number.isNaN(tHit) && d > 1) tHit = k * dt;
  }
  check('nearby trajectories separate to distance 1 in 10–30 time units', tHit > 10 && tHit < 30, `(t = ${tHit.toFixed(1)})`);
  // growth rate over the first 10 units (average)
  const rate = (logs[9] - Math.log(1e-6)) / 10;
  check('average growth rate is of order the Lyapunov exponent (0.5–1.5)', rate > 0.5 && rate < 1.5, `(${rate.toFixed(2)} per unit)`);
  void rhs;
}

// 3. EnKF beats the free-running ensemble
{
  const r = twin({ steps: 2000, dt: 0.01, every: 10, r: 1, idx: [0], members: 30, seed: 2 });
  check('EnKF (x observed) RMSE < free-run RMSE', r.rmseDa < 0.5 * r.rmseFree, `(${r.rmseDa.toFixed(2)} vs ${r.rmseFree.toFixed(2)})`);
  const r3 = twin({ steps: 2000, dt: 0.01, every: 10, r: 1, idx: [0, 1, 2], members: 30, seed: 2 });
  check('observing x, y, z gives a smaller error than x alone', r3.rmseDa < r.rmseDa, `(${r3.rmseDa.toFixed(2)} vs ${r.rmseDa.toFixed(2)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
