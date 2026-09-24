// Checks for the eikonal PINN of the civil tab: run with `node tests/pinn-civil.test.ts`
import { initQuakeNet, quakeLossGrad, getParams, setParams, collocation, makeQuakeTrainer, travelTime, nParams } from '../src/lib/eng/pinn-quake.ts';
import { arrivalTimes, misfitMap, hiddenQuake, V_P } from '../src/lib/eng/quake.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const L = 200;
const stations = [{ x: 30, y: 40 }, { x: 170, y: 30 }, { x: 100, y: 172 }, { x: 36, y: 150 }, { x: 165, y: 140 }, { x: 105, y: 88 }];
const q = hiddenQuake(3);
const times = arrivalTimes(stations, q.src, q.t0).map((t, i) => t + 0.1 * q.draws[i]);
const col = collocation(200, L, 5);

// 1. analytic gradient vs central finite differences (all parameter groups)
{
  const net = initQuakeNet(6, 2, { x: 90, y: 110 }, 3);
  for (let j = 0; j < 6; j++) net.a[j] = 0.2 * Math.sin(j + 1); // non-trivial correction network
  const D = { stations, times, col: col.slice(0, 40), L, lambda: 3 };
  const { grad } = quakeLossGrad(net, D);
  const p0 = getParams(net);
  let maxRel = 0;
  for (let k = 0; k < nParams(6); k++) {
    const h = 1e-6 * Math.max(1, Math.abs(p0[k]));
    const p = Float64Array.from(p0);
    p[k] = p0[k] + h; setParams(net, p); const lp = quakeLossGrad(net, D).loss;
    p[k] = p0[k] - h; setParams(net, p); const lm = quakeLossGrad(net, D).loss;
    const fd = (lp - lm) / (2 * h);
    maxRel = Math.max(maxRel, Math.abs(fd - grad[k]) / Math.max(1e-4, Math.abs(fd)));
  }
  setParams(net, p0);
  check('eikonal PINN gradient matches finite differences', maxRel < 1e-4, `(max rel err ${maxRel.toExponential(2)})`);
}

// 2. training locates the hidden epicentre and satisfies the eikonal equation
{
  const net = initQuakeNet(24, 1, { x: 103, y: 105 }, 0);
  net.t0 = times.reduce((s, t, i) => s + t - Math.hypot(stations[i].x - 103, stations[i].y - 105) / V_P, 0) / times.length;
  const D = { stations, times, col, L };
  const tr = makeQuakeTrainer(net, D, 1500);
  const t0 = performance.now();
  let r = quakeLossGrad(net, D);
  for (let k = 0; k < 1500; k++) r = tr.step();
  const ms = performance.now() - t0;
  const cls = misfitMap(stations, times, 100, L).best;
  const eTrue = Math.hypot(net.xs - q.src.x, net.ys - q.src.y), eCls = Math.hypot(net.xs - cls.x, net.ys - cls.y);
  check('PINN epicentre within 3 km of the truth', eTrue < 3, `(${eTrue.toFixed(2)} km, ${ms.toFixed(0)} ms for 1500 steps)`);
  check('PINN agrees with the classical misfit search', eCls < 1.5, `(${eCls.toFixed(2)} km)`);
  check('eikonal residual is small', r.eik < 1e-3, `(${r.eik.toExponential(1)})`);
  check('origin time recovered within 0.3 s', Math.abs(net.t0 - q.t0) < 0.3, `(${net.t0.toFixed(2)} vs ${q.t0.toFixed(2)} s)`);
  const tt = travelTime(net, { x: 20, y: 180 }, L), exact = Math.hypot(20 - net.xs, 180 - net.ys) / V_P;
  check('learned travel time matches the homogeneous solution', Math.abs(tt - exact) < 0.05 * exact, `(${tt.toFixed(2)} vs ${exact.toFixed(2)} s)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
