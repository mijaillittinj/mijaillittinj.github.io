// PINN versions of the home-figure problems: run with `node tests/pinn-eng.test.ts`
import { initFlameNet, flameLossGrad, flameAdam, evalFlameNet } from '../src/lib/eng/pinn-flame.ts';
import { initKinPinn, kinLossGrad, kinAdam, kOf } from '../src/lib/eng/pinn-kinetics.ts';
import { makeGrid, kernel, initGravNet, gravLossGrad, gravAdam, depthWeights, centroid } from '../src/lib/eng/pinn-gravity.ts';
import { trappingMatrix, sootEmission } from '../src/lib/sat.ts';
import { makeData } from '../src/lib/eng/kinetics.ts';
import { makeSurvey, lambdaOf } from '../src/lib/eng/gravity.ts';
import { matvec, rng } from '../src/lib/linalg.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fdCheck = (params: Float64Array[], grad: Float64Array, L: () => number) => {
  let mx = 0, k = 0;
  for (const arr of params) for (let j = 0; j < arr.length; j++, k++) {
    const h = 1e-6, v = arr[j];
    arr[j] = v + h; const lp = L(); arr[j] = v - h; const lm = L(); arr[j] = v;
    const fd = (lp - lm) / (2 * h);
    mx = Math.max(mx, Math.abs(fd - grad[k]) / Math.max(1e-7, Math.abs(fd)));
  }
  return mx;
};

// ---- flame: emission with self-absorption
{
  const n = 80;
  const r = Float64Array.from({ length: n }, (_, i) => (i + 0.5) / n);
  const fs = Float64Array.from(r, (x) => Math.exp(-(((x - 0.55) / 0.12) ** 2)) + 0.3 * Math.exp(-((x / 0.3) ** 2)));
  const J = Float64Array.from(fs, (f, i) => sootEmission(f, 1700 + 200 * Math.exp(-(((r[i] - 0.6) / 0.25) ** 2))));
  const jm = Math.max(...J); for (let i = 0; i < n; i++) J[i] /= jm;
  const A = trappingMatrix(n, Float64Array.from(fs, (f) => 1.2 * f));
  const g0 = matvec(A, J), gm = Math.max(...g0), R = rng(3);
  const g = Float64Array.from(g0, (v) => v + 0.01 * gm * R.normal());
  const small = initFlameNet(5, 1);
  const e = fdCheck([small.a, small.w, small.b, small.c], flameLossGrad(small, A, r, g).grad, () => flameLossGrad(small, A, r, g).loss);
  check('flame PINN gradient matches finite differences', e < 1e-4, `(${e.toExponential(1)})`);
  const net = initFlameNet(32, 5);
  for (let k = 0; k < 6000; k++) flameAdam(net, flameLossGrad(net, A, r, g).grad, 0.02);
  const f = evalFlameNet(net, r);
  let s = 0; for (let i = 0; i < n; i++) s += (f[i] - J[i]) ** 2;
  const err = Math.sqrt(s / n);
  check('flame PINN recovers the emission with trapping (RMS < 5 % of peak)', err < 0.05, `(${(err * 100).toFixed(1)} %)`);
}

// ---- kinetics: ODE-constrained network with trainable rate constants
{
  const data = makeData(0.6, 0.2, 0.03, 3);
  const cfg = { hidden: 5, lambdaOde: 0.7, scaleUnknown: true, measureA: true, nCol: 9 };
  const p = initKinPinn(5, 2, -0.3, -0.5);
  const e = fdCheck([p.th], kinLossGrad(p, data, cfg).grad, () => kinLossGrad(p, data, cfg).loss);
  check('kinetics PINN gradient matches finite differences', e < 1e-4, `(${e.toExponential(1)})`);
  const run = (scaleUnknown: boolean) => {
    const c = { hidden: 24, lambdaOde: 1, scaleUnknown, measureA: false, nCol: 40 };
    const q = initKinPinn(24, 5, 0, -1), N = 20000;
    for (let i = 0; i < N; i++) {
      c.lambdaOde = 30 ** (i / N);
      kinAdam(q, kinLossGrad(q, data, c).grad, 0.02 * (0.1 + 0.45 * (1 + Math.cos((Math.PI * i) / N))));
    }
    return kOf(q);
  };
  const k = run(false), e1 = Math.abs(10 ** k.l1 / 0.6 - 1), e2 = Math.abs(10 ** k.l2 / 0.2 - 1);
  check('kinetics PINN recovers k1, k2 with a calibrated detector (< 10 %)', e1 < 0.1 && e2 < 0.1, `(k1 ${(10 ** k.l1).toFixed(3)}, k2 ${(10 ** k.l2).toFixed(3)})`);
  const u = run(true), a = 10 ** u.l1, b = 10 ** u.l2;
  const near = (x: number, y: number) => Math.abs(a / x - 1) < 0.15 && Math.abs(b / y - 1) < 0.15;
  check('with unknown calibration it lands in one of the two mirror minima', near(0.6, 0.2) || near(0.2, 0.6), `(k1 ${a.toFixed(3)}, k2 ${b.toFixed(3)}, s ${u.s.toFixed(2)})`);
}

// ---- gravity: density network, depth weighting
{
  const grid = makeGrid(40, 20);
  const sv = makeSurvey({ x0: 100, z0: 150, m: lambdaOf(40, 1000) }, 0.02, 3);
  const K = kernel(grid, sv.x);
  const small = initGravNet(4, 2), w1 = depthWeights(grid, true, 2);
  const e = fdCheck([small.a, small.p, small.q, small.b, small.c], gravLossGrad(small, grid, K, sv.g, w1, 1).grad, () => gravLossGrad(small, grid, K, sv.g, w1, 1).loss);
  check('gravity PINN gradient matches finite differences', e < 1e-4, `(${e.toExponential(1)})`);
  const depth = (dw: boolean) => {
    const net = initGravNet(24, 5), w = depthWeights(grid, dw, 2);
    let rho = new Float64Array(0);
    for (let i = 0; i < 3000; i++) { const r = gravLossGrad(net, grid, K, sv.g, w, 1); rho = r.rho; gravAdam(net, r.grad, 0.02); }
    return centroid(rho, grid).z;
  };
  const z0 = depth(false), z1 = depth(true);
  check('without depth weighting the mass sits near the surface', z0 < 100, `(centroid depth ${z0.toFixed(0)} m, true 150 m)`);
  check('with depth weighting it moves to the right depth (± 30 m)', Math.abs(z1 - 150) < 30, `(${z1.toFixed(0)} m)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
