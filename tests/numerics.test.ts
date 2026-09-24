// Numerical checks for the in-browser demos: run with `node tests/numerics.test.ts`
import { initNet, lossGrad, makeData, makeTrainer, evalNet, exact, type PinnConfig } from '../src/lib/pinn.ts';
import { chordMatrix, project, onionPeel, tikhonov, profile } from '../src/lib/abel.ts';
import { rms, rng } from '../src/lib/linalg.ts';
import { objectives, run } from '../src/lib/optim.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. PINN analytic gradient vs finite differences
{
  const cfg: PinnConfig = { hidden: 6, lambdaPhys: 0.7, learnM: true, mInit: 1.3, lr: 1e-2, seed: 3 };
  const net = initNet(cfg.hidden, cfg.seed, cfg.mInit);
  const data = makeData(2, 5, 0.6, 0.01, 1);
  const col = Float64Array.from({ length: 15 }, (_, i) => i / 14);
  const { g } = lossGrad(net, data, col, cfg);
  const L = () => lossGrad(net, data, col, cfg).loss;
  let maxRel = 0;
  const probe = (arr: Float64Array, ga: Float64Array) => {
    for (let j = 0; j < arr.length; j++) {
      const h = 1e-6, v = arr[j];
      arr[j] = v + h; const lp = L(); arr[j] = v - h; const lm = L(); arr[j] = v;
      const fd = (lp - lm) / (2 * h);
      maxRel = Math.max(maxRel, Math.abs(fd - ga[j]) / Math.max(1e-6, Math.abs(fd)));
    }
  };
  probe(net.a, g.a); probe(net.w, g.w); probe(net.b, g.b);
  const h = 1e-6; const v = net.logm; net.logm = v + h; const lp = L(); net.logm = v - h; const lm = L(); net.logm = v;
  maxRel = Math.max(maxRel, Math.abs((lp - lm) / (2 * h) - g.logm) / Math.abs((lp - lm) / (2 * h)));
  check('PINN gradient matches finite differences', maxRel < 1e-4, `(max rel err ${maxRel.toExponential(2)})`);
}

// 2. PINN recovers m from sparse data
{
  const cfg: PinnConfig = { hidden: 16, lambdaPhys: 1, learnM: true, mInit: 1, lr: 0.01, seed: 2 };
  const net = initNet(cfg.hidden, cfg.seed, cfg.mInit);
  const data = makeData(2, 5, 0.5, 0.01, 4);
  const col = Float64Array.from({ length: 40 }, (_, i) => i / 39);
  const step = makeTrainer(net, cfg);
  for (let k = 0; k < 6000; k++) step(lossGrad(net, data, col, cfg).g);
  const m = Math.exp(net.logm);
  const xs = Array.from({ length: 50 }, (_, i) => i / 49);
  const err = rms(xs.map((x) => evalNet(net, x).u), xs.map((x) => exact(x, 2)));
  check('PINN identifies m ≈ 2 from 5 sensors on [0, 0.5]', Math.abs(m - 2) < 0.2, `(m = ${m.toFixed(3)}, field RMS ${err.toFixed(4)})`);
}

// 3. Abel: exact recovery without noise, regularisation helps with noise
{
  const n = 60;
  const L = chordMatrix(n);
  const f = Float64Array.from({ length: n }, (_, i) => profile('annular', (i + 0.5) / n, 0.5, 0.4));
  const p = project(L, f);
  check('Onion peeling is exact without noise', rms(onionPeel(L, p), f) < 1e-10);
  const r = rng(5); const pmax = Math.max(...p);
  const pn = Float64Array.from(p, (v) => v + 0.02 * pmax * r.normal());
  const e1 = rms(onionPeel(L, pn), f), e2 = rms(tikhonov(L, pn, 1e-3), f);
  check('Tikhonov beats direct inversion at 2 % noise', e2 < e1, `(direct ${e1.toFixed(3)}, regularised ${e2.toFixed(3)})`);
}

// 4. Gradient descent stability bound on the valley (alpha < 2/kappa)
{
  const obj = objectives[0];
  const stable = run(obj, [-2, 1], 'gd', { alpha: 1.9 / 20, steps: 400, k: 20 });
  const unstable = run(obj, [-2, 1], 'gd', { alpha: 2.1 / 20, steps: 400, k: 20 });
  check('GD converges for alpha < 2/kappa', stable.loss.at(-1)! < 1e-6);
  check('GD diverges for alpha > 2/kappa', unstable.diverged || unstable.loss.at(-1)! > 1);
}
if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
