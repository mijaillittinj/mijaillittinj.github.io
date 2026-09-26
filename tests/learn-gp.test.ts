// Checks for the Gaussian-process module: run with `node tests/learn-gp.test.ts`
import { kernel, posterior, logMarginal, fitStep, initialData, truth, type Hyper } from '../src/lib/gp.ts';
import { rng } from '../src/lib/linalg.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const { xs, ys } = initialData(3, 0.1);
const h0: Hyper = { logEll: Math.log(1), logSf2: Math.log(1), logSn2: Math.log(0.01) };

// 1. Gradient of the log marginal likelihood matches finite differences (both kernels)
for (const kind of ['rbf', 'matern52'] as const) {
  const { grad } = logMarginal(kind, xs, ys, h0);
  const keys = ['logEll', 'logSf2', 'logSn2'] as const;
  let maxRel = 0;
  keys.forEach((k, i) => {
    const e = 1e-5, hp = { ...h0, [k]: h0[k] + e }, hm = { ...h0, [k]: h0[k] - e };
    const fd = (logMarginal(kind, xs, ys, hp).lml - logMarginal(kind, xs, ys, hm).lml) / (2 * e);
    maxRel = Math.max(maxRel, Math.abs(fd - grad[i]) / Math.max(1e-6, Math.abs(fd)));
  });
  check(`${kind}: marginal-likelihood gradient matches finite differences`, maxRel < 1e-4, `(${maxRel.toExponential(1)})`);
}

// 2. The posterior interpolates near data and reverts to the prior far from it
{
  const { mean, sd } = posterior('rbf', xs, ys, [xs[0], 3.6, 20], h0);
  check('posterior mean passes close to a data point', Math.abs(mean[0] - ys[0]) < 0.05, `(${Math.abs(mean[0] - ys[0]).toFixed(3)})`);
  check('uncertainty is larger in the gap than at a data point', sd[1] > 3 * sd[0], `(${sd[0].toFixed(3)} vs ${sd[1].toFixed(3)})`);
  check('far from data the sd returns to the prior σ_f', Math.abs(sd[2] - 1) < 1e-3, `(${sd[2].toFixed(4)})`);
}

// 3. Fitting the hyperparameters increases the marginal likelihood; with 20 points the fitted GP
//    tracks the hidden function and its ±2σ band is calibrated
{
  const h: Hyper = { logEll: Math.log(0.3), logSf2: Math.log(1), logSn2: Math.log(0.3) };
  const st = { m: [0, 0, 0], v: [0, 0, 0], t: 0 };
  const l0 = logMarginal('rbf', xs, ys, h).lml;
  for (let k = 0; k < 400; k++) fitStep('rbf', xs, ys, h, st);
  check('fit increases the log marginal likelihood', logMarginal('rbf', xs, ys, h).lml > l0 + 1);

  const R = rng(5);
  const x20 = Array.from({ length: 20 }, (_, i) => 0.3 + (i * 9.4) / 19), y20 = x20.map((x) => truth(x) + 0.1 * R.normal());
  const h2: Hyper = { logEll: 0, logSf2: 0, logSn2: Math.log(0.05) };
  const st2 = { m: [0, 0, 0], v: [0, 0, 0], t: 0 };
  for (let k = 0; k < 600; k++) fitStep('rbf', x20, y20, h2, st2);
  const q = Array.from({ length: 91 }, (_, i) => 0.5 + (i * 9) / 90);
  const { mean, sd } = posterior('rbf', x20, y20, q, h2);
  const rms = Math.sqrt(q.reduce((s2, x, i) => s2 + (mean[i] - truth(x)) ** 2, 0) / q.length);
  const cov = q.filter((x, i) => Math.abs(mean[i] - truth(x)) < 2 * sd[i]).length / q.length;
  check('with 20 points the fitted GP tracks the hidden function (RMS < 0.15)', rms < 0.15, `(${rms.toFixed(3)}; fitted σ_n = ${Math.sqrt(Math.exp(h2.logSn2)).toFixed(3)}, true 0.1)`);
  check('and its ±2σ band covers the function (> 90 %)', cov > 0.9, `(${(cov * 100).toFixed(0)} %)`);
}

// 4. Kernel sanity: Matérn and RBF equal σ_f² at zero distance
check('k(0) = σ_f² for both kernels', Math.abs(kernel('rbf', 1, 1, h0).k - 1) < 1e-12 && Math.abs(kernel('matern52', 1, 1, h0).k - 1) < 1e-12);

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
