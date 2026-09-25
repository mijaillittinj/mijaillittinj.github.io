// Checks for the trustworthy-AI example (computing tab): run with `node tests/eng-trust.test.ts`
import * as T from '../src/lib/eng/trust.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. exact solution satisfies the oscillator equation and the initial conditions
{
  const h = 1e-4;
  let worst = 0;
  for (const x of [0.5, 2, 4.3, 8]) {
    const y = T.truth(x), yp = (T.truth(x + h) - T.truth(x - h)) / (2 * h), ypp = (T.truth(x + h) - 2 * y + T.truth(x - h)) / (h * h);
    worst = Math.max(worst, Math.abs(ypp + 2 * T.ZETA * T.OMEGA * yp + T.OMEGA ** 2 * y));
  }
  const yp0 = (T.truth(h) - T.truth(-h)) / (2 * h);
  check('truth solves the ODE with y(0)=1, y\'(0)=0', worst < 1e-5 && Math.abs(T.truth(0) - 1) < 1e-12 && Math.abs(yp0) < 1e-6, `(residual ${worst.toExponential(1)})`);
}

// 2. analytic gradient (data + physics + decay) matches finite differences
{
  const d = T.makeData(3, 0.05), net = T.initTNet(5, 2);
  const cfg = { physics: 0.3, decay: 1e-3, collocation: T.collocation(15) };
  const { g } = T.lossGradT(net, d.train, cfg);
  let mr = 0, p = 0;
  for (const arr of [net.a, net.w, net.b, net.c]) for (let j = 0; j < arr.length; j++, p++) {
    const h = 1e-6, v = arr[j];
    arr[j] = v + h; const lp = T.lossGradT(net, d.train, cfg).loss; arr[j] = v - h; const lm = T.lossGradT(net, d.train, cfg).loss; arr[j] = v;
    const fd = (lp - lm) / (2 * h);
    mr = Math.max(mr, Math.abs(fd - g[p]) / Math.max(1e-6, Math.abs(fd)));
  }
  check('network gradient matches finite differences', mr < 1e-4, `(max rel err ${mr.toExponential(2)})`);
}

// 3. conformal quantile: finite-sample rule and coverage on exchangeable data
{
  check('conformal quantile picks the ⌈(n+1)(1−α)⌉-th score', T.conformalQuantile([5, 1, 4, 2, 3, 6, 8, 7, 9], 0.8) === 8);
}

// 4. ensembles: coverage, disagreement and the value of physics
const d = T.makeData(3, 0.05);
const train = (phys: boolean, steps: number) => {
  const { nets, cfg } = T.makeEnsemble(phys);
  for (const net of nets) for (let k = 0; k < steps; k++) T.adamT(net, T.lossGradT(net, d.train, cfg).g, T.lrAt(k, steps));
  return nets;
};
const nn = train(false, T.NN_STEPS), pinn = train(true, T.PINN_STEPS);
const muOf = (nets: T.TNet[]) => (x: number) => T.ensembleStats(nets, [x]).mean[0];
const qOf = (nets: T.TNet[]) => T.conformalQuantile(d.calib.map((s) => Math.abs(s.y - muOf(nets)(s.x))), 0.9);
{
  const covNN = T.coverageByRegion(d.test, muOf(nn), qOf(nn));
  check('conformal coverage ≈ 90 % where data exist (NN)', covNN.data > 0.8 && covNN.data < 0.99, `(${(covNN.data * 100).toFixed(0)} %)`);
  check('coverage collapses in the gap and beyond the data (NN)', covNN.gap < 0.6 && covNN.extra < 0.6, `(gap ${(covNN.gap * 100).toFixed(0)} %, beyond ${(covNN.extra * 100).toFixed(0)} %)`);
  const xs = Array.from({ length: 200 }, (_, i) => ((i + 0.5) / 200) * T.L), st = T.ensembleStats(nn, xs);
  const mean = (r: T.Region) => { let s = 0, n = 0; xs.forEach((x, i) => { if (T.region(x) === r) { s += st.sd[i]; n++; } }); return s / n; };
  check('ensemble spread is larger in the gap and beyond the data', mean('gap') > 3 * mean('data') && mean('extra') > 3 * mean('data'), `(data ${mean('data').toFixed(3)}, gap ${mean('gap').toFixed(3)}, beyond ${mean('extra').toFixed(3)})`);
  const eNN = T.rmsInRegion(muOf(nn), 'gap'), eP = T.rmsInRegion(muOf(pinn), 'gap');
  const xNN = T.rmsInRegion(muOf(nn), 'extra'), xP = T.rmsInRegion(muOf(pinn), 'extra');
  check('PINN ensemble error in the gap ≪ plain NN', eP < 0.2 * eNN, `(NN ${eNN.toFixed(3)}, PINN ${eP.toFixed(3)})`);
  check('PINN ensemble extrapolates beyond the data', xP < 0.2 * xNN, `(NN ${xNN.toFixed(3)}, PINN ${xP.toFixed(3)})`);
  const covP = T.coverageByRegion(d.test, muOf(pinn), qOf(pinn));
  check('PINN conformal band keeps its coverage in the gap', covP.gap > 0.75, `(${(covP.gap * 100).toFixed(0)} %)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
