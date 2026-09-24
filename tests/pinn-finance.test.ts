// Black–Scholes PINN (finance tab): run with `node tests/pinn-finance.test.ts`
import { makePinnData, bsCall } from '../src/lib/eng/finance.ts';
import { initFinPinn, finLossGrad, finAdam, sigmaOfNet, evalFin } from '../src/lib/eng/pinn-finance.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const cfg = { r: 0.03, K: 100, lambdaP: 1, lambdaIC: 1, lambdaBC: 1 };

// 1. gradient check
{
  const data = makePinnData(1, 0.25, 0.03, 100, 0.5);
  const p = initFinPinn(4, 2, 0.4);
  const { grad } = finLossGrad(p, data, cfg);
  let mx = 0;
  for (let q = 0; q < p.th.length; q++) {
    const h = 1e-6, v = p.th[q];
    p.th[q] = v + h; const lp = finLossGrad(p, data, cfg).loss; p.th[q] = v - h; const lm = finLossGrad(p, data, cfg).loss; p.th[q] = v;
    const fd = (lp - lm) / (2 * h);
    mx = Math.max(mx, Math.abs(fd - grad[q]) / Math.max(1e-7, Math.abs(fd)));
  }
  check('PINN gradient matches finite differences', mx < 1e-4, `(${mx.toExponential(1)})`);
}
// 2. σ recovery (same schedule as the browser)
for (const sig of [0.2, 0.35]) {
  const data = makePinnData(3, sig, 0.03, 100, 0.5);
  const p = initFinPinn(32, 5, 0.3), N = 10000;
  for (let k = 0; k < N; k++) finAdam(p, finLossGrad(p, data, cfg).grad, 0.01 * (0.1 + 0.45 * (1 + Math.cos((Math.PI * k) / N))));
  const s = sigmaOfNet(p);
  const S = [80, 90, 100, 110, 120], v = evalFin(p, S, 0.5, 100);
  const e = Math.sqrt(S.reduce((a, x, i) => a + (v[i] - bsCall(x, 100, 0.03, sig, 0.5)) ** 2, 0) / S.length);
  check(`PINN recovers σ = ${sig} within 5 %`, Math.abs(s / sig - 1) < 0.05, `(σ̂ = ${s.toFixed(3)}, price RMS ${e.toFixed(3)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
