// Checks for the neural-network module: run with `node tests/learn-mlp.test.ts`
import { initMLP, lossGrad, adam, mse, makeSplit } from '../src/lib/mlp.ts';
let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const { train, val } = makeSplit(12, 40, 0.1, 2);
// 1. Backprop vs finite differences (two hidden layers, with decay)
{
  const net = initMLP(2, 5, 3);
  const { grads } = lossGrad(net, train.x, train.y, 1e-2);
  const L = () => lossGrad(net, train.x, train.y, 1e-2).loss;
  let maxRel = 0;
  net.layers.forEach((Ly, l) => [Ly.W, Ly.b].forEach((p, pi) => {
    for (let i = 0; i < p.length; i++) {
      const h = 1e-6, v = p[i];
      p[i] = v + h; const lp = L(); p[i] = v - h; const lm = L(); p[i] = v;
      const fd = (lp - lm) / (2 * h), g = grads[2 * l + pi][i];
      maxRel = Math.max(maxRel, Math.abs(fd - g) / Math.max(1e-6, Math.abs(fd)));
    }
  }));
  check('backprop gradient matches finite differences', maxRel < 1e-5, `(max rel err ${maxRel.toExponential(2)})`);
}
// 2. Training reduces the loss; a big network on few points overfits (val ≫ train)
{
  const net = initMLP(2, 30, 1);
  const l0 = mse(net, train.x, train.y);
  for (let k = 0; k < 4000; k++) adam(net, lossGrad(net, train.x, train.y).grads, 0.01);
  const lt = mse(net, train.x, train.y), lv = mse(net, val.x, val.y);
  check('training reduces the loss', lt < 0.1 * l0, `(${l0.toFixed(3)} → ${lt.toExponential(1)})`);
  check('large network on 12 points overfits', lv > 3 * lt, `(train ${lt.toExponential(1)}, val ${lv.toExponential(1)})`);
}
if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
