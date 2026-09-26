// Checks for the classifier module: run with `node tests/learn-classifier.test.ts`
import { makeDataset, initNet, lossGrad, adam, accuracy, meanLoss } from '../src/lib/classifier.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. Analytic gradient vs finite differences (tanh and ReLU, with weight decay)
for (const act of ['tanh', 'relu'] as const) {
  const pts = makeDataset('moons', 30, 0.1, 2), net = initNet(['x', 'y', 'x2', 'xy'], [5, 4], act, 3);
  const { grads } = lossGrad(net, pts, 1e-3);
  let maxRel = 0;
  net.layers.forEach((L, l) => {
    for (const [arr, g] of [[L.W, grads[2 * l]], [L.b, grads[2 * l + 1]]] as const) for (let i = 0; i < arr.length; i++) {
      const v = arr[i], h = 1e-6;
      arr[i] = v + h; const lp = lossGrad(net, pts, 1e-3).loss; arr[i] = v - h; const lm = lossGrad(net, pts, 1e-3).loss; arr[i] = v;
      const fd = (lp - lm) / (2 * h);
      if (Math.abs(fd) > 1e-6) maxRel = Math.max(maxRel, Math.abs(fd - g[i]) / Math.abs(fd));
    }
  });
  check(`gradient matches finite differences (${act})`, maxRel < 1e-4, `(max rel err ${maxRel.toExponential(1)})`);
}

const trainTest = (ds: 'circles' | 'xor', fs: ('x' | 'y' | 'x2' | 'y2' | 'xy')[], hidden: number[]) => {
  const tr = makeDataset(ds, 150, 0.08, 1), te = makeDataset(ds, 400, 0.08, 99), net = initNet(fs, hidden, 'tanh', 7);
  for (let k = 0; k < 2000; k++) adam(net, lossGrad(net, tr).grads, 0.02);
  return { acc: accuracy(net, te), loss: meanLoss(net, te) };
};

// 2. A linear model cannot separate circles or XOR; one hidden layer can
{
  const lin = trainTest('circles', ['x', 'y'], []), hid = trainTest('circles', ['x', 'y'], [8]);
  check('circles: linear model near chance', lin.acc < 0.65, `(${(lin.acc * 100).toFixed(0)} %)`);
  check('circles: one hidden layer separates', hid.acc > 0.95, `(${(hid.acc * 100).toFixed(0)} %)`);
  const xl = trainTest('xor', ['x', 'y'], []), xh = trainTest('xor', ['x', 'y'], [8]);
  check('XOR: linear model fails', xl.acc < 0.7, `(${(xl.acc * 100).toFixed(0)} %)`);
  check('XOR: one hidden layer learns it', xh.acc > 0.85, `(${(xh.acc * 100).toFixed(0)} %)`);
}

// 3. Feature engineering: x², y² make circles linearly separable
{
  const r = trainTest('circles', ['x', 'y', 'x2', 'y2'], []);
  check('circles with x², y²: linear model separates', r.acc > 0.95, `(${(r.acc * 100).toFixed(0)} %)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
