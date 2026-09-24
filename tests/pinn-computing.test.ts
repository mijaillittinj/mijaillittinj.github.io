// Checks for the physics-informed neural field of the computing tab: run with `node tests/pinn-computing.test.ts`
import { initImageNet, imageLossGrad, getImageParams, setImageParams, makeImageTrainer } from '../src/lib/eng/pinn-image.ts';
import { motionPsf, blur, tikhonov, psnr, fft2, testPattern } from '../src/lib/eng/deblur.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. analytic gradient vs central finite differences (small image)
{
  const n = 16, net = initImageNet(4, 2, 2), h = motionPsf(n, 4, 30), Hs = fft2(n, h), g = blur(n, testPattern(n), h, 0.01, 1);
  const { grad } = imageLossGrad(net, Hs, g, 1e-3), p0 = getImageParams(net);
  let maxRel = 0;
  for (let k = 0; k < p0.length; k++) {
    const e = 1e-6, p = Float64Array.from(p0);
    p[k] += e; setImageParams(net, p); const lp = imageLossGrad(net, Hs, g, 1e-3).loss;
    p[k] -= 2 * e; setImageParams(net, p); const lm = imageLossGrad(net, Hs, g, 1e-3).loss;
    const fd = (lp - lm) / (2 * e);
    maxRel = Math.max(maxRel, Math.abs(fd - grad[k]) / Math.max(1e-7, Math.abs(fd)));
  }
  setImageParams(net, p0);
  check('neural-field gradient matches finite differences', maxRel < 1e-4, `(max rel err ${maxRel.toExponential(2)})`);
}

// 2. trained only through the blur model, the field deblurs the image
{
  const n = 128, f = testPattern(n), h = motionPsf(n, 15, 20), Hs = fft2(n, h), g = blur(n, f, h, 0.01, 7);
  const net = initImageNet(64, 3, 16), tr = makeImageTrainer(net, Hs, g, 0.02, 0.01);
  const t0 = performance.now();
  let r = tr.step();
  for (let k = 1; k < 300; k++) r = tr.step();
  const ms = (performance.now() - t0) / 300;
  const pB = psnr(g, f), pN = psnr(r.f, f), pT = psnr(tikhonov(n, g, h, 1e-3), f);
  check('PINN deblurs (≥ 5 dB above the blurred photo)', pN > pB + 5, `(${pN.toFixed(1)} vs ${pB.toFixed(1)} dB, ${ms.toFixed(1)} ms/step)`);
  check('PINN at least as good as Tikhonov at λ = 1e-3 (within 1 dB)', pN > pT - 1, `(Tikhonov ${pT.toFixed(1)} dB)`);
  check('output stays in [0, 1]', r.f.every((v) => v >= 0 && v <= 1));
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
