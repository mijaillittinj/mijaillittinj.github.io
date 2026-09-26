// Checks for the computed tomography module: run with `node tests/learn-ct.test.ts`
import { phantom, angles, radon, radonT, fbp, makeSirt, quality } from '../src/lib/ct.ts';
import { rng } from '../src/lib/linalg.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const n = 64, R = rng(2);
const f = phantom(n, 'head');

// 1. The back-projector is the exact adjoint of the projector
{
  const th = angles(12);
  const x = Float64Array.from({ length: n * n }, () => R.normal()), y = Float64Array.from({ length: 12 * n }, () => R.normal());
  const Ax = radon(x, n, th), ATy = radonT(y, n, th);
  let a = 0, b = 0;
  for (let i = 0; i < Ax.length; i++) a += Ax[i] * y[i];
  for (let i = 0; i < x.length; i++) b += x[i] * ATy[i];
  check('radonT is the adjoint of radon', Math.abs(a - b) / Math.abs(a) < 1e-10, `(rel ${(Math.abs(a - b) / Math.abs(a)).toExponential(1)})`);
}

// 2. Filtering matters: FBP beats plain back-projection with many views
{
  const th = angles(120), s = radon(f, n, th);
  const pBp = quality(fbp(s, n, th, 'none'), f, n).psnr, pF = quality(fbp(s, n, th, 'ramp'), f, n).psnr;
  check('filtered back-projection beats plain back-projection', pF > pBp + 5, `(FBP ${pF.toFixed(1)} dB, BP ${pBp.toFixed(1)} dB)`);
}

// 3. Few views: SIRT with non-negativity beats FBP
{
  const th = angles(8), s = radon(f, n, th);
  const sirt = makeSirt(s, n, th, true);
  for (let k = 0; k < 150; k++) sirt.step();
  const pS = quality(sirt.x, f, n).psnr, pF = quality(fbp(s, n, th, 'ramp'), f, n).psnr;
  check('with 8 views, SIRT (x ≥ 0) beats FBP', pS > pF + 3, `(SIRT ${pS.toFixed(1)} dB, FBP ${pF.toFixed(1)} dB)`);
}

// 4. Noise: the apodised Shepp–Logan filter amplifies noise less than the ramp; SIRT semi-convergence
{
  const th = angles(60), s0 = radon(f, n, th);
  let mx = 0;
  for (const v of s0) mx = Math.max(mx, v);
  const s = Float64Array.from(s0, (v) => v + 0.08 * mx * R.normal());
  const pR = quality(fbp(s, n, th, 'ramp'), f, n).psnr, pSL = quality(fbp(s, n, th, 'shepp-logan'), f, n).psnr;
  check('with noise, Shepp–Logan filter beats the ramp', pSL > pR, `(SL ${pSL.toFixed(1)}, ramp ${pR.toFixed(1)} dB)`);
  const sirt = makeSirt(s, n, th, true);
  const hist: number[] = [];
  for (let k = 0; k < 300; k++) { sirt.step(); hist.push(quality(sirt.x, f, n).psnr); }
  const best = Math.max(...hist);
  check('with noise, SIRT error is lowest at an intermediate iteration', hist.indexOf(best) < 250 && best > hist[299] + 0.5, `(best at ${hist.indexOf(best) + 1}, ${best.toFixed(1)} vs ${hist[299].toFixed(1)} dB)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
