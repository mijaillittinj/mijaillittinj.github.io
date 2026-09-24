// Checks for the image-deblurring tab: run with `node tests/eng-computing.test.ts`
import { fft1, fft2, motionPsf, convolve, blur, tikhonov, richardsonLucyStep, psnr, testPattern } from '../src/lib/eng/deblur.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const n = 64;

// 1. FFT round trip and agreement with a direct DFT
{
  const re = Float64Array.from({ length: 16 }, (_, i) => Math.sin(i) + i * 0.1), im = new Float64Array(16);
  const r2 = Float64Array.from(re), i2 = Float64Array.from(im);
  fft1(r2, i2);
  let err = 0;
  for (let k = 0; k < 16; k++) {
    let sr = 0, si = 0;
    for (let t = 0; t < 16; t++) { sr += re[t] * Math.cos((-2 * Math.PI * k * t) / 16); si += re[t] * Math.sin((-2 * Math.PI * k * t) / 16); }
    err = Math.max(err, Math.hypot(sr - r2[k], si - i2[k]));
  }
  check('FFT matches the direct DFT', err < 1e-9, `(max err ${err.toExponential(1)})`);
  const f = testPattern(n), F = fft2(n, f), back = fft2(n, F.re, F.im, true).re;
  let e2 = 0;
  for (let i = 0; i < f.length; i++) e2 = Math.max(e2, Math.abs(back[i] - f[i]));
  check('2D FFT round trip', e2 < 1e-9);
}

// 2. PSF is normalised and convolution with a delta is the identity
{
  const h = motionPsf(n, 9, 30);
  check('PSF sums to one', Math.abs(h.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  const d = new Float64Array(n * n); d[0] = 1;
  const f = testPattern(n), c = convolve(n, f, d);
  check('convolution with a delta is the identity', c.every((v, i) => Math.abs(v - f[i]) < 1e-9));
}

// 3. Noise-free: the inverse filter recovers the image when H has no zeros
{
  const h = motionPsf(n, 0, 0, 1.0); // mild Gaussian defocus, no spectral zeros to machine precision issues
  const f = testPattern(n), g = convolve(n, f, h);
  check('noise-free inverse filter recovers the image', psnr(tikhonov(n, g, h, 0), f) > 40, `(${psnr(tikhonov(n, g, h, 0), f).toFixed(1)} dB)`);
}

// 4. With noise and motion blur: the inverse filter explodes; Tikhonov and Richardson–Lucy improve on the blurred image
{
  const N = 128, f = testPattern(N), h = motionPsf(N, 15, 20), g = blur(N, f, h, 0.01, 1);
  const pBlur = psnr(g, f), pInv = psnr(tikhonov(N, g, h, 0), f), pTik = psnr(tikhonov(N, g, h, 1e-3), f);
  check('inverse filter amplifies noise', pInv < pBlur, `(blurred ${pBlur.toFixed(1)} dB, inverse ${pInv.toFixed(1)} dB)`);
  check('Tikhonov improves on the blurred image', pTik > pBlur + 5, `(${pTik.toFixed(1)} dB)`);
  const H = fft2(N, h), r = Float64Array.from(g, (v) => Math.max(v, 0));
  for (let k = 0; k < 150; k++) richardsonLucyStep(H, g, r);
  check('Richardson–Lucy improves on the blurred image and stays non-negative', psnr(r, f) > pBlur + 3 && r.every((v) => v >= 0), `(${psnr(r, f).toFixed(1)} dB)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
