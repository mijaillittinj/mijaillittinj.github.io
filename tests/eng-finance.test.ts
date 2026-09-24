// Finance tab (Black–Scholes): run with `node tests/eng-finance.test.ts`
import { bsCall, bsDelta, impliedVol, makeMarket, fitSigma, Phi } from '../src/lib/eng/finance.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. reference value (Hull, S=42, K=40, r=0.1, σ=0.2, T=0.5 → 4.76)
{ const c = bsCall(42, 40, 0.1, 0.2, 0.5); check('Black–Scholes reference price (Hull 4.76)', Math.abs(c - 4.759) < 2e-3, `(${c.toFixed(4)})`); }
// 2. Φ accuracy
check('Φ(1.96) ≈ 0.975', Math.abs(Phi(1.96) - 0.9750021) < 1e-6, `(${Phi(1.96).toFixed(7)})`);
// 3. put–call parity with a put from the formula P = K e^{-rT} Φ(−d2) − S Φ(−d1), via C − P = S − K e^{−rT}
{
  const S = 100, K = 110, r = 0.04, s = 0.3, t = 0.7;
  const sq = s * Math.sqrt(t), d1 = (Math.log(S / K) + (r + 0.5 * s * s) * t) / sq, d2 = d1 - sq;
  const P = K * Math.exp(-r * t) * Phi(-d2) - S * Phi(-d1);
  const gap = bsCall(S, K, r, s, t) - P - (S - K * Math.exp(-r * t));
  check('put–call parity', Math.abs(gap) < 1e-9, `(${gap.toExponential(1)})`);
}
// 4. limits
check('price → payoff as τ → 0', Math.abs(bsCall(120, 100, 0.03, 0.3, 1e-8) - 20) < 1e-6);
check('deep OTM price ≈ 0 and delta ≈ 0', bsCall(40, 100, 0.03, 0.2, 0.5) < 1e-8 && bsDelta(40, 100, 0.03, 0.2, 0.5) < 1e-6);
// 5. implied-vol round trip
{
  let mx = 0;
  for (const K of [70, 90, 100, 115, 140]) for (const s of [0.1, 0.25, 0.6]) {
    const p = bsCall(100, K, 0.03, s, 0.8), iv = impliedVol(p, 100, K, 0.03, 0.8);
    if (p > 1e-6) mx = Math.max(mx, Math.abs(iv - s));
  }
  check('implied volatility round trip', mx < 1e-6, `(max err ${mx.toExponential(1)})`);
}
// 6. calibration: flat market recovered; smile market leaves systematic residuals
{
  const flat = makeMarket(7, 0.5, false), f = fitSigma(flat);
  check('least-squares σ recovers a flat market', Math.abs(f.sigma / flat.sigma0 - 1) < 0.03, `(${f.sigma.toFixed(3)} vs ${flat.sigma0.toFixed(3)})`);
  const sm = makeMarket(7, 0, true), g = fitSigma(sm);
  check('a smile cannot be fitted by one σ (larger RMS)', g.rms > 5 * fitSigma(makeMarket(7, 0, false)).rms + 0.02, `(rms ${g.rms.toFixed(3)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
