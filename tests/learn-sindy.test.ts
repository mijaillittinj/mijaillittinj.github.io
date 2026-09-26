// Checks for the SINDy module: run with `node tests/learn-sindy.test.ts`
import { system, makeData, discover, derivatives, simulate, modelRhs } from '../src/lib/sindy.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. Lotka–Volterra coefficients recovered at low noise (defaults of the lab)
{
  const sys = system('lotka');
  const r = discover(sys, makeData(sys, sys.tMax, 0.02, 1, 3), 0.1, false);
  check('Lotka–Volterra: exactly the true terms', r.missing === 0 && r.spurious === 0, `(${r.nTerms} terms)`);
  check('Lotka–Volterra: coefficients within 2 %', r.maxRelErr < 0.02, `(max rel err ${(r.maxRelErr * 100).toFixed(2)} %)`);
}

// 2. Other systems at low noise
for (const id of ['oscillator', 'pendulum', 'lorenz'] as const) {
  const sys = system(id);
  const r = discover(sys, makeData(sys, sys.tMax, 0.005, 1, 3), 0.1, false);
  check(`${id}: true terms recovered`, r.missing === 0 && r.spurious === 0, `(err ${(r.maxRelErr * 100).toFixed(1)} %)`);
}

// 3. Too small a threshold admits spurious terms; too large removes real ones
{
  const sys = system('lotka'), d = makeData(sys, sys.tMax, 0.02, 1, 3);
  check('tiny λ gives spurious terms', discover(sys, d, 0.001, false).spurious > 0);
  check('large λ removes real terms', discover(sys, d, 0.3, false).missing > 0);
}

// 4. Smoothed derivative of a sine is accurate
{
  const dt = 0.01, X = Array.from({ length: 400 }, (_, k) => [Math.sin(k * dt)]);
  const { dX } = derivatives(X, dt, 2); // edge points dropped: dX[i] belongs to sample i + 2
  let e = 0;
  for (let i = 0; i < dX.length; i++) e = Math.max(e, Math.abs(dX[i][0] - Math.cos((i + 2) * dt)));
  check('Savitzky–Golay derivative of sin t', e < 1e-3, `(max err ${e.toExponential(1)})`);
}

// 5. Simulating the discovered oscillator reproduces the data
{
  const sys = system('oscillator'), d = makeData(sys, sys.tMax, 0.005, 1, 3);
  const r = discover(sys, d, 0.1, false);
  const sim = simulate(modelRhs(sys, r.Xi), d.clean[0], sys.dt, d.t.length);
  let e = 0;
  sim.forEach((s, i) => (e = Math.max(e, Math.abs(s[0] - d.clean[i][0]))));
  check('discovered oscillator trajectory matches', e < 0.05, `(max |Δx| ${e.toFixed(3)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
