// Checks for the chemical-kinetics tab: run with `node tests/eng-chemical.test.ts`
import { conc, makeData, misfit, nelderMead } from '../src/lib/eng/kinetics.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. Mass balance and the k1 = k2 limit
{
  let worst = 0;
  for (const t of [0, 0.5, 2, 7]) { const c = conc(t, 0.7, 2.3); worst = Math.max(worst, Math.abs(c.A + c.B + c.C - 1)); }
  check('A + B + C = 1', worst < 1e-12);
  const a = conc(1.3, 0.8, 0.8).B, b = conc(1.3, 0.8, 0.8 * (1 + 1e-5)).B;
  check('k1 = k2 limit is continuous', Math.abs(a - b) < 1e-5, `(${a.toFixed(6)} vs ${b.toFixed(6)})`);
}

// 2. B peaks at t* = ln(k2/k1)/(k2 − k1)
{
  const k1 = 0.5, k2 = 2, ts = Math.log(k2 / k1) / (k2 - k1);
  check('B maximum at t*', conc(ts, k1, k2).B > conc(ts * 0.97, k1, k2).B && conc(ts, k1, k2).B > conc(ts * 1.03, k1, k2).B);
}

// 3. Calibrated B data: the fit recovers (k1, k2)
const l1 = Math.log10(0.4), l2 = Math.log10(2.5);
const data = makeData(0.4, 2.5, 0.02, 11);
{
  const o = { scaleUnknown: false, measureA: false };
  const p = nelderMead((x) => misfit(x[0], x[1], data, o).m, [0, -0.5]);
  const [a, b] = p[p.length - 1];
  check('fit recovers k1, k2 (calibrated)', Math.abs(a - l1) < 0.05 && Math.abs(b - l2) < 0.05, `(${(10 ** a).toFixed(3)}, ${(10 ** b).toFixed(3)})`);
}

// 4. Unknown calibration: (k1, k2) and (k2, k1) are equally good
{
  const o = { scaleUnknown: true, measureA: false };
  const m12 = misfit(l1, l2, data, o).m, m21 = misfit(l2, l1, data, o).m;
  check('k1 <-> k2 symmetry with unknown scale', Math.abs(m12 - m21) < 1e-12 * (1 + m12), `(${m12.toExponential(3)} vs ${m21.toExponential(3)})`);
  const oA = { scaleUnknown: true, measureA: true };
  const a12 = misfit(l1, l2, data, oA).m, a21 = misfit(l2, l1, data, oA).m;
  check('measuring A breaks the symmetry', a21 > 5 * a12, `(ratio ${(a21 / a12).toFixed(1)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
