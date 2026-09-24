// Checks for the teaching examples: run with `node tests/teaching.test.ts`
import {
  fLaminar, frictionFactor, fColebrook, fHaaland, uLaminar, uPowerLaw,
  tankHeight, drainTime, wallTransient, profileMean, lumped, oneTermCentre,
  heatLossPerLength, criticalRadius, planck, wienPeak, bandPower, bandFraction, SIGMA,
  effCounter, effParallel,
} from '../src/lib/teaching.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

// Fluid mechanics
check('laminar f = 64/Re', frictionFactor(1000, 1e-3) === fLaminar(1000) && Math.abs(fLaminar(2000) - 0.032) < 1e-12);
{
  let worst = 0;
  for (const Re of [5e3, 2e4, 1e5, 1e6, 1e7]) for (const e of [0, 1e-5, 1e-4, 1e-3, 1e-2]) worst = Math.max(worst, rel(fHaaland(Re, e), fColebrook(Re, e)));
  check('Haaland within 2% of Colebrook', worst < 0.02, `(max ${(worst * 100).toFixed(2)}%)`);
  const f = fColebrook(1e5, 1e-4), x = 1 / Math.sqrt(f);
  check('Colebrook residual', Math.abs(x + 2 * Math.log10(1e-4 / 3.7 + 2.51 * x / 1e5)) < 1e-10, `(f = ${f.toFixed(5)})`);
}
{
  // both profiles must integrate to the mean velocity: ∫0^1 u/V · 2s ds = 1
  const integ = (u: (s: number) => number) => { let s = 0; const n = 20000; for (let i = 0; i < n; i++) { const r = (i + 0.5) / n; s += u(r) * 2 * r / n; } return s; };
  check('velocity profiles have unit mean', Math.abs(integ(uLaminar) - 1) < 1e-6 && Math.abs(integ((s) => uPowerLaw(s)) - 1) < 1e-3);
}
{
  const h0 = 2, Dt = 1, d = 0.05, Cd = 0.6, td = drainTime(h0, Dt, d, Cd);
  // explicit ODE integration of A dh/dt = −Cd a √(2gh)
  let h = h0, t = 0; const dt = 0.01, ratio = Cd * (d / Dt) ** 2;
  while (h > 1e-6 && t < 2 * td) { h -= dt * ratio * Math.sqrt(2 * 9.81 * Math.max(h, 0)); t += dt; }
  check('Torricelli drain time', rel(t, td) < 0.01 && tankHeight(td, h0, Dt, d, Cd) === 0 && Math.abs(tankHeight(0, h0, Dt, d, Cd) - h0) < 1e-12, `(t_d = ${td.toFixed(1)} s)`);
}

// Heat transfer
{
  const Bi = 0.01, Fo = [10, 50, 100];
  const prof = wallTransient(Bi, Fo);
  const worst = Math.max(...prof.map((p, k) => rel(profileMean(p), lumped(Bi, Fo[k]))));
  check('small Bi: FD mean matches lumped capacitance', worst < 0.01, `(max ${(worst * 100).toFixed(2)}%)`);
}
{
  const Bi = 1, Fo = 0.5;
  const [p] = wallTransient(Bi, [Fo], 81, 5e-4);
  check('Bi = 1: FD centre matches one-term series', rel(p[0], oneTermCentre(Bi, Fo)) < 0.01, `(${p[0].toFixed(4)} vs ${oneTermCentre(Bi, Fo).toFixed(4)})`);
}
{
  const k = 0.05, h = 5, ri = 0.004, rc = criticalRadius(k, h);
  const q = (r: number) => heatLossPerLength(r, ri, k, h, 50);
  check('heat loss is maximal at r_c = k/h', q(rc) > q(rc * 0.97) && q(rc) > q(rc * 1.03));
}
{
  const T = 1500, lm = wienPeak(T);
  check('Wien peak', planck(lm, T) > planck(lm * 0.99, T) && planck(lm, T) > planck(lm * 1.01, T), `(${lm.toFixed(3)} µm)`);
  const total = bandPower(0.05, 2000, T);
  check('∫ Planck = σT⁴ within 1%', rel(total, SIGMA * T ** 4) < 0.01, `(${(100 * rel(total, SIGMA * T ** 4)).toFixed(3)}%)`);
  check('visible fraction of the Sun ≈ 0.37–0.47', bandFraction(0.38, 0.78, 5800) > 0.37 && bandFraction(0.38, 0.78, 5800) < 0.47, `(${bandFraction(0.38, 0.78, 5800).toFixed(3)})`);
}

// Heat exchangers
check('ε–NTU limits', Math.abs(effCounter(50, 0) - 1) < 1e-9 && Math.abs(effParallel(50, 1) - 0.5) < 1e-9 && Math.abs(effCounter(2, 1) - 2 / 3) < 1e-12 && rel(effCounter(1, 0), effParallel(1, 0)) < 1e-12);
check('counterflow ≥ parallel flow', [0.25, 0.5, 1].every((Cr) => [0.5, 1, 2, 4].every((N) => effCounter(N, Cr) >= effParallel(N, Cr) - 1e-12)));

console.log(fails ? `\n${fails} check(s) failed` : '\nall checks passed');
if (fails) process.exit(1);
