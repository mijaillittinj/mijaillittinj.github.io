// Checks for the compressed sensing module: run with `node tests/learn-cs.test.ts`
import { dctMatrix, makeProblem, minNorm, makeFista, debias, omp, synth, relErr, makePhaseMap } from '../src/lib/cs.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const N = 128, Psi = dctMatrix(N);

// 1. The DCT matrix is orthonormal
{
  let d = 0;
  for (let a = 0; a < N; a += 7) for (let b = 0; b < N; b += 5) {
    let s = 0;
    for (let t = 0; t < N; t++) s += Psi[t * N + a] * Psi[t * N + b];
    d = Math.max(d, Math.abs(s - (a === b ? 1 : 0)));
  }
  check('DCT basis is orthonormal', d < 1e-12, `(${d.toExponential(1)})`);
}

// 2. With enough samples, L1 (+ debias) and OMP recover a 5-tone signal exactly; least squares fails
{
  const p = makeProblem(Psi, N, 5, 32, 0, 3);
  const f = makeFista(p, 0.01);
  for (let k = 0; k < 300; k++) f.step();
  const eL1 = relErr(synth(Psi, N, debias(p, f.c, 0.02)), p.x), eOmp = relErr(synth(Psi, N, omp(p, 5)), p.x), eMn = relErr(synth(Psi, N, minNorm(p)), p.x);
  check('L1 + debias recovers the signal from 32 of 128 samples', eL1 < 1e-3, `(${eL1.toExponential(1)})`);
  check('OMP recovers the signal', eOmp < 1e-3, `(${eOmp.toExponential(1)})`);
  check('minimum-norm least squares fails', eMn > 0.5, `(${eMn.toFixed(2)})`);
}

// 3. Too few samples: recovery fails
{
  const p = makeProblem(Psi, N, 5, 12, 0, 3);
  const e = relErr(synth(Psi, N, omp(p, 5)), p.x);
  check('12 samples are not enough for 5 tones', e > 0.3, `(${e.toFixed(2)})`);
}

// 4. Phase transition: success for M well above 2K ln(N/K), failure well below
{
  const pm = makePhaseMap(Psi, N, [3, 8], [12, 64], 4);
  while (!pm.done) pm.step();
  const [k3m12, k3m64, k8m12, k8m64] = pm.rate;
  check('phase transition', k3m64 === 1 && k8m64 >= 0.75 && k8m12 === 0 && k3m12 < 1, `(K3: ${k3m12}, ${k3m64}; K8: ${k8m12}, ${k8m64})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
