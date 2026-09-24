// Checks for the gravity-exploration tab: run with `node tests/eng-mining.test.ts`
import { anomaly, makeSurvey, fitLM, lambdaOf, radiusOf, type Body } from '../src/lib/eng/gravity.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

const truth: Body = { x0: 80, z0: 140, m: lambdaOf(45, 1000) };

// 1. Half-width at half-maximum equals the depth
{
  const peak = anomaly(truth.x0, truth), half = anomaly(truth.x0 + truth.z0, truth);
  check('HWHM = depth', Math.abs(half / peak - 0.5) < 1e-12);
  check('peak of a realistic body is ~0.1–2 mGal', peak > 0.1 && peak < 2, `(${peak.toFixed(3)} mGal)`);
  check('radius/lambda round trip', Math.abs(radiusOf(lambdaOf(45, 1000), 1000) - 45) < 1e-9);
}

// 2. LM recovers the body from noisy data, starting far away
{
  const s = makeSurvey(truth, 0.01, 3);
  const path = fitLM(s, { x0: -200, z0: 60, m: 2 });
  const b = path[path.length - 1];
  const ok = Math.abs(b.x0 - truth.x0) < 10 && Math.abs(b.z0 - truth.z0) / truth.z0 < 0.08 && Math.abs(b.m - truth.m) / truth.m < 0.1;
  check('LM recovers x0, z0, lambda', ok, `(x0 ${b.x0.toFixed(1)}, z0 ${b.z0.toFixed(1)}, m ${b.m.toFixed(2)} vs ${truth.m.toFixed(2)}; ${path.length} iterates)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
