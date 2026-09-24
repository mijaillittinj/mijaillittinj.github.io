// Checks for the MCMC module: run with `node tests/learn-mcmc.test.ts`
import { makeData, startChain, step, quantile } from '../src/lib/mcmc.ts';
let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const data = makeData(12, 22, 15, 30, 1, 4);
const ch = startChain(data, [30, 50], 9);
step(ch, data, 30000, 0.6, 0.6);
const burn = 3000;
const tau = ch.tau.slice(burn), Ti = ch.Tinf.slice(burn);
const ci = (a: number[]) => [quantile(a, 0.025), quantile(a, 0.975)];
const [t0, t1] = ci(tau), [T0, T1] = ci(Ti);
check('95 % interval of τ contains the truth', t0 < 12 && 12 < t1, `([${t0.toFixed(1)}, ${t1.toFixed(1)}])`);
check('95 % interval of T∞ contains the truth', T0 < 22 && 22 < T1, `([${T0.toFixed(1)}, ${T1.toFixed(1)}])`);
const acc = ch.accepted / ch.proposed;
check('acceptance rate reasonable', acc > 0.1 && acc < 0.9, `(${(acc * 100).toFixed(0)} %)`);
const ch2 = startChain(data, [12, 22], 9); step(ch2, data, 3000, 8, 8);
check('huge proposals are mostly rejected', ch2.accepted / ch2.proposed < 0.1, `(${((ch2.accepted / ch2.proposed) * 100).toFixed(1)} %)`);
if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
