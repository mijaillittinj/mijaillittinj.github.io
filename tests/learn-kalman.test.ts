// Checks for the Kalman module: run with `node tests/learn-kalman.test.ts`
import { scenario, runFilter, init, predict, update, ellipse } from '../src/lib/kalman.ts';
let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const rmsd = (a: [number, number][], b: [number, number][], from = 10) => { let s = 0, n = 0; for (let k = from; k < a.length; k++) { s += (a[k][0] - b[k][0]) ** 2 + (a[k][1] - b[k][1]) ** 2; n++; } return Math.sqrt(s / n); };
const dt = 1, r = 3;
const { truth, meas } = scenario(300, dt, r, 5);
const eMeas = rmsd(meas, truth);
const eKF = rmsd(runFilter(meas, dt, 0.03, r), truth);
check('Kalman estimate beats raw GPS', eKF < 0.7 * eMeas, `(KF ${eKF.toFixed(2)} km, GPS ${eMeas.toFixed(2)} km)`);
const eTiny = rmsd(runFilter(meas, dt, 1e-7, r), truth), eHuge = rmsd(runFilter(meas, dt, 100, r), truth);
check('too little process noise lags the turns', eTiny > eKF, `(${eTiny.toFixed(2)})`);
check('too much process noise follows the noise', eHuge > eKF && Math.abs(eHuge - eMeas) < 0.3 * eMeas, `(${eHuge.toFixed(2)})`);
// covariance shrinks after an update
{
  const kf = predict(init([0, 0], r), dt, 0.02), up = update(kf, [1, 1], r);
  check('update shrinks the position covariance', up.P[0] < kf.P[0] && ellipse(up.P).rx < ellipse(kf.P).rx, '');
}
if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
