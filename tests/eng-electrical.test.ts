// Checks for the smart-meter disaggregation tab: run with `node tests/eng-electrical.test.ts`
import { APPLIANCES, MINUTES, BASE_LOAD, syntheticDay, meter, disaggregate, energies, accuracy, energyAccuracy } from '../src/lib/eng/nilm.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const all = APPLIANCES.map(() => true);
const kettle = APPLIANCES.findIndex((a) => a.id === 'kettle'), oven = APPLIANCES.findIndex((a) => a.id === 'oven');

// 1. Forward model: noise-free meter equals base load plus the powers of the appliances on
{
  const S = syntheticDay(4, all), P = meter(S, 0, 1);
  let maxErr = 0;
  for (let t = 0; t < MINUTES; t++) {
    let s = BASE_LOAD;
    APPLIANCES.forEach((a, i) => (s += a.power * S[i][t]));
    maxErr = Math.max(maxErr, Math.abs(P[t] - s));
  }
  check('noise-free meter is the sum of the appliances', maxErr < 1e-9);
  check('kettle and oven draw the same power', APPLIANCES[kettle].power === APPLIANCES[oven].power);
}

// 2. With duration priors the day is disaggregated almost perfectly at moderate noise
for (const seed of [1, 2, 3]) {
  const S = syntheticDay(seed, all), P = meter(S, 40, seed + 10);
  const E = disaggregate(P, 40, all, true);
  const acc = accuracy(S, E, all), eacc = energyAccuracy(S, E);
  check(`day ${seed}: disaggregation with duration priors`, acc > 0.97 && eacc > 0.95, `(state ${(acc * 100).toFixed(1)} %, energy ${(eacc * 100).toFixed(1)} %)`);
}

// 3. Without duration priors, equal-power appliances are confused (kettle vs oven)
{
  const S = syntheticDay(1, all), P = meter(S, 40, 11);
  const flat = energies(disaggregate(P, 40, all, false)), tru = energies(S);
  const kettleErr = Math.abs(flat[kettle] - tru[kettle]) + Math.abs(flat[oven] - tru[oven]);
  check('flat priors confuse kettle and oven', kettleErr > 1, `(kettle ${flat[kettle].toFixed(2)} vs ${tru[kettle].toFixed(2)} kWh, oven ${flat[oven].toFixed(2)} vs ${tru[oven].toFixed(2)} kWh)`);
  const total = (e: number[]) => e.reduce((a, b) => a + b, 0);
  check('flat priors still conserve the total energy', Math.abs(total(flat) - total(tru)) / total(tru) < 0.05);
}

// 4. Absent appliances are never switched on
{
  const pres = all.map((_, i) => i !== oven);
  const S = syntheticDay(2, pres), E = disaggregate(meter(S, 20, 3), 20, pres, true);
  check('absent appliance stays off', E[oven].every((v) => v === 0));
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
