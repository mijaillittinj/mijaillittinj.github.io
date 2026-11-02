// Integrating sphere calibration against radiometric_calibration.py (find_sphere_disc, sphere_mask,
// calibration_constants with median and radial fill, effective_nd_transmission, load_sphere_spectrum,
// fit_colour_temperature): run with `node --experimental-strip-types tests/workshop-rgb-sphere.test.ts`
// Data: the September 2026 sphere (acA2040-55uc) binned 16 × 16, its dark and spectrum CSV.
import { readFileSync } from 'node:fs';
import { loadSphereSpectrum, sniffDelimiter, sniffHeader, resample, ndTransmission, effectiveNdTransmission, findSphereDisc, sphereMask, calibrationConstants, fractionNearCeiling } from '../src/lib/workshop-rgb/sphere.ts';
import { fitColourTemperature, CH } from '../src/lib/workshop-rgb/radiometry.ts';
import { PRESETS, presetPsi } from '../src/lib/workshop-rgb/presets.ts';
import { decideCalibration } from '../src/lib/workshop-rgb/calstate.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const ref = JSON.parse(readFileSync(fx('sphere_ref.json'), 'utf8'));
const TOL = 1e-6; // WEB_SPEC §14.1
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);
const f32 = (p: string) => { const b = readFileSync(fx(p)); const dv = new DataView(b.buffer, b.byteOffset, b.byteLength); return Float64Array.from({ length: b.byteLength / 4 }, (_, i) => dv.getFloat32(4 * i, true)); };

const W = ref.width, H = ref.height;
const img = { width: W, height: H, data: f32('sphere/image_bin16.f32') };
const dark = f32('sphere/dark_bin16.f32');
const satB = new Uint8Array(readFileSync(fx('sphere/sat_bin16.u8')));
const csv = readFileSync(fx('sphere/export1000_01_09_2026.csv'), 'utf8');

// 1. spectrum, separator and header sniffing, per-µm radiance to per nm, colour temperature of the lamp
const d = sniffDelimiter(csv), h = sniffHeader(csv);
check('separator and header lines sniffed as _sniff_delimiter and _sniff_header', d === ref.delimiter && h === ref.skip_header, `(${JSON.stringify(d)}, ${h})`);
const sp = loadSphereSpectrum(csv, { valUnit: 'radiance_um' });
const sumL = sp.L.reduce((a, b) => a + b, 0);
check('spectrum read with the unit conversion', sp.lamNm.length === ref.spectrum.n && sp.lamNm[0] === ref.spectrum.lam0 && rel(sumL, ref.spectrum.sum) < 1e-12);
const Tc = fitColourTemperature(sp.lamNm, sp.L);
check('colour temperature of the lamp equals fit_colour_temperature', Tc === ref.colour_T, `(${Tc} K)`);

// 2. ψ of the acA2040 with its 630 nm cut, ND 0,4 weighted by the sphere spectrum
const aca = PRESETS[1];
const lam = aca.response.lamNm;
const psi = presetPsi(aca);
const Ls = resample(lam, sp.lamNm, sp.L);
const tb = effectiveNdTransmission(lam, psi, Ls, ndTransmission(lam, { kind: 'od', od: ref.nd_od }));
check('effective ND transmission per channel', CH.every((c) => rel(tb[c], ref.tau_bar[c]) < TOL), `(${CH.map((c) => tb[c].toFixed(4)).join(', ')})`);

// 3. disc and mask
const disc = findSphereDisc(img);
check('exit port by moment fit equals find_sphere_disc', rel(disc.cx, ref.disc.cx) < TOL && rel(disc.cy, ref.disc.cy) < TOL && rel(disc.rFit, ref.disc.r_fit) < TOL && disc.mask.reduce((a, b) => a + b, 0) === ref.disc.n,
  `(r = ${disc.rFit.toFixed(2)} bins)`);
const sm = sphereMask(img, dark, disc, satB);
check('usable pixels equal sphere_mask', sm.nDisc === ref.mask.n_disc && sm.nKept === ref.mask.n_kept && CH.every((c) => sm.lostSaturated[c] === ref.mask.lost_saturated[c] && sm.lostDark[c] === ref.mask.lost_dark[c] && rel(sm.medianDn[c], ref.mask.median_dn[c]) < TOL),
  `(${sm.nKept} of ${sm.nDisc})`);

// 4. K_c with median and radial fill
const cal = calibrationConstants(img, dark, ref.texpS, lam, psi, Ls, { tauBar: tb, mask: sm.mask, fill: 'median', disc });
check('band integrals and denominators', CH.every((c) => rel(cal.bandIntegral[c], ref.band_integral[c]) < TOL && rel(cal.denominator[c], ref.denominator[c]) < TOL));
check('K median and relative spread in the disc', CH.every((c) => rel(cal.KDiscMedian[c], ref.K_disc_median[c]) < TOL && rel(cal.KDiscRsd[c], ref.K_disc_rsd[c]) < TOL),
  `(K_G ${cal.KDiscMedian.G.toExponential(4)}, spread ${CH.map((c) => (100 * cal.KDiscRsd[c]).toFixed(1)).join('/')} %)`);
check('K at probe pixels, median fill outside the disc', CH.every((c) => ref.probe.every(([y, x]: number[], k: number) => rel(cal.K[c][y * W + x], ref.K_median_probe[c][k]) < TOL)));
const calR = calibrationConstants(img, dark, ref.texpS, lam, psi, Ls, { tauBar: tb, mask: sm.mask, fill: 'radial', disc });
check('K at probe pixels, radial fill a + bρ² + cρ⁴ outside the disc', CH.every((c) => ref.probe.every(([y, x]: number[], k: number) => rel(calR.K[c][y * W + x], ref.K_radial_probe[c][k]) < TOL)));
check('mask fraction', rel(cal.maskFrac, ref.mask_frac) < 1e-12);

// 5. G near the ceiling (flat_near_ceiling) and the stage 2 decision of a sphere
const near = fractionNearCeiling(img, disc.mask, 65520);
console.log(`      disc above 90 % of full scale (binned)  R ${(100 * near.R).toFixed(1)} %, G ${(100 * near.G).toFixed(1)} %, B ${(100 * near.B).toFixed(1)} %`);
const st = decideCalibration({ source: 'sphere', calibrated: ['R', 'G', 'B'], consistent: true, emissivityKnown: false, ndDeclared: true, texpConfirmed: true, scaleConfirmed: true });
const st2 = decideCalibration({ source: 'sphere', calibrated: ['R', 'G', 'B'], consistent: true, emissivityKnown: false, ndDeclared: false, texpConfirmed: false, scaleConfirmed: true });
check('sphere with a declared ND is absolute, without it relative and with t_exp unconfirmed provisional', st.state === 'absolute' && st.fs === 'ppm' && st2.state === 'relative' && st2.provisional && st2.codes.includes('texp_unconfirmed'));
const st3 = decideCalibration({ source: null, calibrated: [], consistent: true, emissivityKnown: true, texpConfirmed: true, scaleConfirmed: true });
const st4 = decideCalibration({ source: 'file', calibrated: ['G'], consistent: true, emissivityKnown: true, fileAbsolute: true, texpConfirmed: true, scaleConfirmed: true });
check('nothing loaded is uncalibrated, one channel gives cal_one_channel', st3.state === 'uncalibrated' && st3.T === 'warn' && st3.fs === 'no' && st4.state === 'one_channel' && st4.T === 'no');
const st5 = decideCalibration({ source: 'oven', calibrated: ['R', 'G', 'B'], consistent: true, tested: false, emissivityKnown: true, texpConfirmed: true, scaleConfirmed: true });
check('oven with one temperature is untested, T with a warning and f_s provisional (GB3r2-adv-01)', st5.state === 'untested' && st5.T === 'warn' && st5.provisional && st5.codes[0] === 'cal_untested');

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
