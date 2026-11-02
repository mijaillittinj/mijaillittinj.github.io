/**
 * Calibration state of stage 2 (WEB_SPEC §4, etapa 2, and §11.2 of the brief). Decides what can be
 * computed and how it is labelled. No prose here, only codes, the page writes the sentence.
 *
 *   absolute       K_R, K_G (and K_B) pass the oven criterion and ε and windows are known, or a sphere
 *                  with a declared ND filter            T yes, f_s in ppm
 *   relative       K_j pass and ε or a window is unknown (grey)   T yes, f_s relative
 *   inconsistent   the oven criterion fails             T with warning, no f_s   cal_inconsistent
 *   untested       oven with fewer than two usable temperatures in R and G, so the criterion and the
 *                  ratio test could not be run (GB3r2-adv-01)   T with warning, f_s provisional   cal_untested
 *   uncalibrated   nothing loaded, nominal ψ_j          T with warning, no f_s   uncalibrated (no channel calibrated)
 *   one_channel    fewer than two calibrated channels   no T, no f_s             cal_one_channel
 * Provisional is a flag on top: t_exp of the flame or the scale in mm per pixel not confirmed makes f_s
 * provisional (texp_unconfirmed, scale_missing) and leaves T unchanged.
 */
import type { Ch } from './radiometry.ts';

export type CalState = 'absolute' | 'relative' | 'inconsistent' | 'untested' | 'uncalibrated' | 'one_channel';

export interface CalInput {
  /** Where the K_j came from, null when nothing is loaded. */
  source: 'oven' | 'sphere' | 'file' | null;
  /** Channels with a usable K_j. */
  calibrated: Ch[];
  /** Oven criterion of 1a.5 (true for sphere and file calibrations). */
  consistent: boolean;
  /** The oven criterion could be run, two or more usable temperatures in R and G (oven only, default true). */
  tested?: boolean;
  /** Checkbox "ε of the cavity and window transmission are known" (oven), checked by default. */
  emissivityKnown: boolean;
  /** Sphere with a declared ND filter (none, OD or curve). */
  ndDeclared?: boolean;
  /** The file says absolute (for 'file'). */
  fileAbsolute?: boolean;
  texpConfirmed: boolean;
  scaleConfirmed: boolean;
}

export interface CalDecision {
  state: CalState;
  provisional: boolean;
  T: 'yes' | 'warn' | 'no';
  fs: 'ppm' | 'relative' | 'no';
  codes: string[];
  channels: Ch[];
}

export function decideCalibration(x: CalInput): CalDecision {
  const codes: string[] = [];
  const provisional = !x.texpConfirmed || !x.scaleConfirmed;
  if (!x.texpConfirmed) codes.push('texp_unconfirmed');
  if (!x.scaleConfirmed) codes.push('scale_missing');
  if (x.source === null) return { state: 'uncalibrated', provisional, T: 'warn', fs: 'no', codes: ['uncalibrated', ...codes], channels: [] };
  if (!x.consistent) return { state: 'inconsistent', provisional, T: 'warn', fs: 'no', codes: ['cal_inconsistent', ...codes], channels: x.calibrated };
  if (x.source === 'oven' && x.tested === false) {
    const fsU = x.emissivityKnown ? 'ppm' : 'relative';
    return { state: 'untested', provisional: true, T: 'warn', fs: fsU, codes: ['cal_untested', ...codes], channels: x.calibrated };
  }
  if (x.calibrated.length < 2) return { state: 'one_channel', provisional, T: 'no', fs: 'no', codes: ['cal_one_channel', ...codes], channels: x.calibrated };
  const absolute = x.source === 'oven' ? x.emissivityKnown : x.source === 'sphere' ? x.ndDeclared === true : x.fileAbsolute === true;
  return { state: absolute ? 'absolute' : 'relative', provisional, T: 'yes', fs: absolute ? 'ppm' : 'relative', codes, channels: x.calibrated };
}
