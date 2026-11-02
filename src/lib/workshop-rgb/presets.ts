/**
 * Spectral response presets (WEB_SPEC §4, Respuesta espectral). Imported in the bundle, no fetch.
 *   a2A1920-160ucPRO  Sony IMX392 colour, digitised from Basler's figure (manim/data/imx392_response.csv),
 *                     relative response with the mount IR filter already in it, extra IR cut none, read
 *                     per photon (decision after G0, to be confirmed by the oven test of G2), default
 *   acA2040-55uc      responsivity_basler.csv ("Basler, curva de ANNAbel RGB", September 2026 campaign),
 *                     loaded as load_responsivity, with the effective sigmoid cut at 630 nm, width 12 nm
 */
import imx from './data/imx392.json' with { type: 'json' };
import aca from './data/aca2040.json' with { type: 'json' };
import { buildPsi, type IrCut, type Response, type Rgb } from './radiometry.ts';

export interface Preset { id: 'a2A1920-160ucPRO' | 'acA2040-55uc'; sensor: [number, number]; response: Response; basis: 'photon' | 'energy'; irCut: IrCut | null; allowRemoveIr: boolean }

const asResponse = (d: { lamNm: number[]; R: number[]; G: number[]; B: number[]; name: string }, format: 'columns' | 'pairs'): Response => ({
  lamNm: Float64Array.from(d.lamNm), eta: { R: Float64Array.from(d.R), G: Float64Array.from(d.G), B: Float64Array.from(d.B) }, format, name: d.name,
});

export const PRESETS: Preset[] = [
  { id: 'a2A1920-160ucPRO', sensor: [1920, 1200], response: asResponse(imx, 'columns'), basis: 'photon', irCut: null, allowRemoveIr: false },
  { id: 'acA2040-55uc', sensor: [2048, 1100], response: asResponse(aca, 'pairs'), basis: 'photon', irCut: { cutNm: 630, widthNm: 12 }, allowRemoveIr: true },
];

export const DEFAULT_PRESET = PRESETS[0];

/** Preset from the pylon model in a file name (Basler_<model>__…), tag 272, or the sensor size. */
export function presetFor(hint: { name?: string; model?: string; width?: number; height?: number }): { preset: Preset; why: 'name' | 'model' | 'size' | 'default' } {
  const m = hint.name?.match(/Basler_([A-Za-z0-9-]+?)__/);
  for (const p of PRESETS) {
    if (m && m[1] === p.id) return { preset: p, why: 'name' };
    if (hint.model && hint.model.includes(p.id)) return { preset: p, why: 'model' };
  }
  for (const p of PRESETS) {
    const [w, h] = p.sensor;
    if ((hint.width === w && hint.height === h) || (hint.width === h && hint.height === w)) return { preset: p, why: 'size' };
  }
  return { preset: DEFAULT_PRESET, why: 'default' };
}

/** ψ_j of a preset, with an optional IR cut override (null removes it where allowed). */
export function presetPsi(p: Preset, irCut: IrCut | null | undefined = undefined): Rgb<Float64Array> {
  const ir = irCut === undefined ? p.irCut : p.allowRemoveIr || irCut ? irCut : p.irCut;
  return buildPsi(p.response.lamNm, p.response.eta, { basis: p.basis, irCut: ir });
}
