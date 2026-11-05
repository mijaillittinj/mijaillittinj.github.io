/**
 * Messages between the page and the coordinator Worker (WEB_SPEC §7). Every request carries `req`,
 * the answer carries the same `req`. Blobs are posted as they are (a reference, no copy), so the Worker
 * reads each file when it needs it and never holds more than its cache. No message carries text for the
 * user, only codes and data, and the page writes the sentence in its language.
 */
import type { BayerPattern } from '../../lib/workshop-rgb/frames.ts';
import type { DarkMode } from '../../lib/workshop-rgb/oven.ts';
import type { EmLaw, Ch } from '../../lib/workshop-rgb/radiometry.ts';
import type { SpectrumUnit } from '../../lib/workshop-rgb/sphere.ts';
import type { AbelRun, CalUse, EmissionNet, EncOverride, GeometryOpts, RefineKind, Sym, TfsNet, TfsOpts } from './engine.ts';

export interface PsiMsg { lamNm: Float64Array; R: Float64Array; G: Float64Array; B: Float64Array }
export interface SeriesMsg { tC: number; frames: { id: string; texpUs: number }[] }
export interface DarkMsg { id: string; texpUs: number }

export type ToWorker =
  | { req: number; type: 'opts'; group: boolean; bayer: BayerPattern; centre: boolean }
  | { req: number; type: 'add'; items: { id: string; path: string; blob: Blob }[] }
  | { req: number; type: 'forget'; ids: string[] }
  | { req: number; type: 'enc'; ids: string[]; enc: EncOverride | null }
  | { req: number; type: 'scan'; id: string; thumb: boolean; keep: boolean }
  | { req: number; type: 'oven.analyse'; series: SeriesMsg[]; darks: DarkMsg[] }
  | { req: number; type: 'oven.plateau'; tC: number; cy: number; cx: number; radius: number }
  | { req: number; type: 'oven.fit'; plateaus: Record<string, { cy: number; cx: number; radius: number }>; psi: PsiMsg; psiEnergy: PsiMsg | null; emLaw: EmLaw; eps: number; sigmaT: number; darkMode: DarkMode; perT: 'slope' | 'band'; calibrateAt: 'highest' | 'all' }
  | { req: number; type: 'flat.build'; flats: string[]; darks: string[]; cx: number; cy: number }
  | { req: number; type: 'sphere.fit'; images: string[]; darks: string[]; tSphS: number; spectrum: string; valUnit: SpectrumUnit; lamUnit: 'nm' | 'um' | 'A'; nd: { kind: 'none' } | { kind: 'od'; od: number } | { kind: 'curve'; text: string }; fill: 'median' | 'radial'; psi: PsiMsg }
  | { req: number; type: 'flame.load'; ids: string[]; quarter: number; mirror: boolean }
  | { req: number; type: 'flame.prepare'; ids: string[]; backgrounds: string[]; texpUs: number; cal: CalUse | null; darkLine: [number, number, number] | null; geom: GeometryOpts }
  | { req: number; type: 'ruler'; id: string }
  | { req: number; type: 'abel.run'; run: AbelRun }
  | { req: number; type: 'abel.lcurve'; rows: number[]; ref: 'G' | 'sum'; sym: Sym }
  | { req: number; type: 'abel.set'; J: Float64Array[][]; reproj: Float64Array[][]; run: AbelRun }
  | { req: number; type: 'tfs.run'; opts: TfsOpts; psi: PsiMsg }
  | { req: number; type: 'band.run'; n: number; seed: number; sigmaT: number; Tcal: number; eps: number; run: AbelRun; tfs: TfsOpts; psi: PsiMsg; sKrel?: [number, number, number]; scaleRel?: number }
  | { req: number; type: 'refine.run'; kind: RefineKind; factor: number; run: AbelRun; tfs: TfsOpts; psi: PsiMsg; base: { T: Float64Array; fs: Float64Array; mask: Uint8Array; chem: Uint8Array; J: Float64Array[][] }; emission: EmissionNet | null; net: TfsNet | null };

export type FromWorker =
  | { req: number; kind: 'ok'; data: unknown }
  | { req: number; kind: 'error'; code: string; detail: Record<string, string | number> }
  | { req: number; kind: 'progress'; done: number; total: number };

export type { Ch };
