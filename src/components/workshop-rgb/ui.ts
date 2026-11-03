/**
 * Interface of the RGB pyrometry tool (WEB_SPEC §4), plain TypeScript without a framework so the
 * offline single-file copy can be built from the same source with esbuild. `mount` fills the container
 * that Tool.astro writes with five steps in the order of the lab (manim/qa/redesign/DESIGN.md), each with
 * what it is for, its inputs, one main button, its result with one check and a single Options fold.
 * Nothing heavy runs without being asked, and changing a setting marks as outdated only the stages that
 * depend on it. All the heavy work runs in the coordinator Worker (pyro.worker.ts), ANNAbel runs here on
 * WebGPU or on its own pool of helper Workers, created at load time with everything else (§11.5).
 */
import { LinePlot, type Series } from '../../lib/plot.ts';
import { PRESETS, DEFAULT_PRESET, presetFor, type Preset } from '../../lib/workshop-rgb/presets.ts';
import { buildPsi, parseResponseCsv, makeBand, bandX, ratioTable, CH, SP700, type Ch, type EmLaw, type Rgb, type IrCut, type ShortPass } from '../../lib/workshop-rgb/radiometry.ts';
import { scaleTailR, TAIL_CUT_NM, type TailCal } from '../../lib/workshop-rgb/tailcal.ts';
import { decideCalibration, type CalDecision } from '../../lib/workshop-rgb/calstate.ts';
import { T0C, type Aperture, type Plateau, type FrameSignal } from '../../lib/workshop-rgb/oven.ts';
import { LAMBDA_REL_MIN, LAMBDA_REL_MAX } from '../../lib/workshop-rgb/invert.ts';
import type { BayerPattern, ShiftTest } from '../../lib/workshop-rgb/frames.ts';
import { buildProblem, initXavier, layout, trainWithEngine, finishStage1, WorkerPoolEngine, createGpuStage1, webgpuAvailable, helperCount, ITER_LONG, type Projections, type TrainResult, type TrainTick } from '../../lib/workshop-rgb/annabel/index.ts';
import { buildStage2Problem, initStage2, trainStage2, finishStage2, Stage2LocalEngine, STAGE2_DEFAULTS, XCHECK_WARN_P90_K, type KernelTable, type XCheck, type BoundReport, type Stage2Problem } from '../../lib/workshop-rgb/annabel/stage2.ts';
import { Stage2PoolEngine } from '../../lib/workshop-rgb/annabel/workers.ts';
import { createGpuStage2 } from '../../lib/workshop-rgb/annabel/gpu2.ts';
import { defaultHelperFactory } from '../../lib/workshop-rgb/annabel/workers.ts';
import { parseName, parseLog, logFor, parsePfs, parseCrop, cropFor, ignoredSilently, isTiff, ROLES, type Role, type NameInfo, type LogRow, type CropMap } from './names.ts';
import { readZip, writeZip, ZipError } from './zip.ts';
import { npy, npyString, npz } from './npy.ts';
import { sha256Plain } from './sha256.ts';
import { L, num, sig, pct, esc, chName, setLang, getLang, msgHtml, msgText, CH_LETTER, type Lang, type Msg } from './i18n.ts';
import { drawPreview, drawGrey, drawMap, drawBar, circle, line, mapPng, mapAxes, cssVar, theme, type Planes, type MapExtent } from './draw.ts';
import { TOOL_VERSION, TOOL_NAME } from './version.ts';
import type { ToWorker, FromWorker, PsiMsg } from './protocol.ts';
import type { FrameInfo, Prepared, AbelOut, AbelRun, TfsOut, FsMode, CalUse, GeometryOpts, Sym, Vignette, OvenResult } from './engine.ts';
import type { AbelMethod } from '../../lib/workshop-rgb/invert.ts';
// The app trains 3000 iterations; on the example flame the network needs about 30 000 to agree with the
// closed form (148 K median gap at 3000, 12 K at 30 000, WEB_REPORT RGB-ANNAbel etapa 2), so the tool starts there.
const NET_ITERATIONS = 30000;

export interface ExampleSet { file: string; bytes: number; sha256: string; camera: string; available: boolean; preset?: { texpUs: number; mmPerPx: number; zBurnerMm: number; quarter?: number; fineDeg?: number; burnerRow?: number; rowBottom?: number; lambdaRel?: number }; with?: string[] }
export interface MountOptions { lang: Lang; version: string; commit: string; source?: string; built?: string; base: string; examples: ExampleSet[]; pageUrl: string; biases: { key: string; es: string; en: string }[]; ovenFlameBias?: { es: string; en: string }; tailBias?: { es: string; en: string }; tailRange?: { es: string; en: string } }

// ============================================================================ worker plumbing
let worker: Worker;
let reqN = 0;
const pending = new Map<number, { res: (v: unknown) => void; rej: (e: { code: string; detail: Record<string, string | number> }) => void; prog?: (d: number, t: number) => void }>();
type DistOmit<T, K extends keyof T> = T extends unknown ? Omit<T, K> : never;
function ask<R = unknown>(m: DistOmit<ToWorker, 'req'>, prog?: (d: number, t: number) => void): Promise<R> {
  const req = ++reqN;
  return new Promise<R>((res, rej) => {
    pending.set(req, { res: res as (v: unknown) => void, rej, prog });
    worker.postMessage({ ...m, req } as ToWorker);
  });
}
let helpers: Worker[] = [];
let helperIdx = 0;

// ============================================================================ state
type Status = 'pending' | 'reading' | 'ok' | 'error';
interface Item {
  id: string; path: string; blob: Blob; size: number; name: NameInfo; role: Role;
  tC?: number; texpUs?: number; tFrom: 'name' | 'log' | 'typed' | 'none'; texpFrom: 'name' | 'pfs' | 'typed' | 'none' | 'log';
  nameT?: number; logT?: number; status: Status; err?: { code: string; detail: Record<string, string | number> };
  fi?: FrameInfo; sha?: string; shaHow?: string; thumb?: { w: number; h: number; rgba: Uint8ClampedArray };
  pfs?: Record<string, string>; crop?: { x0: number; y0: number }; include: boolean; asym?: number; axis?: number | null; why?: string;
}
interface SeriesView {
  tC: number; aperture: Aperture; plateau: Plateau; texpUs: number[]; plateauId: string; previewTexp: number;
  hotCount: number[]; zeroFrac: number; narrow: boolean; signals: FrameSignal[]; preview: Planes; w: number; h: number;
  prof?: { profile: Plateau['profile']; shape: Record<string, number>; flat: Record<string, number> };
}
interface Calibration {
  format: 'rgbpyro.calibration'; version: 1; tool: { name: string; version: string; commit: string }; created: string;
  mode: 'oven' | 'sphere' | 'file'; K: Rgb<number>; sK: Rgb<number>; sigmaT: number; Tcal: number; eps: number;
  emissivityKnown: boolean; ndDeclared: boolean; absolute: boolean; calibrated: Ch[]; consistent: boolean; tested?: boolean; unit: 'DN12';
  sensor: { width: number; height: number }; frame: { width: number; height: number }; binning: string; scale: number; gx0: number; gy0: number;
  origin: { x0: number; y0: number }; encoding: string; camera: string; preset: string; basis: 'photon' | 'energy'; irCut: IrCut | null; shortPass?: ShortPass | null;
  psi: { name: string; sha256: string; lamNm: number[]; R: number[]; G: number[]; B: number[] };
  vignette: { block: number; bw: number; bh: number; R: string; G: string; B: string } | null; hot: string | null;
  darkLine: [number, number, number] | null; darkLineFit?: Rgb<{ a: number; b: number }> | null; ovenTemps: number[]; texpRangeUs: [number, number]; ratioWorst: number | null;
  plateau?: { cy: number; cx: number; radius: number } | null; darkMode?: string;
  /** IR tail calibration of ψ_R with the oven (tailcal.ts). With applied, K_R = K_G · K_R/K_G of the tail fit and psi is the corrected ψ. */
  tail?: CalTail | null;
}
interface CalTail {
  applied: boolean; s: number; u: number; KRKG: number; uKRKG: number; cutNm: number; temps: number[];
  criterionPct: number; criterionNominalPct: number; criterionFrom: number[]; pass: boolean; strongT: number; strongPct: number;
  driftPct: number; driftNominalPct: number; ratioAfterK: number; ratioBeforeK: number; uFlameK: number; driftBNominalPct: number;
}

/** T and f_s path of stage 5. 'net' is RGB-ANNAbel stage 2 (Magaña et al. 2025), next to the table. */
type TfsPath = 'ratio' | 'fit3' | 'net';
/** Mask bits of TfsOut (engine.ts), saturated height and low signal. */
const M_SAT = 8, M_LOW = 2;
const asTfsPath = (v: unknown): TfsPath => (v === 'fit3' || v === 'net' ? v : 'ratio');
/** What the stage-2 network leaves next to the maps, on the grid of the stage (Nz × Nr, NaN outside). */
interface NetOut { xcheck: XCheck | null; bounds: BoundReport | null; Tcf: Float64Array; fsCf: Float64Array; bestIter: number; completed: number; planned: number; path: string; elapsedS: number; fsRef: number; nTrain: number; nMask: number; rows: number }

const S = {
  o: null as unknown as MountOptions,
  items: [] as Item[], log: [] as LogRow[], crops: {} as CropMap, source: '', example: null as string | null, loadMsgs: [] as Msg[],
  quick: [] as Item[], quickMsgs: [] as Msg[], phoneDismissed: false, busy: '' as string, gpu: null as boolean | null,
  presetId: DEFAULT_PRESET.id as Preset['id'], presetWhy: 'default' as string, presetManual: false, basis: DEFAULT_PRESET.basis as 'photon' | 'energy',
  irOn: true, irCutNm: 630, irWidthNm: 12, sp: false, custom: null as null | { name: string; lamNm: Float64Array; eta: Rgb<Float64Array> },
  eps: 1, sigmaT: 5, known: true, darkMode: 'auto' as 'auto' | 'intercept' | 'strip' | 'files', perT: 'slope' as 'slope' | 'band', calibrateAt: 'highest' as 'highest' | 'all', tailOn: true,
  group: true, bayer: 'RGGB' as BayerPattern, centre: false,
  calPath: 'oven' as 'oven' | 'sphere' | 'file',
  oven: { series: [] as SeriesView[], cur: 0, plateaus: {} as Record<string, { cy: number; cx: number; radius: number }>, darks: [] as { id: string; texpUs: number; level: Rgb<number>; zeroFrac: number }[], ceil12: 4095, result: null as null | (OvenResult & { darkLine: Rgb<{ a: number; b: number }> | null; hot: Int32Array }), view: 1, analysing: false },
  flat: { vignette: null as Vignette | null, w: 0, h: 0 },
  sphere: { spectrum: '' as string, spectrumName: '', valUnit: 'radiance_um' as 'radiance_nm' | 'radiance_um' | 'irradiance_nm' | 'irradiance_um' | 'uW_cm2_nm' | 'counts', lamUnit: 'nm' as 'nm' | 'um' | 'A', nd: '' as '' | 'none' | 'od' | 'curve', od: 0.4, ndCurve: '', ndCurveName: '', fill: 'median' as 'median' | 'radial', tSphUs: NaN, result: null as null | Record<string, unknown> },
  calFileMsgs: [] as Msg[],
  cal: null as Calibration | null,
  flame: { prepErr: '' as string, texpUs: NaN, texpConfirmed: false, times: 'repeat' as 'repeat' | 'instants', instant: '' as string, frames: [] as string[], loaded: false, msgs: [] as Msg[],
    sensorQ: 0, quarter: 0, mirror: false, fineDeg: 0, axisAuto: true, axisManual: NaN, halfWidth: NaN, nr: NaN, rowTop: NaN, rowBottom: NaN, rowsPerHeight: 4,
    burnerOk: false, mmPerPx: NaN, scaleConfirmed: false, scaleUnc: 1, rulerId: '' as string, rulerPts: [] as [number, number][], rulerMm: 10, rulerScale: 1, rulerPreview: null as null | Planes, shift: null as null | (ShiftTest & { pattern: BayerPattern }),
    burnerRow: NaN, zBurner: 0, prepared: null as null | (Prepared & { bgHow?: string; nHot?: number }), height: 0, calUsed: null as CalUse | null, sensorMismatch: false },
  abel: { method: 'tikhonov' as AbelMethod | 'annabel', sym: 'average' as Sym, lambdaRel: 0.1, lambdaAuto: true, lamFlat: false, lcFor: null as unknown, lcSym: '' as string, segments: 12, satLambda: 1e-4, lcurve: null as null | { lambdas: number[]; rho: number[]; eta: number[]; lambda: number; index: number }, out: null as AbelOut | null, outMethod: '' as string, compare: null as null | Record<string, Float64Array>,
    annabel: { running: false, ctrl: null as AbortController | null, tick: null as TrainTick | null, estimate: null as null | { path: string; sPerIter: number }, iterations: ITER_LONG, result: null as TrainResult | null, msgs: [] as Msg[] } },
  tfs: { path: 'ratio' as TfsPath, net: { running: false, ctrl: null as AbortController | null, tick: null as TrainTick | null, estimate: null as null | { path: string; sPerIter: number }, iterations: NET_ITERATIONS, result: null as NetOut | null, msgs: [] as Msg[] }, pair: 'RG' as 'RG' | 'BG' | 'BR', em: 'group' as 'group' | 'coeffs' | 'constant', a: 0.284, b: 0.272, c: 0.057, E: 0.3, sigFrac: 0.02, islandFrac: 0.05, zCut: NaN, chemThr: 0.1, fsChannel: 'RG' as FsMode, out: null as TfsOut | null, dec: null as CalDecision | null },
  res: { Tmin: 1400, Tmax: 2100, Tmanual: false, Tout: 0, bandHow: '' as '' | 'oven' | 'noise' | 'nocal', fsMax: NaN, heights: [] as number[], band: null as null | { Tlo: Float64Array; Thi: Float64Array; fslo: Float64Array; fshi: Float64Array }, seed: 42, nBand: 30, view: 'T' as string },
  stale: {} as Record<number, string>,
  /** Interface only, never saved, open state of the Options fold of each step (DESIGN §0.3). */
  ui: { open: {} as Record<number, boolean> },
  /** Ground truth of the synthetic flame when the loaded set brings verdad_T_fs.csv, for the line of step 5. */
  truth: null as null | { zg: Float64Array; rp: Float64Array; T: Float64Array; F: Float64Array; nr: number },
  /** Step whose loader brought the last files, so its messages are shown there (1 furnace, 3 flame). */
  loadFrom: 1 as 1 | 3,
  runRef: null as null | { name: string; inputs: { name: string; sha256: string; role: string }[] },
  warnings: new Set<string>(),
};
let idN = 0;
let timesSet = false;

const kB = (b: number) => (b >= 1e6 ? `${num(b / 1e6, 1)} MB` : `${num(b / 1e3, 0)} kB`);
const tLabel = (tC: number) => `${num(tC, Number.isInteger(tC) ? 0 : 1)} °C`;
/** Image format in words, no internal codes on screen (GB3r2-adv-11). */
const binName = (b: string) => ({ rgb2x2: L('imágenes RGB agrupadas 2 × 2', 'RGB images binned 2 × 2'), rgb1x1: L('imágenes RGB sin agrupar', 'RGB images without binning'), bayer2x2: L('mosaico Bayer', 'a Bayer mosaic') } as Record<string, string>)[b] ?? b;
const encName = (e: string) => ({ u12_shifted: L('12 bits desplazados', '12-bit, left-shifted'), u12: L('12 bits', '12-bit'), u16: '16 bits', u8: '8 bits', float: L('flotante', 'float') } as Record<string, string>)[e] ?? e;
const roleName = (r: Role) => ({ oven: L('horno', 'furnace'), dark: L('oscuro', 'dark'), flat: L('campo plano', 'flat field'), sphere: L('esfera', 'sphere'), flame: L('llama', 'flame'), background: L('fondo', 'background'), ruler: L('regla', 'ruler'), ignore: L('ignorar', 'ignore') } as Record<Role, string>)[r];
const subs = (t: string) => t.replace(/f_s/g, 'f<sub>s</sub>').replace(/E_G\/E_R/g, 'E<sub>G</sub>/E<sub>R</sub>').replace(/K_j/g, 'K<sub>j</sub>');
const fin = (x: number) => Number.isFinite(x);
/** Number with its sign, +342 or −29. */
const sgn = (x: number, d = 0) => (fin(x) && x > 0 ? `+${num(x, d)}` : num(x, d));

// ============================================================================ spectral response
function preset(): Preset { return PRESETS.find((p) => p.id === S.presetId) ?? DEFAULT_PRESET; }
function irCutNow(): IrCut | null {
  const p = preset();
  if (S.custom) return null;
  if (!p.allowRemoveIr) return p.irCut;
  return S.irOn ? { cutNm: S.irCutNm, widthNm: S.irWidthNm } : null;
}
/** Short-pass filter declared in front of the lens (preset "pasa-cortos 700 nm", decision after G4 r3). It enters τ(λ). */
const spNow = (): ShortPass | null => (S.sp ? SP700 : null);
const spName = () => (S.sp ? ' SP 700 nm OD 3' : '');
/** ψ_j used downstream of the calibration, the corrected ψ_R when the oven calibration calibrated its IR tail. */
function psiNow(basis = S.basis): { lamNm: Float64Array; psi: Rgb<Float64Array>; name: string } {
  const p = psiBase(basis);
  const t = S.cal?.mode === 'oven' ? S.cal.tail : null;
  if (!t?.applied) return p;
  return { lamNm: p.lamNm, psi: scaleTailR(p.lamNm, p.psi, t.s, t.cutNm), name: `${p.name}${tailName(t.s)}` };
}
const tailName = (s: number) => L(` · cola de R sobre 700 nm × ${num(s, 3)}`, ` · R tail above 700 nm × ${num(s, 3)}`);
/** ψ_j from the response settings, before any correction of the oven (what the oven fit and the sphere use). */
function psiBase(basis = S.basis): { lamNm: Float64Array; psi: Rgb<Float64Array>; name: string } {
  if (S.cal && S.cal.mode === 'file') {
    const c = S.cal.psi;
    return { lamNm: Float64Array.from(c.lamNm), psi: { R: Float64Array.from(c.R), G: Float64Array.from(c.G), B: Float64Array.from(c.B) }, name: c.name };
  }
  if (S.custom) return { lamNm: S.custom.lamNm, psi: buildPsi(S.custom.lamNm, S.custom.eta, { basis, shortPass: spNow() }), name: `${S.custom.name}${spName()}` };
  const p = preset();
  return { lamNm: p.response.lamNm, psi: buildPsi(p.response.lamNm, p.response.eta, { basis, irCut: irCutNow(), shortPass: spNow() }), name: `${p.id}${irCutNow() ? ` IR ${irCutNow()!.cutNm} nm` : ''}${spName()}` };
}
const psiMsg = (basis = S.basis, base = false): PsiMsg => { const p = base ? psiBase(basis) : psiNow(basis); return { lamNm: p.lamNm, R: p.psi.R, G: p.psi.G, B: p.psi.B }; };
function psiSha(p = psiNow()): string {
  const all = new Float64Array(p.lamNm.length * 4);
  all.set(p.lamNm, 0); all.set(p.psi.R, p.lamNm.length); all.set(p.psi.G, 2 * p.lamNm.length); all.set(p.psi.B, 3 * p.lamNm.length);
  return sha256Plain(new Uint8Array(all.buffer));
}
function emLaw(): EmLaw {
  return S.tfs.em === 'group' ? { kind: 'group' } : S.tfs.em === 'constant' ? { kind: 'constant', E: S.tfs.E } : { kind: 'coeffs', a: S.tfs.a, b: S.tfs.b, c: S.tfs.c };
}
function cameraOf(items: Item[]): { cam: string; why: string } | null {
  for (const it of items) {
    if (it.name.model) return { cam: it.name.model, why: L('el nombre de pylon', 'the pylon name') };
    if (it.fi?.model) return { cam: it.fi.model, why: L('la etiqueta 272 del TIFF', 'TIFF tag 272') };
  }
  for (const it of items) if (it.fi) {
    const r = presetFor({ width: it.fi.width, height: it.fi.height });
    if (r.why === 'size') return { cam: r.preset.id, why: L('el tamaño del sensor', 'the sensor size') };
  }
  const c = Object.values(S.crops).find((x) => (x as { camera?: string }).camera) as { camera?: string } | undefined;
  if (c?.camera) return { cam: c.camera, why: 'recorte.json' };
  return null;
}
function autoPreset() {
  if (S.presetManual) return;
  const c = cameraOf(S.items);
  if (!c) return;
  const p = PRESETS.find((x) => c.cam.includes(x.id) || x.id.includes(c.cam));
  if (p && p.id !== S.presetId) { S.presetId = p.id; invalidate(1, 'psi'); }
  S.presetWhy = c.why;
}

// ============================================================================ outdated stages
function invalidate(from: number, reason: string) {
  for (let s = from; s <= 6; s++) {
    const has = s === 1 ? !!S.oven.result || !!S.cal : s === 3 ? !!S.flame.prepared : s === 4 ? !!S.abel.out : s === 5 || s === 6 ? !!S.tfs.out : false;
    if (has) S.stale[s] = reason;
  }
  if (from <= 6) S.res.band = null;
}

// ============================================================================ files in
interface InFile { path: string; blob: Blob }
async function walkEntry(e: FileSystemEntry, prefix: string, out: InFile[]) {
  if (e.isFile) {
    const f: File = await new Promise((res, rej) => (e as FileSystemFileEntry).file(res, rej));
    out.push({ path: prefix + e.name, blob: f });
  } else if (e.isDirectory) {
    const r = (e as FileSystemDirectoryEntry).createReader();
    for (;;) {
      const batch: FileSystemEntry[] = await new Promise((res, rej) => r.readEntries(res, rej));
      if (!batch.length) break;
      for (const c of batch) await walkEntry(c, prefix + e.name + '/', out);
    }
  }
}
async function expand(files: InFile[], msgs: Msg[]): Promise<InFile[]> {
  const out: InFile[] = [];
  for (const f of files) {
    if (/\.zip$/i.test(f.path)) {
      try {
        const root = f.path.replace(/\.zip$/i, '').split('/').slice(0, -1).join('/');
        for (const e of await readZip(f.blob)) {
          const p = root ? `${root}/${e.path}` : e.path;
          if (ignoredSilently(p)) continue;
          out.push({ path: p, blob: await e.blob() });
        }
      } catch (e) {
        msgs.push({ code: e instanceof ZipError ? (e.code === 'zip_unsupported' ? 'zip_unsupported' : 'zip_bad') : 'truncated', files: [f.path] });
      }
    } else out.push(f);
  }
  return out.filter((f) => !ignoredSilently(f.path));
}
function filesFromInput(inp: HTMLInputElement): InFile[] {
  return Array.from(inp.files ?? []).map((f) => ({ path: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name, blob: f }));
}

async function loadFiles(files: InFile[], source: string, example: string | null = null) {
  S.loadMsgs = [];
  const all = await expand(files, S.loadMsgs);
  S.source = source; S.example = example;
  const pfsByDir = new Map<string, Record<string, string>>();
  for (const f of all) {
    const base = f.path.split('/').pop()!;
    if (/^registro\.csv$/i.test(base)) S.log = parseLog(await f.blob.text());
    else if (/^recorte\.json$/i.test(base)) { try { Object.assign(S.crops, parseCrop(await f.blob.text())); } catch { /* the table shows the crop as unknown */ } }
    else if (/\.pfs$/i.test(base)) pfsByDir.set(f.path.split('/').slice(0, -1).join('/'), parsePfs(await f.blob.text()));
    else if (/^verdad_T_fs\.csv$/i.test(base)) S.truth = parseTruth(await f.blob.text());
    else if (/\.csv$/i.test(base) && /(^|\/)(esfera|sphere)\//i.test(f.path)) { S.sphere.spectrum = await f.blob.text(); S.sphere.spectrumName = f.path; }
    else if (/run\.json$/i.test(base)) await loadRun(f.blob, base);
  }
  const added: Item[] = [];
  for (const f of all.filter((x) => isTiff(x.path))) {
    const info = parseName(f.path);
    const it: Item = {
      id: `f${++idN}`, path: f.path, blob: f.blob, size: f.blob.size, name: info, role: info.role,
      tC: info.tC, texpUs: info.texpUs, nameT: info.tC, tFrom: info.tC !== undefined ? 'name' : 'none', texpFrom: info.texpUs !== undefined ? 'name' : 'none',
      status: 'pending', include: true, crop: cropFor(f.path, S.crops),
    };
    const lr = logFor(f.path, S.log);
    if (lr?.tC !== undefined) { it.logT = lr.tC; it.tC = lr.tC; it.tFrom = 'log'; }
    if (lr?.texpUs !== undefined && it.texpUs === undefined) { it.texpUs = lr.texpUs; it.texpFrom = 'log'; }
    const dir = f.path.split('/').slice(0, -1).join('/');
    let pfs: Record<string, string> | undefined;
    for (const [d, p] of pfsByDir) if (dir === d || dir.startsWith(d + '/')) pfs = p;
    if (pfs) {
      it.pfs = pfs;
      const e = Number(pfs.ExposureTime);
      if (fin(e) && e > 0) {
        if (it.texpUs !== undefined && Math.abs(e - it.texpUs) > 0.5) S.loadMsgs.push({ code: 'pfs_texp', vars: { a: e, b: it.texpUs, file: f.path } });
        it.texpUs = e; it.texpFrom = 'pfs';
      }
    }
    added.push(it);
  }
  S.items.push(...added);
  sortItems();
  await ask({ type: 'add', items: added.map((it) => ({ id: it.id, path: it.path, blob: it.blob })) });
  invalidate(1, 'files');
  renderAll();
  await scanItems(added);
  await harmonise();
  autoPreset();
  checkRun();
  renderAll();
  if (S.items.some((it) => it.role === 'oven' && it.status === 'ok')) await analyseOven();
  if (added.some((it) => it.role === 'flame' && it.status === 'ok')) await loadFlame();
  if (added.some((it) => it.role === 'ruler' && it.status === 'ok') && !S.items.some((x) => x.id === S.flame.rulerId && x.role === 'ruler')) { S.flame.rulerId = added.find((it) => it.role === 'ruler')!.id; await loadRuler(); }
  if (added.some((it) => it.role === 'sphere') && !added.some((it) => it.role === 'oven')) { S.calPath = 'sphere'; S.ui.open[1] = true; const s = added.find((it) => it.role === 'sphere'); if (s?.texpUs) S.sphere.tSphUs = s.texpUs; }
  renderAll();
  announce(added.some((it) => it.role === 'flame') && !added.some((it) => it.role === 'oven' || it.role === 'sphere') ? 3 : 1);
}
/** verdad_T_fs.csv of the synthetic flame (z_mm, r_mm, T_K, fs_ppm on a regular grid), r = 0 repeats the first column. */
function parseTruth(text: string): typeof S.truth {
  try {
    const rows = text.split(/\r?\n/).slice(1).filter((l) => l.trim()).map((l) => l.split(',').map(Number));
    const zg = Float64Array.from([...new Set(rows.map((r) => r[0]))].sort((a, b) => a - b));
    const rg = [...new Set(rows.map((r) => r[1]))].sort((a, b) => a - b);
    const nr = rg.length;
    if (zg.length < 2 || nr < 2 || rows.length !== zg.length * nr) return null;
    const T = new Float64Array(zg.length * (nr + 1)), F = new Float64Array(zg.length * (nr + 1));
    rows.forEach((r, q) => { const k = Math.floor(q / nr), i = q % nr; T[k * (nr + 1) + i + 1] = r[2]; F[k * (nr + 1) + i + 1] = r[3]; if (i === 0) { T[k * (nr + 1)] = r[2]; F[k * (nr + 1)] = r[3]; } });
    return { zg, rp: Float64Array.from([0, ...rg]), T, F, nr };
  } catch { return null; }
}
function sortItems() {
  const order = (r: Role) => ROLES.indexOf(r);
  S.items.sort((a, b) => order(a.role) - order(b.role) || (a.tC ?? 1e9) - (b.tC ?? 1e9) || (a.texpUs ?? 0) - (b.texpUs ?? 0) || a.path.localeCompare(b.path));
}
async function scanItems(list: Item[]) {
  let k = 0;
  for (const it of list) {
    k++;
    S.busy = L(`Leyendo ${k} de ${list.length}`, `Reading ${k} of ${list.length}`);
    it.status = 'reading';
    renderBusy();
    try {
      const r = await ask<{ info: FrameInfo; sha: string; shaHow: string; thumb: Item['thumb'] }>({ type: 'scan', id: it.id, thumb: it.role === 'flame', keep: it.role !== 'ignore' });
      it.fi = r.info; it.sha = r.sha; it.shaHow = r.shaHow; if (r.thumb) it.thumb = r.thumb; it.status = 'ok';
    } catch (e) { it.status = 'error'; it.err = e as Item['err']; }
    if (k % 6 === 0 || k === list.length) { renderStage1(); renderStage3(); }
  }
  S.busy = '';
}
/** The encoding of a series is decided with its brightest frame and applied to its darks (WEB_SPEC §5.3). */
async function harmonise() {
  const groups: [Item[], Item[]][] = [
    [S.items.filter((it) => it.role === 'oven'), S.items.filter((it) => it.role === 'dark' && !/(campo_plano|flat|esfera|sphere)/i.test(it.path))],
    [S.items.filter((it) => it.role === 'flat'), S.items.filter((it) => it.role === 'dark' && /(campo_plano|flat)/i.test(it.path))],
    [S.items.filter((it) => it.role === 'sphere'), S.items.filter((it) => it.role === 'dark' && /(esfera|sphere)/i.test(it.path))],
    [S.items.filter((it) => it.role === 'flame'), S.items.filter((it) => it.role === 'background')],
  ];
  const redo: Item[] = [];
  for (const [main, deps] of groups) {
    const ok = main.filter((it) => it.fi && it.fi.bits === 16);
    if (!ok.length) continue;
    // the encoding comes from the single frames, an average of shifted frames is not a multiple of 16 and saturated pixels push it up (GB3r2-adv-14)
    const singles = ok.filter((it) => !isAverage(it));
    const b = (singles.length ? singles : ok).reduce((x, y) => (Math.max(...y.fi!.max12) / y.fi!.ceil12 > Math.max(...x.fi!.max12) / x.fi!.ceil12 ? y : x));
    const enc = { encoding: b.fi!.encoding, toDn12: b.fi!.toDn12, ceiling: b.fi!.ceiling };
    const odd = [...main, ...deps].filter((it) => it.fi && it.fi.bits === 16 && it.fi.encoding !== enc.encoding);
    if (!odd.length) continue;
    await ask({ type: 'enc', ids: odd.map((it) => it.id), enc });
    redo.push(...odd);
  }
  if (redo.length) await scanItems(redo);
}

/** The sets an example button loads, the furnace of `with` first and then the set itself. */
function exampleSets(ex: ExampleSet): ExampleSet[] {
  return [...(ex.with ?? []).map((f) => S.o.examples.find((x) => x.file === f)).filter((x): x is ExampleSet => !!x), ex];
}
async function loadExample(ex: ExampleSet) {
  const sets = exampleSets(ex);
  if (sets.some((x) => !x.available)) { S.loadMsgs = [{ code: 'example_missing' }]; renderStage0(); return; }
  S.busy = L('Bajando el juego de ejemplo', 'Downloading the example set');
  renderBusy();
  try {
    const files: InFile[] = [];
    for (const x of sets) {
      const r = await fetch(`${S.o.base}teaching/workshop-rgb-pyrometry/files/${x.file}`);
      if (!r.ok) throw new Error('http');
      files.push({ path: x.file, blob: await r.blob() });
    }
    S.busy = '';
    const cleared = S.items.length > 0;
    if (cleared) await clearTable();
    // each example starts from the default view, so one set does not inherit the rotation or the rows of another
    const pr = ex.preset;
    Object.assign(S.flame, { quarter: pr?.quarter ?? 0, mirror: false, fineDeg: pr?.fineDeg ?? 0, axisAuto: true, axisManual: NaN, rowTop: NaN, rowBottom: pr?.rowBottom ?? NaN, burnerRow: pr?.burnerRow ?? NaN, burnerOk: false,
      texpUs: NaN, texpConfirmed: false, mmPerPx: NaN, scaleConfirmed: false, zBurner: 0 });
    Object.assign(S.abel, { lambdaRel: pr?.lambdaRel ?? 0.1, lambdaAuto: pr?.lambdaRel === undefined, lamFlat: pr?.lambdaRel !== undefined, lcurve: null, lcFor: null });
    S.loadFrom = 1;
    await loadFiles(files, ex.file, ex.file);
    if (cleared) { S.loadMsgs.unshift({ code: 'example_cleared', level: 'info' }); renderAll(); }
    // the documented case of the set, so the result compares with its ground truth (GB3r2-estudiante-03)
    if (pr) {
      Object.assign(S.flame, { texpUs: pr.texpUs, texpConfirmed: true, mmPerPx: pr.mmPerPx, scaleConfirmed: true, zBurner: pr.zBurnerMm });
      S.flame.msgs = [{ code: 'example_preset', vars: { a: num(pr.texpUs), b: num(pr.mmPerPx, 2), c: num(pr.zBurnerMm, 2) }, level: 'info' }, ...S.flame.msgs];
      invalidate(3, 'flame');
      renderAll();
    }
  } catch {
    S.busy = '';
    S.loadMsgs = [{ code: 'example_failed' }];
    renderAll();
  }
}

// ============================================================================ quick review (stage 0)
async function quickReview(files: InFile[]) {
  S.quickMsgs = [];
  const all = await expand(files, S.quickMsgs);
  const list: Item[] = all.filter((f) => isTiff(f.path)).map((f) => ({ id: `q${++idN}`, path: f.path, blob: f.blob, size: f.blob.size, name: parseName(f.path), role: parseName(f.path).role, tFrom: 'none', texpFrom: 'none', status: 'pending', include: true }));
  S.quick = list;
  await ask({ type: 'add', items: list.map((it) => ({ id: it.id, path: it.path, blob: it.blob })) });
  renderQuick();
  for (const it of list) {
    try { const r = await ask<{ info: FrameInfo; sha: string }>({ type: 'scan', id: it.id, thumb: false, keep: false }); it.fi = r.info; it.sha = r.sha; it.status = 'ok'; }
    catch (e) { it.status = 'error'; it.err = e as Item['err']; }
    renderQuick();
  }
  await ask({ type: 'forget', ids: list.map((it) => it.id) });
  const q = document.getElementById('wr-quick');
  const a = document.getElementById('wr-announce'), al = document.getElementById('wr-alert');
  const nw = q ? q.querySelectorAll('.wr-msg').length : 0;
  if (a) a.textContent = L(`Revisión rápida lista, ${list.length} ${list.length === 1 ? 'archivo' : 'archivos'}, ${nw} ${nw === 1 ? 'aviso' : 'avisos'}.`, `Quick check done, ${list.length} ${list.length === 1 ? 'file' : 'files'}, ${nw} ${nw === 1 ? 'warning' : 'warnings'}.`);
  if (al) al.textContent = list.filter((it) => it.status === 'error').map((it) => `${it.path}. ${msgText({ code: it.err?.code ?? 'not_tiff' }).text}`).join(' ');
}

// ============================================================================ oven
function ovenSeriesInput() {
  const ov = S.items.filter((it) => it.role === 'oven' && it.status === 'ok' && it.tC !== undefined && it.texpUs !== undefined);
  const temps = [...new Set(ov.map((it) => it.tC!))].sort((a, b) => a - b);
  return temps.map((tC) => ({ tC, frames: ov.filter((it) => it.tC === tC).map((it) => ({ id: it.id, texpUs: it.texpUs! })) }));
}
function darkItems() { return S.items.filter((it) => it.role === 'dark' && it.status === 'ok' && it.texpUs !== undefined && !/(campo_plano|flat|esfera|sphere)/i.test(it.path)); }
async function analyseOven() {
  const series = ovenSeriesInput();
  if (!series.length) return;
  S.oven.analysing = true;
  S.busy = L('Buscando la abertura y la meseta', 'Finding the aperture and the plateau');
  renderAll();
  try {
    const r = await ask<{ series: SeriesView[]; darks: typeof S.oven.darks; ceil12: number }>({ type: 'oven.analyse', series, darks: darkItems().map((it) => ({ id: it.id, texpUs: it.texpUs! })) }, (d, t) => { S.busy = L(`Meseta, ${d} de ${t} imágenes`, `Plateau, ${d} of ${t} images`); renderBusy(); });
    S.oven.series = r.series; S.oven.darks = r.darks; S.oven.ceil12 = r.ceil12;
    S.oven.plateaus = {};
    for (const s of r.series) S.oven.plateaus[String(s.tC)] = { cy: s.plateau.cy, cx: s.plateau.cx, radius: s.plateau.radius };
    S.oven.cur = Math.max(0, r.series.length - 1);
    S.oven.result = null;
  } catch (e) { S.loadMsgs.push({ code: (e as { code: string }).code }); }
  S.oven.analysing = false;
  S.busy = '';
  renderAll();
}
function darkModeNow(): 'intercept' | 'strip' | 'files' {
  if (S.darkMode !== 'auto') return S.darkMode;
  return darkItems().length ? 'files' : 'intercept';
}
async function fitOvenNow() {
  if (!S.oven.series.length) return;
  S.busy = L('Ajustando K por canal', 'Fitting K per channel');
  renderBusy();
  try {
    const other = S.basis === 'photon' ? 'energy' : 'photon';
    const r = await ask<OvenResult & { darkLine: Rgb<{ a: number; b: number }> | null; hot: Int32Array }>({
      type: 'oven.fit', plateaus: S.oven.plateaus, psi: psiMsg(S.basis, true), psiEnergy: psiMsg(other, true), emLaw: emLaw(), eps: S.eps, sigmaT: S.sigmaT,
      darkMode: darkModeNow(), perT: S.perT, calibrateAt: S.calibrateAt,
    }, (d, t) => { S.busy = L(`Señal de la meseta, ${d} de ${t}`, `Plateau signal, ${d} of ${t}`); renderBusy(); });
    S.oven.result = r;
    delete S.stale[1];
    setCalFromOven();
  } catch (e) { S.loadMsgs.push({ code: (e as { code: string }).code }); }
  S.busy = '';
  renderAll();
  announce(1);
}
const b64 = (a: ArrayBufferView) => { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000)); return btoa(s); };
const unb64F32 = (s: string) => { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); };
const unb64I32 = (s: string) => { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Int32Array(u.buffer); };
function basePsiRecord(p = psiNow()) {
  return { name: p.name, sha256: psiSha(p), lamNm: Array.from(p.lamNm), R: Array.from(p.psi.R), G: Array.from(p.psi.G), B: Array.from(p.psi.B) };
}
function sensorOf(it: Item | undefined): { width: number; height: number } {
  const c = it ? (cropFor(it.path, S.crops) as { sensor?: [number, number] } | undefined) : undefined;
  if (c?.sensor) return { width: c.sensor[1], height: c.sensor[0] };
  const p = preset();
  return { width: p.sensor[0], height: p.sensor[1] };
}
function setCalFromOven() {
  const r = S.oven.result;
  if (!r) return;
  const f = r.fit;
  const topT = f.Ttop - T0C;
  const top = S.oven.series.find((s) => Math.abs(s.tC - topT) < 1e-6) ?? S.oven.series[S.oven.series.length - 1];
  const it = S.items.find((x) => x.id === top.plateauId)!;
  const fi = it.fi!;
  const crop = it.crop ?? { x0: 0, y0: 0 };
  const allT = S.items.filter((x) => x.role === 'oven' && x.texpUs !== undefined).map((x) => x.texpUs!);
  const dl = r.darkLine;
  // IR tail of ψ_R calibrated with the oven, by default with three or more temperatures (professor after G5)
  const tc = r.tail;
  const tail: CalTail | null = tc && tc.ok ? {
    applied: S.tailOn, s: tc.s, u: tc.u.s, KRKG: tc.KRKG, uKRKG: tc.u.KRKG, cutNm: TAIL_CUT_NM, temps: tc.temps,
    criterionPct: tc.criterion.worstPct, criterionNominalPct: tc.criterion.nominalPct, criterionFrom: tc.criterion.from, pass: tc.criterion.pass,
    strongT: tc.criterion.strongT, strongPct: tc.criterion.strongPct, driftPct: tc.driftPct, driftNominalPct: tc.driftNominalPct,
    ratioAfterK: tc.ratioTest.afterK, ratioBeforeK: tc.ratioTest.beforeK, uFlameK: tc.u.flameK, driftBNominalPct: tc.driftNominalByChannelPct.B,
  } : null;
  const on = !!tail?.applied;
  const base = psiBase();
  const psiRec = on ? basePsiRecord({ lamNm: base.lamNm, psi: scaleTailR(base.lamNm, base.psi, tail!.s, tail!.cutNm), name: `${base.name}${tailName(tail!.s)}` }) : basePsiRecord(base);
  // with the tail, K_G stays the oven K_G (nominal ψ_G) and K_R = K_G · K_R/K_G of the tail fit; blue does not pass and stays out
  const KG = f.channels.G.K, sKG = f.channels.G.sK;
  const K = on ? { R: KG * tail!.KRKG, G: KG, B: f.channels.B.K } : { R: f.channels.R.K, G: f.channels.G.K, B: f.channels.B.K };
  const sK = on ? { R: KG * tail!.KRKG * Math.hypot(sKG / KG, tail!.uKRKG / tail!.KRKG), G: sKG, B: f.channels.B.sK } : { R: f.channels.R.sK, G: f.channels.G.sK, B: f.channels.B.sK };
  const consistent = on ? tail!.pass : f.consistent;
  const calibrated: Ch[] = on ? (tail!.pass ? ['R', 'G'] : []) : f.calibrated;
  S.cal = {
    format: 'rgbpyro.calibration', version: 1, tool: { name: TOOL_NAME, version: TOOL_VERSION, commit: S.o.commit }, created: new Date().toISOString(),
    mode: 'oven', K, sK,
    sigmaT: S.sigmaT, Tcal: f.channels.R.Tcal, eps: S.eps, emissivityKnown: S.known, ndDeclared: false, absolute: S.known, calibrated, consistent, tested: on ? true : f.ratio.RG.tested, unit: 'DN12',
    sensor: sensorOf(it), frame: { width: fi.width, height: fi.height }, binning: fi.binning, scale: fi.scale, gx0: fi.gx0, gy0: fi.gy0, origin: crop, encoding: fi.encoding,
    camera: cameraOf(S.items)?.cam ?? preset().id, preset: S.custom ? S.custom.name : preset().id, basis: S.basis, irCut: irCutNow(), shortPass: spNow(), psi: psiRec,
    vignette: S.flat.vignette ? { block: S.flat.vignette.block, bw: S.flat.vignette.bw, bh: S.flat.vignette.bh, R: b64(S.flat.vignette.R), G: b64(S.flat.vignette.G), B: b64(S.flat.vignette.B) } : null,
    hot: r.hot.length ? b64(r.hot) : null, darkLine: null, darkLineFit: dl, ovenTemps: S.oven.series.map((s) => s.tC),
    texpRangeUs: [Math.min(...allT), Math.max(...allT)], ratioWorst: r.ratioWorst, plateau: S.oven.plateaus[String(top.tC)], darkMode: darkModeNow(), tail,
  };
  invalidate(2, 'cal');
}
async function buildFlat() {
  const flats = S.items.filter((it) => it.role === 'flat' && it.status === 'ok');
  if (!flats.length) return;
  const darks = S.items.filter((it) => it.role === 'dark' && it.status === 'ok' && /(campo_plano|flat)/i.test(it.path));
  const top = S.oven.series[S.oven.series.length - 1];
  const pl = top ? S.oven.plateaus[String(top.tC)] : null;
  try {
    const r = await ask<{ vignette: Vignette; w: number; h: number }>({ type: 'flat.build', flats: flats.map((x) => x.id), darks: darks.map((x) => x.id), cx: pl?.cx ?? 0, cy: pl?.cy ?? 0 });
    S.flat = { vignette: r.vignette, w: r.w, h: r.h };
    if (S.cal?.mode === 'oven') setCalFromOven();
    invalidate(3, 'cal');
  } catch (e) { S.loadMsgs.push({ code: (e as { code: string }).code }); }
  renderAll();
}

// ============================================================================ sphere
async function fitSphere() {
  const imgs = S.items.filter((it) => it.role === 'sphere' && it.status === 'ok');
  const darks = S.items.filter((it) => it.role === 'dark' && it.status === 'ok' && /(esfera|sphere)/i.test(it.path));
  const msgs: Msg[] = [];
  if (!S.sphere.spectrum) msgs.push({ code: 'spectrum_missing' });
  if (!S.sphere.nd) msgs.push({ code: 'nd_required' });
  if (!imgs.length) msgs.push({ code: 'name_unread', vars: { file: 'esfera/' } });
  if (msgs.length) { S.sphere.result = { msgs }; renderStage1(); return; }
  const tS = (fin(S.sphere.tSphUs) ? S.sphere.tSphUs : imgs[0].texpUs ?? NaN) * 1e-6;
  S.busy = L('Calculando la esfera', 'Computing the sphere');
  renderBusy();
  try {
    const nd = S.sphere.nd === 'none' ? { kind: 'none' as const } : S.sphere.nd === 'od' ? { kind: 'od' as const, od: S.sphere.od } : { kind: 'curve' as const, text: S.sphere.ndCurve };
    const r = await ask<Record<string, unknown>>({ type: 'sphere.fit', images: imgs.map((x) => x.id), darks: darks.map((x) => x.id), tSphS: tS, spectrum: S.sphere.spectrum, valUnit: S.sphere.valUnit, lamUnit: S.sphere.lamUnit, nd, fill: S.sphere.fill, psi: psiMsg(S.basis, true) });
    S.sphere.result = r;
    const K = r.K as Rgb<number>, rsd = r.rsd as Rgb<number>, vig = r.vignette as Vignette;
    const it = imgs[0], fi = it.fi!;
    S.cal = {
      format: 'rgbpyro.calibration', version: 1, tool: { name: TOOL_NAME, version: TOOL_VERSION, commit: S.o.commit }, created: new Date().toISOString(),
      mode: 'sphere', K, sK: { R: K.R * rsd.R, G: K.G * rsd.G, B: K.B * rsd.B }, sigmaT: NaN, Tcal: NaN, eps: 1, emissivityKnown: false, ndDeclared: true, absolute: true,
      calibrated: ['R', 'G', 'B'], consistent: true, unit: 'DN12', sensor: sensorOf(it), frame: { width: fi.width, height: fi.height }, binning: fi.binning, scale: fi.scale, gx0: fi.gx0, gy0: fi.gy0,
      origin: it.crop ?? { x0: 0, y0: 0 }, encoding: fi.encoding, camera: cameraOf([it])?.cam ?? preset().id, preset: S.custom ? S.custom.name : preset().id, basis: S.basis, irCut: irCutNow(), shortPass: spNow(), psi: basePsiRecord(psiBase()),
      vignette: { block: vig.block, bw: vig.bw, bh: vig.bh, R: b64(vig.R), G: b64(vig.G), B: b64(vig.B) }, hot: null, darkLine: null, ovenTemps: [], texpRangeUs: [tS * 1e6, tS * 1e6], ratioWorst: null,
    };
    invalidate(2, 'cal');
  } catch (e) { S.sphere.result = { msgs: [{ code: (e as { code: string }).code }] }; }
  S.busy = '';
  renderAll();
  announce(1);
}

// ============================================================================ calibration file
async function loadCalFile(f: File) {
  S.calFileMsgs = [];
  try {
    const j = JSON.parse(await f.text()) as Record<string, unknown>;
    let cal: Calibration | null = null;
    if (j.format === 'rgbpyro.calibration') { cal = j as unknown as Calibration; if (cal.unit !== 'DN12') throw new Error('cal_bad'); }
    else {
      // ANNAbel calibration converted with convertir_calibracion_annabel.py, K by channel, ψ and G_tab
      const K0 = (j.K ?? j.K_disc_median ?? { R: j.K_R, G: j.K_G, B: j.K_B }) as Record<string, number>;
      // ANNAbel works in 16-bit DN (12 bits shifted, × 16) and the tool in 12-bit DN
      const unit = typeof j.unit === 'string' ? j.unit : '';
      const f16 = unit === 'DN12' ? 1 : 1 / 16;
      S.calFileMsgs.push({ code: unit === 'DN12' ? 'cal_unit_dn12' : unit === 'DN16' ? 'cal_unit_dn16' : 'cal_unit_missing', level: unit ? 'info' : 'warn' });
      const Kraw = { R: K0.R * f16, G: K0.G * f16, B: K0.B * f16 };
      const psi = (j.psi ?? {}) as Record<string, number[]>;
      const lam = (j.lambda_nm ?? j.lamNm ?? psi.lambda_nm ?? psi.lamNm) as number[] | undefined;
      if (!Kraw || !fin(Kraw.R) || !lam || !psi.R) throw new Error('cal_bad');
      cal = {
        format: 'rgbpyro.calibration', version: 1, tool: { name: 'annabel', version: String(j.format_version ?? ''), commit: '' }, created: String(j.created ?? ''),
        mode: 'file', K: { R: Kraw.R, G: Kraw.G, B: Kraw.B }, sK: { R: NaN, G: NaN, B: NaN }, sigmaT: NaN, Tcal: NaN, eps: 1, emissivityKnown: false, ndDeclared: true, absolute: j.absolute !== false,
        calibrated: ['R', 'G', 'B'], consistent: true, unit: 'DN12', sensor: { width: Number((j.sensor as number[] | undefined)?.[1] ?? 0), height: Number((j.sensor as number[] | undefined)?.[0] ?? 0) }, frame: { width: 0, height: 0 },
        binning: String(j.binning ?? 'rgb2x2'), scale: 2, gx0: 0, gy0: 0, origin: { x0: 0, y0: 0 }, encoding: String(j.encoding ?? 'u12_shifted'), camera: String(j.camera ?? ''), preset: String(j.camera ?? ''),
        basis: (j.basis as 'photon' | 'energy') ?? 'photon', irCut: null, psi: { name: f.name, sha256: '', lamNm: lam, R: psi.R, G: psi.G, B: psi.B }, vignette: null, hot: null, darkLine: null, ovenTemps: [], texpRangeUs: [NaN, NaN], ratioWorst: null,
      };
    }
    cal.mode = cal.mode === 'oven' || cal.mode === 'sphere' ? cal.mode : 'file';
    if (cal.basis && cal.basis !== S.basis) S.calFileMsgs.push({ code: 'cal_basis', vars: { a: cal.basis === 'photon' ? L('por fotón', 'per photon') : L('por energía', 'per energy'), b: S.basis === 'photon' ? L('por fotón', 'per photon') : L('por energía', 'per energy') }, level: 'info' });
    const fileCal: Calibration = { ...cal, mode: 'file', calibrated: cal.calibrated ?? ['R', 'G', 'B'] };
    S.cal = fileCal;
    gtabCheck(j);
    invalidate(2, 'cal');
  } catch { S.calFileMsgs = [{ code: 'cal_bad' }]; }
  renderAll();
  announce(1);
}
/** Recompute G_j(T) from the ψ_j of the file and compare with its G_tab at 1e-6 (WEB_SPEC 1c). */
function gtabCheck(j: Record<string, unknown>) {
  const gt = j.G_tab as Record<string, number[]> | undefined, Tg = (j.T_grid ?? j.T_tab) as number[] | undefined;
  if (!gt || !Tg || !S.cal) return;
  const c = S.cal.psi;
  const lam = Float64Array.from(c.lamNm);
  // the file's G_tab comes from kernel_table (trapezoid), recomputed the same way
  const tab = kernelTrapz(lam, c, Tg);
  let worst = 0;
  for (const ch of CH) for (let i = 0; i < Tg.length; i++) if (gt[ch]?.[i]) worst = Math.max(worst, Math.abs(tab[ch][i] / gt[ch][i] - 1));
  if (worst > 1e-6) S.calFileMsgs.push({ code: 'gtab_mismatch', vars: { a: worst.toExponential(1) } });
}
function kernelTrapz(lam: Float64Array, c: Calibration['psi'], Tg: number[]): Rgb<Float64Array> {
  const C1 = 2 * 6.62607015e-34 * 299792458 ** 2, C2 = (6.62607015e-34 * 299792458) / 1.380649e-23;
  const law = emLaw();
  const E = (l: number) => { if (law.kind === 'constant') return law.E; const u = l * 1e-3, Lg = Math.log(u); const a = law.kind === 'group' ? 0.284 : law.a, b = law.kind === 'group' ? 0.272 : law.b, cc = law.kind === 'group' ? 0.057 : law.c; return a + b * Lg * Lg + (cc * Lg) / u; };
  const out = { R: new Float64Array(Tg.length), G: new Float64Array(Tg.length), B: new Float64Array(Tg.length) };
  for (let t = 0; t < Tg.length; t++) for (const ch of CH) {
    let s = 0;
    for (let k = 0; k + 1 < lam.length; k++) {
      const f = (i: number) => { const l = lam[i] * 1e-9; return c[ch][i] * ((6 * Math.PI * E(lam[i])) / l) * (C1 / l ** 5 / (Math.exp(C2 / (l * Tg[t])) - 1)) * 1e-9; };
      s += 0.5 * (f(k) + f(k + 1)) * (lam[k + 1] - lam[k]);
    }
    out[ch][t] = s;
  }
  return out;
}
function downloadCal() {
  if (!S.cal) return;
  save(new Blob([JSON.stringify(S.cal, null, 1)], { type: 'application/json' }), `${FN.cal()}_${S.cal.mode}_${stamp()}.json`);
}

// ============================================================================ calibration state (stage 2)
function calDecision(): CalDecision {
  const c = S.flame.sensorMismatch ? null : S.cal;
  return decideCalibration({
    source: c ? (c.mode === 'oven' ? 'oven' : c.mode === 'sphere' ? 'sphere' : 'file') : null,
    calibrated: c?.calibrated ?? [], consistent: c?.consistent ?? false, tested: c?.mode === 'oven' ? c.tested !== false : true, emissivityKnown: c?.mode === 'oven' ? S.known : false,
    ndDeclared: c?.ndDeclared, fileAbsolute: c?.absolute, texpConfirmed: S.flame.texpConfirmed, scaleConfirmed: S.flame.scaleConfirmed,
  });
}

// ============================================================================ flame (stage 3)
function flameItems() { return S.items.filter((it) => it.role === 'flame' && it.status === 'ok'); }
const isAverage = (it: Item) => /promedio|average/i.test(it.path.split('/').pop() ?? '');
function bgItems() {
  const all = S.items.filter((it) => it.role === 'background' && it.status === 'ok');
  // a background average (fondo_promedio_N…) already contains the single backgrounds
  return all.some(isAverage) ? all.filter(isAverage) : all;
}
async function loadFlame() {
  const fl = flameItems();
  if (!fl.length) return;
  if (!fin(S.flame.texpUs)) { const t = fl.find((it) => it.texpUs !== undefined)?.texpUs; if (t !== undefined) S.flame.texpUs = t; }
  S.busy = L('Midiendo la asimetría de cada cuadro', 'Measuring the asymmetry of each frame');
  renderBusy();
  try {
    const q = analysisQuarter();
    const r = await ask<{ id: string; axis: number | null; asym: number; nSat: number; thumb: Item['thumb'] }[]>({ type: 'flame.load', ids: fl.map((x) => x.id), quarter: q, mirror: S.flame.mirror }, (d, t) => { S.busy = L(`Asimetría, ${d} de ${t} cuadros`, `Asymmetry, ${d} of ${t} frames`); renderBusy(); });
    const msgs: Msg[] = [];
    if (fl.some((it) => it.name.timeS !== undefined)) { msgs.push({ code: 'times_in_names', level: 'info' }); if (!timesSet) S.flame.times = 'instants'; }
    if (S.flame.times === 'instants' && !S.flame.instant) S.flame.instant = fl[0].id;
    S.flame.msgs = [];
    detectTransposed();
    // a flame lying on its side gives meaningless axes, so nothing is discarded until it is turned (GB3r1-estudiante-16)
    const lying = S.flame.msgs.some((m) => m.code === 'transposed') && S.flame.quarter === 0;
    const instants = S.flame.times === 'instants' && fl.some((it) => it.name.timeS !== undefined);
    const avg = fl.filter(isAverage);
    const pool = r.filter((x) => !avg.length || avg.some((a) => a.id === x.id));
    const axes = pool.map((x) => x.axis).filter((x): x is number => x !== null);
    const med = axes.length ? axes.slice().sort((a, b) => a - b)[Math.floor(axes.length / 2)] : NaN;
    let tilted = 0, warned = 0, far = 0;
    for (const x of r) {
      const it = S.items.find((y) => y.id === x.id)!;
      it.asym = x.asym; it.axis = x.axis; if (x.thumb) it.thumb = x.thumb;
      it.include = true; it.why = '';
      // an averaged file (promedio_N…) already contains the single frames (GB3r1-estudiante-07)
      if (avg.length && !isAverage(it)) { it.include = false; it.why = 'average'; continue; }
      if (lying || instants) continue;
      if (x.asym > 0.4) { it.include = false; it.why = 'tilted'; tilted++; }
      else if (x.axis !== null && fin(med) && Math.abs(x.axis - med) > 8) { it.include = false; it.why = 'axis'; far++; }
      else if (x.asym > 0.2) warned++;
    }
    if (lying) msgs.push({ code: 'orient_first', level: 'info' });
    else if (instants) {
      const one = r.find((x) => x.id === S.flame.instant);
      if (one && one.asym > 0.2) msgs.push({ code: 'tilt_warn_one', vars: { a: num(one.asym, 2) } });
    } else {
      if (tilted) msgs.push({ code: tilted === 1 ? 'tilted_1' : 'tilted', vars: { n: tilted, m: pool.length } });
      if (warned) msgs.push({ code: warned === 1 ? 'tilt_warn_1' : 'tilt_warn', vars: { n: warned } });
      if (axes.length > 1) {
        const spread = Math.max(...axes) - Math.min(...axes);
        if (far || spread > 4) msgs.push({ code: 'axis_spread', vars: { a: spread, n: far, b: 8 } });
      }
    }
    if (avg.length) msgs.push({ code: 'average_only', vars: { file: avg.map((a) => a.path.split('/').pop()).join(', '), n: fl.length - avg.length }, level: 'info' });
    // frames of another size leave the list by name, the largest group stays (GB3r2-adv-08)
    const sizeOf = (it: Item) => `${it.fi!.width} × ${it.fi!.height}`;
    const bySize = new Map<string, Item[]>();
    for (const it of fl.filter((x) => x.fi)) bySize.set(sizeOf(it), [...(bySize.get(sizeOf(it)) ?? []), it]);
    if (bySize.size > 1) {
      const main = [...bySize.entries()].sort((a, b) => b[1].length - a[1].length)[0][0];
      const odd = fl.filter((it) => it.fi && sizeOf(it) !== main);
      for (const it of odd) { it.include = false; it.why = L(`tamaño ${sizeOf(it)}, distinto de la serie`, `size ${sizeOf(it)}, unlike the series`); }
      msgs.push({ code: 'frames_mixed_size', vars: { a: main, n: odd.length }, files: odd.map((it) => `${it.path} (${sizeOf(it)})`) });
    }
    if (!fl.some((it) => it.texpUs !== undefined)) msgs.push({ code: 'name_unread_flame', vars: { file: fl[0].path } });
    S.flame.msgs = [...msgs, ...S.flame.msgs];
    S.flame.loaded = true;
  } catch (e) { S.flame.msgs = [{ code: (e as { code: string }).code }]; }
  S.busy = '';
  invalidate(3, 'flame');
  renderAll();
}
function analysisQuarter() { return (((S.flame.quarter - S.flame.sensorQ) % 4) + 4) % 4; }
function detectTransposed() {
  const it = flameItems()[0];
  if (!it?.fi) return;
  const sen = S.cal?.sensor ?? sensorOf(it);
  const w = it.fi.width, h = it.fi.height;
  const fits = (W: number, H: number) => w <= W && h <= H;
  const rewritten = !!it.fi.software && !/pylon|basler/i.test(it.fi.software);
  const exact = w === sen.height && h === sen.width && w !== h;
  if ((exact || (rewritten && !fits(sen.width, sen.height) && fits(sen.height, sen.width)) || (rewritten && S.cal && w === S.cal.frame.height && h === S.cal.frame.width && w !== h)) && !S.flame.msgs.some((m) => m.code === 'transposed')) S.flame.msgs.push({ code: 'transposed' });
}
async function loadRuler() {
  const id = S.flame.rulerId;
  if (!id) return;
  try {
    const r = await ask<{ preview: Planes; scale: number; w: number; h: number; shift: null | (ShiftTest & { pattern: BayerPattern }) }>({ type: 'ruler', id });
    S.flame.rulerPreview = r.preview; S.flame.rulerScale = r.scale; S.flame.rulerPts = []; S.flame.shift = r.shift;
  } catch { /* the table shows the file as unreadable */ }
  renderStage3();
}
/** Crop origin of the flame in sensor coordinates after turning the file back by sensorQ quarter turns. */
function flameCropSensor(it: Item): { x0: number; y0: number } {
  const c = it.crop ?? { x0: 0, y0: 0 };
  const q = ((S.flame.sensorQ % 4) + 4) % 4;
  if (!q || !it.fi) return c;
  const sen = S.cal?.sensor ?? sensorOf(it);
  const Wf = q % 2 ? sen.height : sen.width, Hf = q % 2 ? sen.width : sen.height;
  const w = it.fi.width, h = it.fi.height;
  const corners = [[c.x0, c.y0], [c.x0 + w - 1, c.y0 + h - 1]];
  const rot = ([x, y]: number[]) => (q === 1 ? [Hf - 1 - y, x] : q === 2 ? [Wf - 1 - x, Hf - 1 - y] : [y, Wf - 1 - x]);
  const r = corners.map(rot);
  return { x0: Math.min(r[0][0], r[1][0]), y0: Math.min(r[0][1], r[1][1]) };
}
function calUseFor(it: Item): CalUse | null {
  S.flame.sensorMismatch = false;
  const c = S.cal;
  if (!c) return null;
  const fi = it.fi!;
  const camF = cameraOf([it])?.cam, camC = c.camera;
  const senF = sensorOf(it);
  const mism = (camF && camC && !camF.includes(camC) && !camC.includes(camF)) || (c.sensor.width && (senF.width !== c.sensor.width || senF.height !== c.sensor.height) && (senF.width !== c.sensor.height || senF.height !== c.sensor.width));
  if (mism) { S.flame.sensorMismatch = true; return null; }
  const vig = c.vignette ? { block: c.vignette.block, bw: c.vignette.bw, bh: c.vignette.bh, R: unb64F32(c.vignette.R), G: unb64F32(c.vignette.G), B: unb64F32(c.vignette.B) } : null;
  void fi;
  return { K: [c.K.R, c.K.G, c.K.B], vignette: vig, ox: c.origin.x0, oy: c.origin.y0, scale: c.scale, gx0: c.gx0, gy0: c.gy0, hot: c.hot ? unb64I32(c.hot) : null, calW: Math.floor((c.frame.width - c.gx0) / c.scale), calH: Math.floor((c.frame.height - c.gy0) / c.scale) };
}
function flameIds(): string[] {
  const fl = flameItems().filter((it) => it.include);
  // GB3 fix, the instant chosen in the selector is the one used, even if the frame list left it out
  if (S.flame.times === 'instants' && flameItems().some((it) => it.name.timeS !== undefined)) { const one = flameItems().find((it) => it.id === S.flame.instant) ?? fl[0]; return one ? [one.id] : []; }
  return fl.map((it) => it.id);
}
async function prepareFlameNow() {
  S.flame.prepErr = '';
  const ids = flameIds();
  const msgs: Msg[] = [];
  if (!ids.length) { S.flame.msgs = [{ code: 'need_stage', vars: { a: '2' } }]; renderStage3(); return; }
  const first = S.items.find((it) => it.id === ids[0])!;
  const bgs = bgItems().filter((b) => !(b.fi && first.fi && (b.fi.width !== first.fi.width || b.fi.height !== first.fi.height)));
  for (const b of bgs) if (b.texpUs !== undefined && fin(S.flame.texpUs) && Math.abs(b.texpUs - S.flame.texpUs) > 0.5) msgs.push({ code: b.role === 'background' ? 'bg_texp' : 'dark_texp', vars: { a: b.texpUs, b: S.flame.texpUs, file: b.path.split('/').pop() ?? b.path }, level: 'error' });
  if (msgs.some((m) => m.level === 'error')) { S.flame.msgs = msgs; renderStage3(); return; }
  const cal = calUseFor(first);
  if (S.flame.sensorMismatch && S.cal) msgs.push({ code: 'sensor_mismatch', vars: { a: `${S.cal.sensor.width} × ${S.cal.sensor.height}`, b: `${sensorOf(first).width} × ${sensorOf(first).height}` } });
  if (S.cal && !S.flame.sensorMismatch && first.fi && S.cal.binning !== first.fi.binning) msgs.push({ code: 'method_mismatch', vars: { a: binName(S.cal.binning), b: binName(first.fi.binning) } });
  if (S.cal && !S.flame.sensorMismatch && first.fi && S.cal.encoding !== first.fi.encoding && S.cal.mode !== 'file') msgs.push({ code: 'encoding_mismatch', vars: { a: encName(S.cal.encoding), b: encName(first.fi.encoding) } });
  if (S.cal && fin(S.cal.texpRangeUs[0]) && fin(S.flame.texpUs) && (S.flame.texpUs < S.cal.texpRangeUs[0] || S.flame.texpUs > S.cal.texpRangeUs[1]) && S.cal.mode === 'oven') msgs.push({ code: 'texp_outside_cal', vars: { a: S.flame.texpUs } });
  if (!fin(S.flame.texpUs)) { msgs.push({ code: 'name_unread_flame', vars: { file: first.path }, level: 'error' }); S.flame.msgs = msgs; renderStage3(); return; }
  // flat_missing is said in stage 1, where the flat field is loaded (GB3r3-adv-07)
  const dlf = S.cal?.darkLineFit;
  const darkLine = !bgs.length && dlf && !S.flame.sensorMismatch ? ([dlf.R.a + dlf.R.b * S.flame.texpUs * 1e-6, dlf.G.a + dlf.G.b * S.flame.texpUs * 1e-6, dlf.B.a + dlf.B.b * S.flame.texpUs * 1e-6] as [number, number, number]) : null;
  const cs = flameCropSensor(first);
  const g: GeometryOpts = {
    sensorQuarter: S.flame.sensorQ, sensorMirror: false, quarter: analysisQuarter(), mirror: S.flame.mirror, fineDeg: S.flame.fineDeg,
    axisManual: S.flame.axisAuto ? null : S.flame.axisManual, halfWidth: fin(S.flame.halfWidth) ? S.flame.halfWidth : null, nr: fin(S.flame.nr) ? S.flame.nr : null,
    rowTop: fin(S.flame.rowTop) ? S.flame.rowTop : null, rowBottom: fin(S.flame.rowBottom) ? S.flame.rowBottom : null, rowsPerHeight: S.flame.rowsPerHeight,
    mmPerPxSensor: fin(S.flame.mmPerPx) ? S.flame.mmPerPx : 1, burnerRow: fin(S.flame.burnerRow) ? S.flame.burnerRow : null, zBurnerMm: S.flame.zBurner, cropX0: cs.x0, cropY0: cs.y0,
  };
  if (S.flame.fineDeg) msgs.push({ code: 'fine_rotation', level: 'info' });
  S.busy = L('Preparando las proyecciones', 'Preparing the projections');
  renderBusy();
  try {
    const r = await ask<{ prepared: Prepared; bgHow: string; nHot: number }>({ type: 'flame.prepare', ids, backgrounds: bgs.map((b) => b.id), texpUs: S.flame.texpUs, cal, darkLine, geom: g }, (d, t) => { S.busy = L(`Sumando cuadros, ${d} de ${t}`, `Adding frames, ${d} of ${t}`); renderBusy(); });
    const p = r.prepared;
    S.flame.prepared = { ...p, bgHow: r.bgHow, nHot: r.nHot };
    S.flame.calUsed = cal;
    S.flame.height = Math.floor(p.nz / 2);
    if (r.bgHow === 'border') msgs.push({ code: 'bg_estimated' });
    if (r.bgHow === 'darkline') msgs.push({ code: 'bg_darkline' });
    if (p.satRows.length) msgs.push({ code: 'rows_saturated', vars: { n: p.satRows.length } });
    for (const c of p.axisCodes) msgs.push({ code: c === 'axis_search_too_small' || c === 'axis_empty' || c === 'axis_no_rows' ? 'axis_none' : c, level: c === 'axis_search_reduced' ? 'info' : 'warn' });
    if (p.edgeRatio > 0.01) msgs.push({ code: fin(S.flame.nr) ? 'edge_signal' : 'edge_signal_file', vars: { a: Math.round(p.edgeRatio * 1000) / 10 } });
    if (!fin(S.flame.mmPerPx)) msgs.push({ code: 'scale_none' });
    else if (!S.flame.scaleConfirmed) msgs.push({ code: 'scale_missing', level: 'info' });
    if (!S.flame.texpConfirmed) msgs.push({ code: 'texp_unconfirmed', level: 'info' });
    if (looksUpsideDown(p)) msgs.push({ code: 'upside_down' });
    detectTransposed();
    S.flame.msgs = [...msgs, ...S.flame.msgs.filter((m) => ['transposed', 'tilted', 'tilted_1', 'tilt_warn', 'tilt_warn_1', 'tilt_warn_one', 'axis_spread', 'times_in_names', 'average_only', 'frames_mixed_size', 'orient_first'].includes(m.code) && !msgs.some((x) => x.code === m.code))];
    delete S.stale[3];
    invalidate(4, 'flame');
    S.abel.out = null; S.tfs.out = null;
  } catch (e) { S.flame.msgs = [...msgs, { code: (e as { code: string }).code, level: 'error' }]; S.flame.prepErr = (e as { code: string }).code; }
  S.busy = '';
  renderAll();
  announce(3);
}
/**
 * A sooting laminar flame closes towards its tip. Width of each height where Σ_j P_j passes 10 % of the
 * row maximum, over the heights with at least 10 % of the brightest one. If the upper fifth is clearly
 * wider than the lower fifth the flame is probably upside down (GB3r3-adv-02). It is a hint, the box decides.
 */
function looksUpsideDown(p: Prepared): boolean {
  const rows: { w: number }[] = [];
  let gmax = 0;
  const sumAt = (k: number, i: number) => { let v = 0; for (let c = 0; c < 3; c++) { const a = p.right[c][k][i], b = p.left[c][k][i]; v += (fin(a) ? a : 0) + (fin(b) ? b : 0); } return v; };
  const mx: number[] = [];
  for (let k = 0; k < p.nz; k++) { let m = 0; for (let i = 0; i < p.nr; i++) m = Math.max(m, sumAt(k, i)); mx.push(m); gmax = Math.max(gmax, m); }
  for (let k = 0; k < p.nz; k++) {
    if (!(mx[k] > 0.1 * gmax)) continue;
    let w = 0;
    for (let i = 0; i < p.nr; i++) if (sumAt(k, i) > 0.1 * mx[k]) w = i + 1;
    rows.push({ w });
  }
  if (rows.length < 10) return false;
  const n5 = Math.max(2, Math.floor(rows.length / 5));
  const med = (a: number[]) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
  const low = med(rows.slice(0, n5).map((r) => r.w)), high = med(rows.slice(-n5).map((r) => r.w));
  return high > 1.3 * low;
}
function downloadNpz() {
  const p = S.flame.prepared;
  if (!p) return;
  const nz = p.nz, nr = p.nr;
  const b = new Float32Array(nz * 2 * nr * 3);
  for (let k = 0; k < nz; k++) for (let c = 0; c < 3; c++) for (let i = 0; i < nr; i++) {
    b[(k * 2 * nr + nr + i) * 3 + c] = p.right[c][k][i];
    b[(k * 2 * nr + nr - 1 - i) * 3 + c] = p.left[c][k][i];
  }
  const x = new Float64Array(2 * nr);
  for (let i = 0; i < nr; i++) { x[nr + i] = p.rMm[i]; x[nr - 1 - i] = -p.rMm[i]; }
  const d = calDecision();
  const abs = !!S.flame.calUsed && d.fs !== 'no';
  const meta = { units: abs ? 'W m^-2 sr^-1' : 'relative units', calibration_status: abs ? (d.provisional ? 'absolute_provisional' : 'absolute') : 'relative', provenance: { tool: TOOL_NAME, version: TOOL_VERSION, commit: S.o.commit, created: new Date().toISOString(), inputs: flameIds().map((id) => S.items.find((it) => it.id === id)!).map((it) => ({ name: it.path, sha256: it.sha })) }, frame_diagnostics: [], format_version: 1 };
  const blob = npz({ b: npy(b, [1, nz, 2 * nr, 3]), r: npy(Float64Array.from(p.rMm), [nr]), x: npy(x, [2 * nr]), z: npy(Float64Array.from(p.zMm), [nz]), times: npy(new Float64Array([0]), [1]), meta_json: npyString(JSON.stringify(meta)), format_version: npy(new Int32Array([1]), []) });
  save(blob, `${L('proyecciones', 'projections')}_${stamp()}.npz`);
}

// ============================================================================ Abel (stage 4)
function abelRun(): AbelRun { return { method: S.abel.method, sym: S.abel.sym, lambdaRel: S.abel.lambdaRel, segments: S.abel.segments, satLambda: S.abel.satLambda }; }
/** Stages 4 to 6 need the burner at the bottom of the preview, in the recommended path and step by step (GB3r3-adv-01). */
function burnerGate(): boolean {
  if (S.flame.burnerOk) { S.flame.msgs = S.flame.msgs.filter((m) => m.code !== 'burner_needed'); return true; }
  S.flame.msgs = [{ code: 'burner_needed', level: 'error' }, ...S.flame.msgs.filter((m) => m.code !== 'burner_needed')];
  renderAll(); announce(3); document.getElementById('wr-burnerok')?.focus();
  return false;
}
async function runAbelNow() {
  if (!S.flame.prepared) return;
  if (!burnerGate()) return;
  if (S.abel.method === 'annabel') { await runAnnabel(); return; }
  S.busy = L('Invirtiendo todas las alturas', 'Inverting every height');
  renderBusy();
  try {
    S.abel.out = await ask<AbelOut>({ type: 'abel.run', run: abelRun() });
    S.abel.outMethod = S.abel.method;
    delete S.stale[4];
    invalidate(5, 'abel');
  } catch (e) { S.abel.annabel.msgs = [{ code: (e as { code: string }).code }]; }
  S.busy = '';
  renderAll();
  announce(4);
}
async function runLcurve() {
  const p = S.flame.prepared;
  if (!p) return;
  const rows = Array.from({ length: p.nz }, (_, k) => k).filter((k) => k >= Math.floor(p.nz / 3) && k < Math.ceil((2 * p.nz) / 3));
  S.busy = L('Calculando la curva L', 'Computing the L-curve');
  renderBusy();
  try {
    S.abel.lcurve = await ask<{ lambdas: number[]; rho: number[]; eta: number[]; lambda: number; index: number }>({ type: 'abel.lcurve', rows, ref: 'G', sym: S.abel.sym });
    S.abel.lcFor = p; S.abel.lcSym = S.abel.sym;
    // the corner is applied at once, so Compute does not run with the old λ (GB3r2-estudiante-16), and the
    // later steps go out of date only when the value changes (DESIGN §4, E4-6)
    // with smoothing set by hand or fixed, the curve is only drawn, «Back to automatic» applies it (redesign g)
    const lc = S.abel.lcurve as (typeof S.abel.lcurve & { kappa?: number[] }) | null;
    if (lc && S.abel.lambdaAuto && !hasCorner(lc)) {
      const was = S.abel.lambdaRel;
      S.abel.lambdaRel = 0.1; S.abel.lambdaAuto = false; S.abel.lamFlat = true;
      if (was !== 0.1) invalidate(4, 'abel');
    }
    else if (lc && fin(lc.lambda) && S.abel.lambdaAuto) {
      const changed = Math.abs(lc.lambda / S.abel.lambdaRel - 1) > 1e-9;
      S.abel.lambdaRel = lc.lambda; S.abel.lamFlat = false;
      if (changed) invalidate(4, 'abel');
    }
  } catch { /* the plot stays empty */ }
  S.busy = '';
  renderBusy();
  renderAll();
}
/** False when the curvature of the L-curve is nowhere positive inside the range, so the curve has no corner to pick. */
function hasCorner(lc: { kappa?: number[] }): boolean {
  const k = lc.kappa;
  if (!k || k.length < 3) return true;
  return k.slice(1, -1).some((x) => fin(x) && x > 0);
}
/** The main button of step 3. With automatic smoothing and Tikhonov the L-curve corner comes first (DESIGN §4, §9.1). */
async function abelClick() {
  if (!S.flame.prepared) return;
  if (!burnerGate()) return;
  if (S.abel.method === 'tikhonov' && S.abel.lambdaAuto) {
    if (S.abel.lcFor !== S.flame.prepared || S.abel.lcSym !== S.abel.sym || !S.abel.lcurve) await runLcurve();
    if (S.abel.lambdaAuto && S.abel.lcurve && fin(S.abel.lcurve.lambda)) S.abel.lambdaRel = S.abel.lcurve.lambda;
  }
  await runAbelNow();
}
async function compareMethods() {
  const p = S.flame.prepared;
  if (!p) return;
  const out: Record<string, Float64Array> = {};
  for (const m of ['onion', 'tikhonov', 'sat'] as AbelMethod[]) {
    const r = await ask<AbelOut>({ type: 'abel.run', run: { ...abelRun(), method: m } });
    out[m] = r.J[1][S.flame.height];
  }
  if (S.abel.out) await ask<AbelOut>({ type: 'abel.run', run: { ...abelRun(), method: S.abel.outMethod === 'annabel' ? 'tikhonov' : (S.abel.outMethod as AbelMethod) } }).then((r) => { if (S.abel.outMethod !== 'annabel') S.abel.out = r; });
  S.abel.compare = out;
  renderStage4();
}
/** Projections of the stage for ANNAbel, both halves as the contract of annabel_app/core/io.py, saturated heights left out. */
function annabelProjections(): { proj: Projections; keep: number[] } | null {
  const p = S.flame.prepared;
  if (!p) return null;
  const keep = Array.from({ length: p.nz }, (_, k) => k).filter((k) => !p.satRows.includes(k));
  const nz = keep.length, nr = p.nr;
  const b = new Float32Array(nz * 2 * nr * 3);
  keep.forEach((k, kk) => { for (let c = 0; c < 3; c++) for (let i = 0; i < nr; i++) { b[(kk * 2 * nr + nr + i) * 3 + c] = p.right[c][k][i]; b[(kk * 2 * nr + nr - 1 - i) * 3 + c] = p.left[c][k][i]; } });
  return { proj: { nt: 1, nz, nr, b, rMm: p.rMm, zMm: keep.map((k) => p.zMm[k]), times: [0] }, keep };
}
const poolFactory = () => helpers[helperIdx++ % helpers.length];
async function estimateAnnabel() {
  const a = annabelProjections();
  if (!a) return;
  const A = S.abel.annabel;
  A.msgs = [];
  S.busy = L('Midiendo 20 iteraciones de la red neuronal', 'Timing 20 neural network iterations');
  renderBusy();
  try {
    const prob = buildProblem(a.proj);
    const p0 = initXavier(S.res.seed);
    if (S.gpu) {
      const g = await createGpuStage1(prob, p0, 23);
      try { g.encodeIters(3); await g.done(); const t0 = performance.now(); g.encodeIters(20); await g.done(); A.estimate = { path: 'webgpu', sPerIter: (performance.now() - t0) / 1000 / 20 }; } finally { g.destroy(); }
    } else {
      const eng = await WorkerPoolEngine.create(prob, layout().nParams, { n: helpers.length, factory: poolFactory });
      const p = p0.slice();
      await eng.evaluate(p, true);
      const t0 = performance.now();
      for (let i = 0; i < 20; i++) await eng.evaluate(p, true);
      A.estimate = { path: 'workers', sPerIter: (performance.now() - t0) / 1000 / 20 };
    }
    const min = (A.estimate.sPerIter * A.iterations) / 60;
    if (min > 15 || A.estimate.path !== 'webgpu') A.msgs.push({ code: 'annabel_slow', vars: { a: Math.max(1, Math.ceil(min)) } });
  } catch (e) { A.msgs.push({ code: /webgpu|device/i.test(String((e as Error).message)) ? 'webgpu_lost' : 'worker_failed' }); }
  S.busy = '';
  renderBusy();
  renderStage4();
}
async function runAnnabel() {
  const a = annabelProjections();
  if (!a) return;
  const A = S.abel.annabel;
  const p = S.flame.prepared!;
  A.running = true; A.msgs = []; A.ctrl = new AbortController(); A.tick = null;
  renderStage4();
  try {
    const prob = buildProblem(a.proj);
    const p0 = initXavier(S.res.seed);
    const topts = { iterations: A.iterations, signal: A.ctrl.signal, onTick: (t: TrainTick) => { A.tick = t; renderAnnabelProgress(); } };
    let train: TrainResult;
    if (S.gpu) {
      const g = await createGpuStage1(prob, p0, A.iterations);
      try { train = await g.train({ ...topts, itersPerSubmit: 200 }); } finally { g.destroy(); }
    } else {
      const eng = await WorkerPoolEngine.create(prob, layout().nParams, { n: helpers.length, factory: poolFactory });
      train = await trainWithEngine(eng, p0, topts);
    }
    A.result = train;
    if (train.endReason === 'stopped_by_user') A.msgs.push({ code: 'annabel_stopped', vars: { n: train.bestIter }, level: 'info' });
    const absolute = !!S.flame.calUsed;
    const res = finishStage1(prob, a.proj.b, { rMm: a.proj.rMm, zMm: a.proj.zMm, times: a.proj.times }, train, { absolute });
    const src = res.jSI ?? Float32Array.from(res.jPerMm, (v) => v * 1e3);
    const nr = p.nr;
    const J: Float64Array[][] = [[], [], []], R: Float64Array[][] = [[], [], []];
    for (let c = 0; c < 3; c++) for (let k = 0; k < p.nz; k++) { J[c].push(new Float64Array(nr).fill(NaN)); R[c].push(new Float64Array(nr).fill(NaN)); }
    a.keep.forEach((k, kk) => { for (let i = 0; i < nr; i++) for (let c = 0; c < 3; c++) { J[c][k][i] = src[(kk * nr + i) * 3 + c]; R[c][k][i] = res.P[(kk * nr + i) * 3 + c]; } });
    S.abel.out = await ask<AbelOut>({ type: 'abel.set', J, reproj: R, run: abelRun() });
    S.abel.outMethod = 'annabel';
    delete S.stale[4];
    invalidate(5, 'abel');
  } catch (e) { A.msgs.push({ code: /webgpu|device|lost/i.test(String((e as Error).message)) ? 'webgpu_lost' : 'worker_failed' }); }
  A.running = false; A.ctrl = null;
  renderAll();
}

// ============================================================================ T and f_s (stage 5) and results (stage 6)
/** Channels for f_s, R and G by default since G4 r2 (director). An old run.json without the field, or with a value not listed, gets R and G. */
function asFsMode(v: unknown): FsMode { return (['RG', 'R', 'G', 'RGB', 'B'] as const).includes(v as FsMode) ? (v as FsMode) : 'RG'; }
function tfsWeights(): [number, number, number] {
  const ch = S.cal?.calibrated ?? ['R', 'G', 'B'];
  return [ch.includes('R') ? 1 : 0, ch.includes('G') ? 1 : 0, ch.includes('B') || !S.cal ? 1 : 0];
}
async function runTfsNow() {
  if (!S.abel.out) return;
  if (!burnerGate()) return;
  S.busy = L('Calculando la temperatura y la fracción de hollín', 'Computing the temperature and the soot volume fraction');
  renderBusy();
  try {
    S.tfs.dec = calDecision();
    S.tfs.net.result = null;
    S.tfs.out = await ask<TfsOut>(tfsRunMsg());
    if (S.tfs.path === 'net') { S.busy = ''; renderBusy(); await runNet(); }
    delete S.stale[5]; delete S.stale[6];
    S.res.band = null;
    const p = S.flame.prepared!;
    if (!S.res.heights.length) {
      // the three default heights fall inside the rows with cells in use, so every profile has data
      const t = S.tfs.out!, rows: number[] = [];
      for (let k = 0; k < p.nz; k++) for (let i = 0; i < p.nr; i++) if (t.mask[k * p.nr + i] & 1) { rows.push(k); break; }
      const k0 = rows.length ? rows[0] : 0, k1 = rows.length ? rows[rows.length - 1] : p.nz - 1;
      S.res.heights = [0.25, 0.5, 0.75].map((f) => Math.round(k0 + f * (k1 - k0)));
    }
    defaultScales();
  } catch (e) { S.flame.msgs.push({ code: (e as { code: string }).code }); }
  S.busy = '';
  renderAll();
  announce(5);
}
/** The worker computes the table paths. For the network it gives the closed form (fit3), its masks and the table. */
const workerPath = (): 'ratio' | 'fit3' => (S.tfs.path === 'net' ? 'fit3' : S.tfs.path);
const tfsRunMsg = () => ({ type: 'tfs.run' as const, opts: { path: workerPath(), pair: S.tfs.pair, emLaw: emLaw(), sigFrac: S.tfs.sigFrac, islandFrac: S.tfs.islandFrac, chemBG: calDecision().state !== 'uncalibrated', zCutMm: fin(S.tfs.zCut) ? S.tfs.zCut : null, chemThr: S.tfs.chemThr, fsChannel: S.tfs.fsChannel, refine: true, weights: tfsWeights() }, psi: psiMsg() });
/** Stage-1 emission of stage 4 for the network: every height with finite J in the three channels, one frame,
 *  W m⁻³ sr⁻¹ (j_to_si = 1), kernels G_j(T) of the table on 1000 to 2600 K, chan_w of the calibration. */
function netInput(t: TfsOut): { J: Float32Array; grid: { nt: number; nz: number; nr: number; rMm: Float64Array; zMm: number[]; times: number[] }; kern: KernelTable; keep: number[] } | null {
  const p = S.flame.prepared, o = S.abel.out;
  if (!p || !o || !t.table.G) return null;
  const nr = p.nr;
  const keep = Array.from({ length: p.nz }, (_, k) => k).filter((k) => [0, 1, 2].every((c) => Array.from(o.J[c][k]).every(fin)));
  const J = new Float32Array(keep.length * nr * 3);
  keep.forEach((k, kk) => { for (let i = 0; i < nr; i++) for (let c = 0; c < 3; c++) J[(kk * nr + i) * 3 + c] = o.J[c][k][i]; });
  const G = t.table.G, nT = t.table.T.length;
  const K = new Float32Array(3 * nT);
  K.set(G.R, 0); K.set(G.G, nT); K.set(G.B, 2 * nT);
  // f_s,ref of the app in absolute mode (20 ppm); without a calibration f_s is relative and the bound follows the closed form
  let fsRef = 2e-5;
  if (calDecision().fs !== 'ppm') { let m = 0; for (let q = 0; q < t.fs.length; q++) if (t.mask[q] & 1 && fin(t.fs[q])) m = Math.max(m, t.fs[q]); fsRef = m > 0 ? 4 * m : 1; }
  return { J, grid: { nt: 1, nz: keep.length, nr, rMm: p.rMm, zMm: keep.map((k) => p.zMm[k]), times: [0] }, kern: { Tgrid: t.table.T, K, chanW: tfsWeights(), jToSi: 1, fsRef }, keep };
}
function netEngine(prob: Stage2Problem, nParams: number) {
  return helpers.length ? Stage2PoolEngine.create(prob, nParams, { n: helpers.length, factory: poolFactory }) : Promise.resolve(new Stage2LocalEngine(prob));
}
function netError(e: unknown): Msg {
  const m = String((e as Error)?.message ?? e);
  if (m === 'stage2_two_channels') return { code: 'net_two_channels', level: 'error' };
  if (m === 'stage2_no_signal') return { code: 'net_no_signal', level: 'error' };
  return { code: /webgpu|device|lost/i.test(m) ? 'webgpu_lost' : 'worker_failed', level: 'error' };
}
async function estimateNet() {
  const N = S.tfs.net;
  if (!S.abel.out || N.running) return;
  N.msgs = N.msgs.filter((m) => m.code !== 'net_slow');
  S.busy = L('Midiendo 20 iteraciones de la red', 'Timing 20 network iterations');
  renderBusy();
  try {
    // the table and its masks, without touching the result of stage 5
    const t = S.tfs.out && !S.stale[5] ? S.tfs.out : await ask<TfsOut>(tfsRunMsg());
    const inp = netInput(t);
    if (!inp) throw new Error('no_emission');
    const prob = buildStage2Problem(inp.J, inp.grid, inp.kern, { iterations: N.iterations });
    const p0 = initStage2(S.res.seed, prob.sizes);
    if (S.gpu) {
      const g = await createGpuStage2(prob, p0, 23);
      try { g.encodeIters(3); await g.done(); const t0 = performance.now(); g.encodeIters(20); await g.done(); N.estimate = { path: 'webgpu', sPerIter: (performance.now() - t0) / 1000 / 20 }; } finally { g.destroy(); }
    } else {
      const eng = await netEngine(prob, p0.length);
      await eng.evaluate(p0, true);
      const t0 = performance.now();
      for (let i = 0; i < 5; i++) await eng.evaluate(p0, true);
      N.estimate = { path: eng.path, sPerIter: (performance.now() - t0) / 1000 / 5 };
    }
    const min = (N.estimate.sPerIter * N.iterations) / 60;
    if (min > 5) N.msgs.push({ code: 'net_slow', vars: { a: Math.max(1, Math.ceil(min)) } });
  } catch (e) { N.msgs.push(netError(e)); }
  S.busy = '';
  renderBusy();
  renderStage5();
}
/** RGB-ANNAbel stage 2 on the emission of stage 4, then T and f_s of the network replace those of the
 *  closed form inside the display mask of the app. The closed form stays as the cross-check. */
async function runNet() {
  const t = S.tfs.out, N = S.tfs.net, p = S.flame.prepared;
  if (!t || !p) return;
  const inp = netInput(t);
  if (!inp) return;
  N.running = true; N.msgs = []; N.ctrl = new AbortController(); N.tick = null; N.result = null;
  renderStage5();
  try {
    const prob = buildStage2Problem(inp.J, inp.grid, inp.kern, { iterations: N.iterations });
    const p0 = initStage2(S.res.seed, prob.sizes);
    const topts = { iterations: N.iterations, signal: N.ctrl.signal, onTick: (tk: TrainTick) => { N.tick = tk; renderNetProgress(); } };
    let train: TrainResult;
    if (S.gpu) {
      const g = await createGpuStage2(prob, p0, N.iterations);
      try { train = await g.train({ ...topts, itersPerSubmit: 200 }); } finally { g.destroy(); }
    } else train = await trainStage2(await netEngine(prob, p0.length), p0, prob, topts);
    const res = finishStage2(prob, inp.J, inp.grid, inp.kern, train);
    if (train.endReason === 'stopped_by_user') N.msgs.push({ code: 'net_stopped', vars: { n: train.bestIter }, level: 'info' });
    const nr = p.nr, n = p.nz * nr;
    const Tcf = new Float64Array(n).fill(NaN), fsCf = new Float64Array(n).fill(NaN);
    const T = new Float64Array(n).fill(NaN), fs = new Float64Array(n).fill(NaN);
    const used = new Uint8Array(n);
    inp.keep.forEach((k, kk) => { for (let i = 0; i < nr; i++) { const q = k * nr + i, sq = kk * nr + i; Tcf[q] = res.Tcf[sq]; fsCf[q] = res.fsCf[sq]; if (!(t.mask[q] & (M_SAT | M_LOW)) && res.mask[sq]) { T[q] = res.T[sq]; fs[q] = res.fs[sq]; used[q] = 1; } } });
    for (let q = 0; q < n; q++) { t.mask[q] = used[q] ? t.mask[q] | 1 : (t.mask[q] & ~1) | (t.mask[q] & (M_SAT | M_LOW) ? 0 : M_LOW); t.flags[q] = 0; t.chi2rel[q] = NaN; }
    t.T = T; t.fs = fs;
    N.result = { xcheck: res.xcheck, bounds: res.bounds, Tcf, fsCf, bestIter: train.bestIter, completed: train.completedIters, planned: train.plannedIters, path: train.path, elapsedS: train.elapsedS, fsRef: inp.kern.fsRef, nTrain: res.meta.nTrain, nMask: res.meta.nMask, rows: inp.keep.length };
  } catch (e) { N.msgs.push(netError(e)); S.tfs.out = null; }
  N.running = false; N.ctrl = null;
}
/** Values of the zone used (mask bit 1), sorted. */
function usedSorted(v: ArrayLike<number>): number[] {
  const t = S.tfs.out;
  if (!t) return [];
  const out: number[] = [];
  for (let q = 0; q < v.length; q++) if (t.mask[q] === 1 && fin(v[q])) out.push(v[q]);
  return out.sort((a, b) => a - b);
}
const pctOf = (s: number[], q: number) => (s.length ? s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))] : NaN);
/** Colour scales from the zone used (GB3r1-fisico-05, estudiante-18). T from p0,5 to p99,5 rounded to 10 K, f_s to p95. */
function defaultScales() {
  const t = S.tfs.out;
  if (!t) return;
  const T = usedSorted(t.T);
  if (!S.res.Tmanual && T.length) { S.res.Tmin = Math.floor(pctOf(T, 0.005) / 10) * 10; S.res.Tmax = Math.ceil(pctOf(T, 0.995) / 10) * 10; if (S.res.Tmax <= S.res.Tmin) S.res.Tmax = S.res.Tmin + 10; }
  const fs = fsShown();
  if (fs) { const v = usedSorted(fs.v).filter((x) => x > 0); S.res.fsMax = v.length ? pctOf(v, 0.95) : 1; }
}
async function runBandNow() {
  const p = S.flame.prepared;
  if (!p || !S.tfs.out) return;
  S.busy = L(`Banda de incertidumbre, ${S.res.nBand} repeticiones`, `Uncertainty band, ${S.res.nBand} repetitions`);
  renderBusy();
  try {
    const c = S.flame.calUsed ? S.cal : null;
    const sKrel = (c && c.mode === 'oven' && S.oven.result ? CH.map((ch) => { const x = S.oven.result!.fit.channels[ch]; return fin(x.sKstat / x.K) ? x.sKstat / x.K : 0; }) : [0, 0, 0]) as [number, number, number];
    const scaleRel = fin(S.flame.mmPerPx) && fin(S.flame.scaleUnc) ? S.flame.scaleUnc / 100 : 0;
    S.res.bandHow = c?.mode === 'oven' ? 'oven' : c ? 'noise' : 'nocal';
    S.res.band = await ask({ type: 'band.run', n: S.res.nBand, seed: S.res.seed, sigmaT: c && fin(c.sigmaT) ? c.sigmaT : 0, Tcal: c && fin(c.Tcal) ? c.Tcal : NaN, eps: S.eps, sKrel, scaleRel, run: abelRun(), tfs: { path: workerPath(), pair: S.tfs.pair, emLaw: emLaw(), sigFrac: S.tfs.sigFrac, islandFrac: S.tfs.islandFrac, chemBG: calDecision().state !== 'uncalibrated', zCutMm: fin(S.tfs.zCut) ? S.tfs.zCut : null, chemThr: S.tfs.chemThr, fsChannel: S.tfs.fsChannel, refine: false, weights: tfsWeights() }, psi: psiMsg() });
  } catch (e) { S.flame.msgs.push({ code: (e as { code: string }).code }); }
  S.busy = '';
  renderBusy();
  renderStage6();
}
/** f_s as shown, ppm with an absolute calibration, relative to its maximum with a relative one, null otherwise. */
function fsShown(): { v: Float64Array; unit: string; factor: number; ref?: string } | null {
  const t = S.tfs.out;
  if (!t) return null;
  const d = S.stale[6] && S.tfs.dec ? S.tfs.dec : calDecision();
  if (d.fs === 'no') return null;
  if (d.fs === 'ppm') return { v: t.fs.map((x) => x * 1e6), unit: 'ppm', factor: 1e6 };
  // relative to the 95th percentile of the zone used, not to the maximum, which is noise on the axis (GB3r2-adv-06)
  const v = usedSorted(t.fs).filter((x) => x > 0);
  const ref = v.length ? pctOf(v, 0.95) : 1;
  return { v: t.fs.map((x) => x / ref), unit: L('relativa', 'relative'), factor: 1 / ref, ref: L('su percentil 95 en la zona usada', 'its 95th percentile in the zone used') };
}
async function recommended() {
  if (!flameItems().length) { S.flame.msgs = [{ code: 'need_stage', vars: { a: '2' } }]; renderAll(); return; }
  if (!fin(S.flame.mmPerPx)) { S.flame.msgs = [{ code: 'scale_needed', level: 'error' }, ...S.flame.msgs.filter((m) => m.code !== 'scale_needed')]; renderAll(); announce(3); document.getElementById('wr-mmpx')?.focus(); return; }
  let autoFit = false;
  if ((!S.cal || S.stale[1]) && S.calPath === 'oven' && S.oven.series.length) { await fitOvenNow(); autoFit = true; }
  S.abel.method = 'tikhonov'; S.abel.sym = 'average'; S.tfs.path = 'ratio'; S.tfs.pair = 'RG'; S.tfs.fsChannel = 'RG'; S.tfs.sigFrac = 0.02;
  await prepareFlameNow();
  if (!S.flame.prepared) return;
  // an upside-down flame passes every test, so the student says where the burner is, now with the preview in sight (GB3r2-adv-02, r3-adv-01)
  if (!burnerGate()) return;
  // a fixed smoothing (the example, or an L-curve without a corner) stays as it is
  if (S.abel.lambdaAuto) {
    await runLcurve();
    if (S.abel.lambdaAuto && S.abel.lcurve) S.abel.lambdaRel = S.abel.lcurve.lambda;
  }
  await runAbelNow();
  await runTfsNow();
  await runBandNow();
  if (autoFit) { S.flame.msgs.unshift({ code: 'oven_fit_auto', level: 'info' }); renderStage3(); }
  announce(6);
  document.getElementById('wr-s6')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================================ downloads
function save(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}
/** Local date and time in file names, yyyymmdd_hhmm (GB3r2-estudiante-21, it was UTC). */
const stamp = () => { const d = new Date(), z = (x: number) => String(x).padStart(2, '0'); return `${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}_${z(d.getHours())}${z(d.getMinutes())}`; };
/** File names in the page language (GB3r2-estudiante-21). */
const FN = { results: () => L('resultados', 'results'), resultsPt: () => L('resultados_punto', 'results'), map: () => L('mapa', 'map'), all: () => L('pirometria_rgb', 'rgb_pyrometry'), cal: () => L('calibracion', 'calibration') };
function settings() {
  return {
    language: getLang(), grouping: S.group ? '2x2' : '1x1', bayer: S.bayer, centreRB: S.centre, shiftTest: shiftRecord(),
    response: { preset: S.presetId, presetWhy: S.presetWhy, manual: S.presetManual, basis: S.basis, irCut: irCutNow(), shortPass: spNow(), custom: S.custom?.name ?? null },
    oven: { eps: S.eps, sigmaT: S.sigmaT, emissivityKnown: S.known, darkMode: darkModeNow(), perT: S.perT, calibrateAt: S.calibrateAt, tailCalibration: S.tailOn, plateaus: S.oven.plateaus },
    sphere: { valUnit: S.sphere.valUnit, lamUnit: S.sphere.lamUnit, nd: S.sphere.nd, od: S.sphere.od, fill: S.sphere.fill, tSphUs: S.sphere.tSphUs },
    calPath: S.calPath,
    flame: { burnerConfirmed: S.flame.burnerOk, texpUs: S.flame.texpUs, texpConfirmed: S.flame.texpConfirmed, times: S.flame.times, sensorQ: S.flame.sensorQ, quarter: S.flame.quarter, mirror: S.flame.mirror, fineDeg: S.flame.fineDeg, axisAuto: S.flame.axisAuto, axisManual: S.flame.axisManual, axisUsed: S.flame.prepared?.axis ?? null, halfWidth: S.flame.halfWidth, nr: S.flame.nr, rowTop: S.flame.rowTop, rowBottom: S.flame.rowBottom, rowsPerHeight: S.flame.rowsPerHeight, mmPerPx: S.flame.mmPerPx, scaleConfirmed: S.flame.scaleConfirmed, scaleUncPct: S.flame.scaleUnc, burnerRow: S.flame.burnerRow, zBurner: S.flame.zBurner },
    abel: { method: S.abel.outMethod || S.abel.method, sym: S.abel.sym, lambdaRel: S.abel.lambdaRel, lambdaAuto: S.abel.lambdaAuto, reg: 'gamma2', segments: S.abel.segments, satLambda: S.abel.satLambda, annabelIterations: S.abel.annabel.iterations, annabelBest: S.abel.annabel.result?.bestIter ?? null },
    tfs: { path: S.tfs.path, pair: S.tfs.pair, sigFrac: S.tfs.sigFrac, islandFrac: S.tfs.islandFrac, zCutMm: S.tfs.zCut, chemThr: S.tfs.chemThr, fsChannel: S.tfs.fsChannel,
           net: S.tfs.path === 'net' ? { method: 'RGB-ANNAbel stage 2 (Magana et al. 2025), annabel_app/core/stage2.py', iterations: S.tfs.net.iterations, bestIter: S.tfs.net.result?.bestIter ?? null, completed: S.tfs.net.result?.completed ?? null, path: S.tfs.net.result?.path ?? null, fsRef: S.tfs.net.result?.fsRef ?? null,
             config: { hiddenLayers: STAGE2_DEFAULTS.hiddenLayers, width: STAGE2_DEFAULTS.width, zSub: STAGE2_DEFAULTS.zSub, signalThreshold: STAGE2_DEFAULTS.signalThreshold, lambdaRatio: STAGE2_DEFAULTS.lambdaRatio, lr: STAGE2_DEFAULTS.lr, lrMinFrac: STAGE2_DEFAULTS.lrMinFrac }, xcheck: S.tfs.net.result?.xcheck ?? null } : null },
    results: { Tmin: S.res.Tmin, Tmax: S.res.Tmax, heights: S.res.heights, nBand: S.res.nBand },
  };
}
function runRecord() {
  const p = S.flame.prepared, t = S.tfs.out;
  let Tmax = -Infinity, at: [number, number] = [NaN, NaN], fsMax = -Infinity, atf: [number, number] = [NaN, NaN];
  const fs = fsShown();
  if (p && t) for (let k = 0; k < p.nz; k++) for (let i = 0; i < p.nr; i++) {
    const q = k * p.nr + i;
    if (fin(t.T[q]) && t.T[q] > Tmax) { Tmax = t.T[q]; at = [p.zMm[k], p.rMm[i]]; }
    if (fs && fin(fs.v[q]) && fs.v[q] > fsMax) { fsMax = fs.v[q]; atf = [p.zMm[k], p.rMm[i]]; }
  }
  const used = new Set(flameIds());
  const d = calDecision();
  const Tu = t ? usedSorted(t.T) : [], Fu = fs ? usedSorted(fs.v) : [];
  return {
    tool: { name: TOOL_NAME, version: TOOL_VERSION, commit: S.o.commit, source_sha256: S.o.source ?? null, built: S.o.built ?? null, url: S.o.pageUrl },
    created: new Date().toString(), createdISO: new Date().toISOString(),
    inputs: S.items.map((it) => ({ role: it.role, name: it.path, bytes: it.size, sha256: it.sha ?? null, sha256How: it.shaHow ?? null, width: it.fi?.width ?? null, height: it.fi?.height ?? null, encoding: it.fi?.encoding ?? null, ceiling: it.fi?.ceiling ?? null, T_C: it.tC ?? null, texpUs: it.texpUs ?? null, included: it.role === 'flame' ? used.has(it.id) : it.role !== 'ignore', why: it.role === 'flame' && !used.has(it.id) ? (it.why || 'excluded') : null })),
    settings: settings(), calibration: S.cal, calibrationState: d, emLaw: emLaw(),
    psi: { source: psiNow().name, sha256: psiSha(), basis: S.basis },
    seed: S.res.seed,
    results: p && t ? { nz: p.nz, nr: p.nr, drMm: p.drMm, zMm: [p.zMm[0], p.zMm[p.nz - 1]], used_pixels: Tu.length, T_median_K: pctOf(Tu, 0.5), T_p5_K: pctOf(Tu, 0.05), T_p95_K: pctOf(Tu, 0.95), fs_median: fs ? pctOf(Fu, 0.5) : null, fs_p95: fs ? pctOf(Fu, 0.95) : null, Tmax_K: Tmax, Tmax_at_z_r_mm: at, fsMax: fs ? fsMax : null, fsUnit: fs ? (fs.unit === 'ppm' ? 'ppm' : 'relative') : null, fsMax_at_z_r_mm: atf, max_note: L('el máximo del campo incluye el ruido amplificado en el eje, lean la mediana y el percentil 95', 'the maximum of the field includes the noise amplified on the axis, read the median and p95'), method: S.abel.outMethod } : null,
    warnings: [...collectCodes()],
  };
}
function collectCodes(): Set<string> {
  const s = new Set<string>();
  for (const m of [...S.loadMsgs, ...S.flame.msgs, ...S.abel.annabel.msgs, ...S.calFileMsgs]) s.add(m.code);
  for (const c of calDecision().codes) s.add(c);
  return s;
}
/** Column name of f_s, with _provisional when it is (GB3r2-adv-04). */
function fsCol(fs: { unit: string } | null, d: CalDecision): string {
  if (!fs) return 'fs';
  return (fs.unit === 'ppm' ? 'fs_ppm' : 'fs_rel') + (d.provisional ? '_provisional' : '');
}
/** Why f_s is provisional, in words. */
function provReason(): string {
  const r: string[] = [];
  if (!S.flame.texpConfirmed) r.push(L('el tiempo de exposición de la llama no está confirmado', 'the flame exposure time is not confirmed'));
  if (!S.flame.scaleConfirmed) r.push(L('la escala en mm por píxel no está confirmada', 'the scale in mm per pixel is not confirmed'));
  if (calDecision().state === 'untested') r.push(L('la calibración del horno no se pudo probar', 'the oven calibration could not be tested'));
  return r.join(L(' y ', ' and ')) || L('falta confirmar un dato', 'a value is not confirmed');
}
function csvText(point = getLang() !== 'es'): string {
  const p = S.flame.prepared!, t = S.tfs.out!, J = S.abel.out!.J;
  const fs = fsShown();
  const dec = point ? ',' : ';';
  const n = (x: number) => (fin(x) ? (point ? String(+x.toPrecision(7)) : String(+x.toPrecision(7)).replace('.', ',')) : '');
  const d = S.stale[6] && S.tfs.dec ? S.tfs.dec : calDecision();
  // without a calibration j is not in W m⁻³ sr⁻¹ (GB3r2-adv-19), a provisional f_s says so in its column (adv-04)
  // with a relative calibration K_j carries an unknown common grey factor, j is not in W m⁻³ sr⁻¹ either (GB3r3-adv-08)
  const jU = d.state === 'uncalibrated' || d.fs === 'relative' ? 'rel' : 'W_m3_sr';
  const head = ['z_mm', 'r_mm', 'T_K', ...(fs ? [fsCol(fs, d)] : []), `jR_${jU}`, `jG_${jU}`, `jB_${jU}`, 'T_los_K', 'mask_bits'];
  const rows = [head.join(dec)];
  for (let k = 0; k < p.nz; k++) for (let i = 0; i < p.nr; i++) {
    const q = k * p.nr + i;
    rows.push([n(p.zMm[k]), n(p.rMm[i]), n(t.T[q]), ...(fs ? [n(fs.v[q])] : []), n(J[0][k][i]), n(J[1][k][i]), n(J[2][k][i]), n(t.Tlos[q]), String(t.mask[q])].join(dec));
  }
  const notes = [L('mask_bits 1 usado, 2 señal baja, 4 quimioluminiscencia, 8 altura saturada, 16 razón fuera de la tabla, sumados cuando hay varios', 'mask_bits 1 used, 2 low signal, 4 chemiluminescence, 8 saturated height, 16 ratio outside the table, added when several apply')];
  notes.push(stateLabel(d).replace(/f_s/g, 'fs'));
  if (fs && d.provisional) notes.push(L(`fs provisional porque ${provReason()}`, `fs provisional because ${provReason()}`));
  if (fs && fs.unit !== 'ppm') notes.push(L(`fs_rel es fs dividida por ${fs.ref}`, `fs_rel is fs divided by ${fs.ref}`));
  if (jU === 'rel') notes.push(d.state === 'uncalibrated' ? L('sin calibración j queda en unidades relativas', 'without a calibration j is in relative units') : L('con calibración relativa j lleva un factor gris común desconocido y queda en unidades relativas', 'with a relative calibration j carries an unknown common grey factor and is in relative units'));
  return notes.map((x, i) => `# ${i ? '' : `${TOOL_NAME} ${TOOL_VERSION} ${S.o.commit}. `}${x}`).join('\n') + '\n' + rows.join('\n') + '\n';
}
function npyFields(): Record<string, Uint8Array> {
  const p = S.flame.prepared!, t = S.tfs.out!, J = S.abel.out!.J;
  const fs = fsShown();
  const plane = (f: (k: number, i: number) => number) => { const a = new Float32Array(p.nz * p.nr); for (let k = 0; k < p.nz; k++) for (let i = 0; i < p.nr; i++) a[k * p.nr + i] = f(k, i); return npy(a, [p.nz, p.nr]); };
  const out: Record<string, Uint8Array> = {
    T_K: plane((k, i) => t.T[k * p.nr + i]),
    T_los_K: plane((k, i) => t.Tlos[k * p.nr + i]),
    mask: npy(Uint8Array.from(t.mask), [p.nz, p.nr]),
    r_mm: npy(Float64Array.from(p.rMm), [p.nr]), z_mm: npy(Float64Array.from(p.zMm), [p.nz]),
  };
  const d = S.stale[6] && S.tfs.dec ? S.tfs.dec : calDecision();
  const jr = d.state === 'uncalibrated' || d.fs === 'relative' ? '_rel' : '';
  out[`jR${jr}`] = plane((k, i) => J[0][k][i]); out[`jG${jr}`] = plane((k, i) => J[1][k][i]); out[`jB${jr}`] = plane((k, i) => J[2][k][i]);
  if (fs) out[fsCol(fs, d)] = plane((k, i) => fs.v[k * p.nr + i]);
  return out;
}
/** Extent of a result map in mm, cell edges, for the axes (director after GB3 r1). */
function mapExtent(p: Prepared): MapExtent & { aspect: number } {
  const dz = p.zMm.length > 1 ? (p.zMm[p.nz - 1] - p.zMm[0]) / (p.nz - 1) : p.drMm;
  const rMax = p.rMm[p.nr - 1] + p.drMm / 2;
  const zBot = Math.min(p.zMm[0], p.zMm[p.nz - 1]) - Math.abs(dz) / 2, zTop = Math.max(p.zMm[0], p.zMm[p.nz - 1]) + Math.abs(dz) / 2;
  return { rMax, zBot, zTop, aspect: (2 * rMax) / (zTop - zBot) };
}
const tickFmt = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? num(Math.round(v)) : num(v, 1)).replace(/^-/, '−');
async function pngOf(which: 'T' | 'fs'): Promise<Blob | null> {
  const cv = document.getElementById(which === 'T' ? 'wr-map-T' : 'wr-map-fs') as HTMLCanvasElement | null;
  if (!cv) return null;
  const fs = fsShown();
  const p = S.flame.prepared!;
  const label = calDecision();
  const lab = stateLabel(label);
  return mapPng(cv, which === 'T' ? 't' : 'fs', {
    title: which === 'T' ? `T(r, z) · ${lab}` : `fₛ(r, z) · ${lab}`, unit: which === 'T' ? 'K' : fs?.unit ?? '',
    vmin: which === 'T' ? num(S.res.Tmin) : '0', vmax: which === 'T' ? num(S.res.Tmax) : sig(fs ? S.res.fsMax : 1, 2),
    axes: L(`r de −${num(p.rMm[p.nr - 1] + p.drMm / 2, 2)} a ${num(p.rMm[p.nr - 1] + p.drMm / 2, 2)} mm, z de ${num(p.zMm[0], 1)} a ${num(p.zMm[p.nz - 1], 1)} mm`, `r from −${num(p.rMm[p.nr - 1] + p.drMm / 2, 2)} to ${num(p.rMm[p.nr - 1] + p.drMm / 2, 2)} mm, z from ${num(p.zMm[0], 1)} to ${num(p.zMm[p.nz - 1], 1)} mm`),
  }, 4, { ...mapExtent(p), fmt: tickFmt });
}
async function downloadAll() {
  const enc = new TextEncoder();
  const files: { path: string; data: Uint8Array }[] = [];
  files.push({ path: 'run.json', data: enc.encode(JSON.stringify(runRecord(), null, 1)) });
  if (S.cal) files.push({ path: `${FN.cal()}.json`, data: enc.encode(JSON.stringify(S.cal, null, 1)) });
  if (S.tfs.out && S.flame.prepared && S.abel.out) {
    files.push({ path: `${FN.results()}.csv`, data: enc.encode(csvText()) });
    for (const [k, v] of Object.entries(npyFields())) files.push({ path: `npy/${k}.npy`, data: v });
    for (const w of ['T', 'fs'] as const) { const b = await pngOf(w); if (b) files.push({ path: `${FN.map()}_${w}_${theme()}.png`, data: new Uint8Array(await b.arrayBuffer()) }); }
  }
  save(writeZip(files), `${FN.all()}_${stamp()}.zip`);
}

async function clearTable() {
  await ask({ type: 'forget', ids: S.items.map((it) => it.id) });
  S.items = []; S.oven.series = []; S.oven.result = null; S.cal = S.cal?.mode === 'file' ? S.cal : null; S.flame.prepared = null; S.abel.out = null; S.tfs.out = null;
  S.crops = {}; S.log = []; S.flame.loaded = false; S.flame.rulerPreview = null; S.flame.rulerId = ''; S.flame.shift = null; S.flame.instant = ''; S.loadMsgs = []; S.flame.msgs = []; S.sphere.result = null; S.stale = {}; S.truth = null;
}

// ============================================================================ run.json in
async function loadRun(b: Blob, name: string) {
  try {
    const j = JSON.parse(await b.text()) as { tool?: { name?: string }; settings?: ReturnType<typeof settings>; inputs?: { name: string; sha256: string; role: string }[]; seed?: number };
    if (j.tool?.name !== TOOL_NAME || !j.settings) throw new Error('run_bad');
    const s = j.settings;
    S.presetId = s.response.preset as Preset['id']; S.presetManual = s.response.manual; S.basis = s.response.basis;
    if (s.response.irCut) { S.irOn = true; S.irCutNm = s.response.irCut.cutNm; S.irWidthNm = s.response.irCut.widthNm; } else S.irOn = false;
    S.sp = !!(s.response as { shortPass?: ShortPass | null }).shortPass;
    const wasCentre = S.centre;
    S.centre = (s as { centreRB?: boolean }).centreRB === true;
    if (S.centre !== wasCentre) reopts();
    Object.assign(S, { eps: s.oven.eps, sigmaT: s.oven.sigmaT, known: s.oven.emissivityKnown, perT: s.oven.perT, calibrateAt: s.oven.calibrateAt, calPath: s.calPath });
    S.darkMode = s.oven.darkMode as typeof S.darkMode;
    S.tailOn = (s.oven as { tailCalibration?: boolean }).tailCalibration !== false;
    const f = s.flame;
    Object.assign(S.flame, { burnerOk: f.burnerConfirmed === true, texpUs: f.texpUs ?? NaN, texpConfirmed: f.texpConfirmed, times: f.times, sensorQ: f.sensorQ, quarter: f.quarter, mirror: f.mirror, fineDeg: f.fineDeg, axisAuto: f.axisAuto, axisManual: f.axisManual ?? NaN, halfWidth: f.halfWidth ?? NaN, nr: f.nr ?? NaN, rowTop: f.rowTop ?? NaN, rowBottom: f.rowBottom ?? NaN, rowsPerHeight: f.rowsPerHeight, mmPerPx: f.mmPerPx ?? NaN, scaleConfirmed: f.scaleConfirmed, scaleUnc: f.scaleUncPct, burnerRow: f.burnerRow ?? NaN, zBurner: f.zBurner });
    // a log repeats its own smoothing, with or without the key lambdaAuto (DESIGN §9, C4)
    Object.assign(S.abel, { method: s.abel.method, sym: s.abel.sym, lambdaRel: s.abel.lambdaRel, lambdaAuto: false, lamFlat: false, segments: s.abel.segments, satLambda: s.abel.satLambda });
    S.abel.annabel.iterations = s.abel.annabelIterations ?? ITER_LONG;
    S.tfs.net.iterations = (s.tfs as { net?: { iterations?: number } | null }).net?.iterations ?? NET_ITERATIONS;
    Object.assign(S.tfs, { path: asTfsPath(s.tfs.path), pair: s.tfs.pair, sigFrac: s.tfs.sigFrac, islandFrac: s.tfs.islandFrac ?? 0.05, zCut: s.tfs.zCutMm ?? NaN, chemThr: s.tfs.chemThr, fsChannel: asFsMode(s.tfs.fsChannel) });
    Object.assign(S.res, { Tmin: s.results.Tmin, Tmax: s.results.Tmax, heights: s.results.heights ?? [], nBand: s.results.nBand ?? 30, seed: j.seed ?? 42 });
    S.runRef = { name, inputs: (j.inputs ?? []).filter((x) => x.sha256) };
    S.loadMsgs.push({ code: 'run_loaded', vars: { file: name }, level: 'info' });
    checkRun();
  } catch { S.loadMsgs.push({ code: 'run_bad' }); }
  renderAll();
}
function checkRun() {
  const r = S.runRef;
  if (!r || !S.items.length) return;
  S.loadMsgs = S.loadMsgs.filter((m) => m.code !== 'run_sha_ok' && m.code !== 'run_sha_mismatch');
  const bad: string[] = [];
  let ok = 0;
  for (const ref of r.inputs) {
    const base = ref.name.split('/').pop();
    const it = S.items.find((x) => x.sha === ref.sha256) ?? S.items.find((x) => x.path.split('/').pop() === base);
    if (!it?.sha) continue;
    if (it.sha === ref.sha256) ok++; else bad.push(it.path);
  }
  if (bad.length) S.loadMsgs.push({ code: 'run_sha_mismatch', vars: { n: bad.length }, files: bad });
  else if (ok) S.loadMsgs.push({ code: 'run_sha_ok', vars: { n: ok }, level: 'info' });
}

// ============================================================================ rendering helpers
const plots = new Map<string, LinePlot>();
function plot(id: string, o: { xlabel?: string; ylabel?: string; logy?: boolean; height?: number; legend?: boolean }, series: Series[], lim: { xlim?: [number, number]; ylim?: [number, number] } = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  let p = plots.get(id);
  if (!p || !el.contains(p.svg)) { el.replaceChildren(); p = new LinePlot(el, { width: 440, height: o.height ?? 250, xlabel: o.xlabel, ylabel: o.ylabel, logy: o.logy, legend: o.legend ?? false }); plots.set(id, p); }
  // LinePlot draws every point, so masked values (NaN) are dropped and the line joins the finite ones
  const clean = series.map((sr) => {
    const keep: number[] = [];
    for (let i = 0; i < sr.x.length; i++) if (fin(sr.x[i]) && fin(sr.y[i]) && (!sr.y2 || fin(sr.y2[i]))) keep.push(i);
    if (keep.length === sr.x.length) return sr;
    return { ...sr, x: keep.map((i) => sr.x[i]), y: keep.map((i) => sr.y[i]), y2: sr.y2 ? keep.map((i) => sr.y2![i]) : undefined };
  }).filter((sr) => sr.x.length > 0);
  p.render(clean, lim);
}
const chCls = ['s-chR', 's-chG', 's-chB'];
/** A selector that finds the same control after a section is drawn again, by id or by its data attributes (GB3r1-web-01). */
const FOCUS_KEYS = ['method', 'calpath', 'series', 'view', 'incl', 'npy'];
function focusKey(a: HTMLElement): string | null {
  if (a.id) return `#${CSS.escape(a.id)}`;
  const d = a.dataset;
  if (d.id && d.f) return `[data-id="${CSS.escape(d.id)}"][data-f="${CSS.escape(d.f)}"]`;
  for (const k of FOCUS_KEYS) if (d[k] !== undefined) return `[data-${k}="${CSS.escape(d[k]!)}"]`;
  return null;
}
function keepFocus(fn: () => void) {
  const a = document.activeElement as HTMLElement | null;
  const key = a && a !== document.body ? focusKey(a) : null;
  const sel = a && (a as HTMLInputElement).selectionStart;
  fn();
  if (!key || !a || a.isConnected) return;
  const b = document.querySelector(key) as HTMLInputElement | null;
  if (b && b !== a) { b.focus({ preventScroll: true }); try { if (sel !== null && sel !== undefined && b.setSelectionRange) b.setSelectionRange(sel, sel); } catch { /* not a text field */ } }
}
/**
 * One persistent live region for the result of each action, and one alert region for errors (GB3r1-web-02).
 * Regions that are drawn again with their text are not announced by screen readers.
 */
function announce(stage: number) {
  const sec = document.getElementById(`wr-s${stage}`);
  const st = document.getElementById(`wr-st${stage}`)?.textContent?.trim() ?? '';
  const warn = sec ? sec.querySelectorAll('.wr-msg--warn').length : 0;
  const errs = sec ? Array.from(sec.querySelectorAll('.wr-msg--error')).map((e) => e.textContent?.trim() ?? '') : [];
  const head = sec?.querySelector('h2')?.textContent?.trim() ?? '';
  const a = document.getElementById('wr-announce');
  if (a) a.textContent = `${head}. ${st}.${warn ? L(` ${warn} ${warn === 1 ? 'aviso' : 'avisos'} en este paso.`, ` ${warn} ${warn === 1 ? 'warning' : 'warnings'} in this step.`) : ''}`;
  const e = document.getElementById('wr-alert');
  if (e) e.textContent = errs.join(' ');
}
let pointerDown = false, pendingRender = false;
/** Draws again after the click that blurred a field, so that click reaches its button (GB3r1-estudiante-02). */
function renderSoon() { if (pointerDown) pendingRender = true; else renderAll(); }
function renderBusy() {
  const e = document.getElementById('wr-busy');
  if (e) { e.textContent = S.busy; e.hidden = !S.busy; }
}
function renderAll() {
  keepFocus(() => { renderStage0(); renderStage1(); renderStage2(); renderStage3(); renderStage4(); renderStage5(); renderStage6(); renderNav(); });
  renderBusy();
}
/** The flat zone lives in step 1, in the view or in Options, so its tabs draw the step again. */
function renderPlateau() { renderStage1(); }
function stateLabel(d: CalDecision): string {
  const base = d.state === 'absolute' ? L('Calibración absoluta', 'Absolute calibration') : d.state === 'relative' ? L('Calibración relativa. f_s en unidades relativas', 'Relative calibration. f_s in relative units') : d.state === 'inconsistent' ? L('Calibración no consistente', 'Inconsistent calibration') : d.state === 'untested' ? L('Calibración sin probar, una sola temperatura del horno', 'Untested calibration, a single oven temperature') : d.state === 'one_channel' ? L('Un solo canal calibrado', 'Only one channel calibrated') : L('Sin calibrar', 'Uncalibrated');
  return d.provisional && d.fs !== 'no' ? `${base}. ${L('f_s provisional', 'f_s provisional')}` : base;
}
function tableHtml(head: string[], rows: string[][], cls = '', caption = ''): string {
  return `<div class="wr-tablewrap"><table class="${cls}">${caption ? `<caption>${caption}</caption>` : ''}<thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
const numIn = (id: string, value: number, attrs: string, label: string, note = '') => `<label class="control wr-num"><span class="control__head"><span>${label}</span></span><input type="number" id="${id}" value="${fin(value) ? value : ''}" ${attrs}>${note ? `<span class="meta">${note}</span>` : ''}</label>`;
const sel = (id: string, opts: [string, string][], value: string, label: string, note = '') => `<label class="control"><span class="control__head"><span>${label}</span></span><select id="${id}">${opts.map(([v, t]) => `<option value="${v}"${v === value ? ' selected' : ''}>${t}</option>`).join('')}</select>${note ? `<span class="meta">${note}</span>` : ''}</label>`;
const chk = (id: string, on: boolean, label: string, note = '') => `<label class="wr-chk"><input type="checkbox" id="${id}"${on ? ' checked' : ''}> ${label}${note ? `<span class="meta">${note}</span>` : ''}</label>`;
const btn = (id: string, text: string, disabled = false) => `<button class="wr-btn wr-btn--quiet" type="button" id="${id}"${disabled ? ' disabled' : ''}>${text}</button>`;
const fileBtn = (id: string, text: string, accept: string, dir = false) => `<label class="wr-btn wr-btn--box wr-filebtn">${text}<input type="file" id="${id}" ${accept ? `accept="${accept}"` : ''} ${dir ? 'webkitdirectory' : ''} multiple class="sr-only"></label>`;

// ============================================================================ step layout (DESIGN.md §0)
/**
 * Five steps in the order of the lab. Each step shows what it is for, its inputs, one main button and its
 * result with one check. Everything else goes in a single closed fold, Options, at the end of the step.
 */
type Kind = 'todo' | 'ok' | 'warn' | 'stop' | 'stale';
interface Pill { k: Kind; t: string }
const pTodo = (): Pill => ({ k: 'todo', t: L('Pendiente', 'To do') });
const pOk = (es = '', en = ''): Pill => ({ k: 'ok', t: es ? L(`Listo, ${es}`, `Done, ${en}`) : L('Listo', 'Done') });
const pWarn = (es = '', en = ''): Pill => ({ k: 'warn', t: es ? L(`Aviso, ${es}`, `Warning, ${en}`) : L('Aviso', 'Warning') });
const pStop = (es: string, en: string): Pill => ({ k: 'stop', t: L(`Detenido, falta ${es}`, `Stopped, ${en} missing`) });
const pStale = (): Pill => ({ k: 'stale', t: L('Desactualizado, vuelvan a calcular', 'Out of date, compute again') });
const pBusy = (t: string): Pill => ({ k: 'todo', t });
const BURNER_STOP = (): Pill => pStop('confirmar el quemador', 'burner confirmation');
function stepHead(n: number, title: string, pillId: string, p: Pill, purpose: string): string {
  return `<header class="wr-head"><h2>${n} · ${subs(title)}</h2><p class="wr-pill wr-pill--${p.k}"><span id="${pillId}">${subs(p.t)}</span></p></header>
  <p class="wr-purpose">${subs(purpose)}</p>`;
}
/** The one fold of a step. Its open state lives in S.ui.open, so drawing the step again keeps it open. */
function optsBlock(n: number, parts: string[]): string {
  const body = parts.filter(Boolean).join('');
  if (!body) return '';
  return `<details class="wr-opts" id="wr-opt-${n}"${S.ui.open[n] ? ' open' : ''}><summary>${L('Opciones', 'Options')}</summary><div class="wr-opts__body">${body}</div></details>`;
}
const osec = (title: string, body: string, id = '') => (body.trim() ? `<section class="wr-osec"${id ? ` id="${id}"` : ''}><h3>${subs(title)}</h3>${body}</section>` : '');
const RETAKE = ['saturated', 'plateau_moved', 'dark_clipped', 'dark_clipped_nodark', 'texp_outside_cal', 'quick_zeros'];
/** Messages split between the view (errors and the warnings of `view`) and the Options of the step (the rest). */
function splitMsgs(list: Msg[], view: string[]): { v: Msg[]; o: Msg[] } {
  const v: Msg[] = [], o: Msg[] = [];
  for (const m of list) {
    if (m.code === 'example_preset') continue;
    // an example cannot be photographed again, so the advice to retake goes to Options
    if (S.example && RETAKE.includes(m.code) && m.level !== 'error') { o.push(m); continue; }
    if (m.level === 'error' || (m.level !== 'info' && view.includes(m.code))) v.push(m); else o.push(m);
  }
  return { v, o };
}
const msgBox = (l: Msg[], id = '') => (l.length ? `<div class="wr-msgs"${id ? ` id="${id}"` : ''}>${msgHtml(l)}</div>` : '');
const check = (ok: boolean, html: string, code = '') => `<p class="wr-check wr-check--${ok ? 'ok' : 'warn'}"${code ? ` data-code="${code}"` : ''}><span class="wr-check__mark" aria-hidden="true">${ok ? '✓' : '⚠'}</span> ${html}</p>`;
const pbtn = (id: string, text: string, disabled: boolean, why = '', extra = '') => `<p class="wr-act"><button class="wr-btn wr-btn--primary" type="button" id="${id}"${disabled ? ' disabled' : ''}>${subs(text)}</button>${extra}${disabled && why ? `<span class="wr-why">${why}</span>` : ''}</p>`;
/** Files that could not be read are said where they were loaded. */
const FILE_CODES = ['not_tiff', 'bigtiff', 'compression', 'predictor', 'sample_type', 'samples', 'truncated', 'u8_gamma', 'float_nosat', 'too_large', 'mixed_bits', 'missing_tag', 'lzw_bad_code', 'ycbcr', 'zip_unsupported', 'zip_bad', 'out_of_memory', 'rewritten', 'series_mismatch', 'response_bad', 'oven_no_aperture'];
function loadMsgsFor(step: 1 | 3): Msg[] { return S.loadFrom === step ? S.loadMsgs.filter((m) => !/^(example_|run_)/.test(m.code)) : []; }

function renderNav() {
  const el = document.getElementById('wr-nav');
  if (!el) return;
  const items: [number, string, string, Pill][] = [[1, 'wr-s1', L('Calibración', 'Calibration'), st1()], [2, 'wr-s3', L('Llama', 'Flame'), st3()], [3, 'wr-s4', 'Abel', st4()], [4, 'wr-s5', L('T y f_s', 'T and f_s'), st5()], [5, 'wr-s6', L('Resultados', 'Results'), st6()]];
  const word = (k: Kind) => (k === 'ok' ? L('listo', 'done') : k === 'todo' ? L('pendiente', 'to do') : L('con aviso', 'with a notice'));
  el.innerHTML = `<ol>${items.map(([n, id, name, p]) => `<li><a href="#${id}"><span class="wr-dot wr-dot--${p.k}" aria-hidden="true">${p.k === 'ok' ? '●' : p.k === 'todo' ? '○' : '◐'}</span> ${n}<span class="wr-stepname"> ${subs(name)}</span><span class="sr-only">, ${word(p.k)}</span></a></li>`).join('')}</ol>`;
}

// ============================================================================ top bar (stage 0)
function browserOk(): boolean { return typeof Worker !== 'undefined' && !!globalThis.crypto?.subtle && typeof DecompressionStream !== 'undefined'; }
function renderStage0() {
  const el = document.getElementById('wr-s0');
  if (!el) return;
  // the workshop example, the laboratory flame with the furnace it is calibrated with
  const k = Math.max(0, S.o.examples.findIndex((x) => /laboratorio/.test(x.file)));
  const ex = S.o.examples[k];
  const exMb = ex ? exampleSets(ex).reduce((a, x) => a + x.bytes, 0) / 1e6 : 0;
  const msgs: Msg[] = S.loadMsgs.filter((m) => /^example_/.test(m.code) && m.level !== 'info');
  if (!browserOk()) msgs.push({ code: 'browser_missing', level: 'error' });
  el.innerHTML = `<p class="wr-bar0">${ex ? `<button class="wr-btn wr-btn--box" type="button" id="wr-ex-${k}">${L('Cargar el ejemplo', 'Load the example')}</button> <span class="meta">${L(`Horno y llama del laboratorio, ${num(exMb, 0)} MB.`, `Furnace and laboratory flame, ${num(exMb, 0)} MB.`)}</span>` : ''}<span class="sr-only" id="wr-st0">${L('Listo', 'Done')}</span></p>
  ${msgBox(msgs)}`;
}
function renderQuick() {
  const el = document.getElementById('wr-quick');
  if (!el) return;
  if (!S.quick.length) { el.innerHTML = msgHtml(S.quickMsgs); return; }
  const msgs: Msg[] = [...S.quickMsgs];
  const rows = [...S.quick].sort((a, b) => (a.name.tC ?? 1e9) - (b.name.tC ?? 1e9) || (a.name.texpUs ?? 0) - (b.name.texpUs ?? 0)).map((it) => {
    const s = it.fi;
    if (it.status !== 'ok' || !s) {
      if (it.status === 'error') msgs.push({ code: it.err?.code ?? 'not_tiff', files: [it.path] });
      return [`<span class="wr-path">${esc(it.path)}</span>`, `<em>${it.status === 'error' ? L('no se pudo leer', 'could not be read') : L('leyendo', 'reading')}</em>`, '', '', ''];
    }
    if (s.encoding === 'u8') msgs.push({ code: 'u8_gamma', files: [it.path] });
    if (s.encoding === 'float') msgs.push({ code: 'float_nosat', files: [it.path] });
    const sat = s.satRaw.map((v, c) => (v > 0 ? chName(c) : '')).filter(Boolean);
    if (sat.length) msgs.push({ code: 'saturated', vars: { channel: sat.join(L(' y ', ' and ')) }, files: [it.path] });
    s.p999.forEach((v, c) => { if (s.satRaw[c] === 0 && v >= 0.9 * s.ceil12) msgs.push({ code: 'quick_high', vars: { channel: c }, files: [it.path] }); });
    if (s.zeroFrac.some((z) => z > 0.01)) msgs.push({ code: 'quick_zeros', files: [it.path] });
    const n = it.name;
    const read: string[] = [];
    if (n.tC !== undefined) read.push(tLabel(n.tC));
    if (n.texpUs !== undefined) read.push(`${num(n.texpUs)} µs`);
    if (n.role !== 'oven') read.push(roleName(n.role));
    return [`<span class="wr-path">${esc(it.path)}</span>`, s.p999.map((v) => pct(v / s.ceil12, 0)).join(' / '), s.satRaw.map((v) => num(v)).join(' / '), s.zeroFrac.map((z) => pct(z, 1)).join(' / '), read.length ? read.join(', ') : `<em>${L('no se entiende', 'not understood')}</em>`];
  });
  el.innerHTML = tableHtml([L('Archivo', 'File'), L('Uso de la escala', 'Use of full scale') + ' R / G / B', L('Saturados', 'Saturated') + ' R / G / B', L('Ceros', 'Zeros') + ' R / G / B', L('Nombre', 'Name')], rows, 'wr-quick') + msgBox(msgs);
}

// ============================================================================ step 1 · calibration with the furnace
const ovenItems = () => S.items.filter((it) => it.role === 'oven');
const ovenMissing = () => S.items.filter((it) => it.role === 'oven' && it.status !== 'error' && (it.tC === undefined || it.texpUs === undefined));
function st1(): Pill {
  if (S.oven.analysing && S.busy) return pBusy(esc(S.busy));
  if (S.stale[1]) return pStale();
  const c = S.cal;
  if (c) {
    if (c.mode === 'sphere') return pOk('esfera integradora', 'integrating sphere');
    if (c.mode === 'file') return pOk('archivo', 'file');
    if ((c.calibrated ?? []).length === 1) return pWarn('un solo canal', 'one channel only');
    if (c.tested === false) return pWarn('sin probar', 'not tested');
    if (!c.consistent) return pWarn('no pasa la prueba', 'test not passed');
    return c.absolute ? pOk('absoluta', 'absolute') : pOk('relativa', 'relative');
  }
  if (ovenMissing().length) return pStop('la temperatura o el tiempo de algunos archivos', 'temperature or time of some files');
  return pTodo();
}
/** Warnings of the flat zone of one series, the same tests as before the redesign. */
function plateauMsgs(s: SeriesView): Msg[] {
  const pl = S.oven.plateaus[String(s.tC)];
  if (!pl) return [];
  const flat = s.prof?.flat ?? { R: s.plateau.flatness.R, G: s.plateau.flatness.G, B: s.plateau.flatness.B, RG: s.plateau.flatRG, BG: s.plateau.flatBG };
  const ok = Math.max(flat.R, flat.G, flat.B) <= 0.0101 && Math.max(flat.RG, flat.BG) <= 0.00505;
  const off = Math.hypot(pl.cx - s.aperture.cx, pl.cy - s.aperture.cy);
  const out: Msg[] = [];
  if (off + pl.radius > s.aperture.r) out.push({ code: 'plateau_outside', vars: { T: tLabel(s.tC) } });
  else if (!ok) out.push({ code: 'plateau_not_flat', vars: { T: tLabel(s.tC), a: Math.round(1000 * Math.max(flat.R, flat.G, flat.B)) / 10 } });
  if (off > 2) out.push({ code: 'plateau_off_centre', vars: { a: Math.round(off * 10) / 10 }, level: 'info' });
  return out;
}
/** The furnace moved, said once with the largest shift (DESIGN §7). */
function plateauMoved(): Msg[] {
  const cs = S.oven.series.map((x) => { const it = S.items.find((i) => i.id === x.plateauId); const c = it?.crop ?? { x0: 0, y0: 0 }; const sc = it?.fi?.scale ?? 1; return { x: c.x0 + sc * x.aperture.cx, y: c.y0 + sc * x.aperture.cy }; });
  let d = 0;
  for (let i = 1; i < cs.length; i++) d = Math.max(d, Math.hypot(cs[i].x - cs[i - 1].x, cs[i].y - cs[i - 1].y));
  return d > 3 ? [{ code: 'plateau_moved', vars: { a: Math.round(d) } }] : [];
}
/** Saturation and the dark level of the furnace images, in one notice each. */
function ovenImageMsgs(): Msg[] {
  const msgs: Msg[] = [];
  if (!S.oven.series.length) return msgs;
  const satCh = new Set<number>(), satFiles: string[] = [];
  for (const s of S.oven.series) s.texpUs.forEach((t, i) => {
    const sg = s.signals[i];
    let hit = false;
    CH.forEach((c, k) => { if (sg.nSat[c] > 0) { satCh.add(k); hit = true; } });
    if (hit) satFiles.push(`${tLabel(s.tC)}, ${num(t)} µs`);
    if (t < 100) msgs.push({ code: 'short_exposure', vars: { a: t }, level: 'info' });
  });
  if (satCh.size) msgs.push({ code: 'saturated', vars: { channel: [...satCh].sort().map((k) => chName(k)).join(L(' y ', ' and ')) }, files: satFiles, level: 'info' });
  const hasDark = darkItems().length > 0;
  if (!hasDark) msgs.push({ code: 'dark_estimated', level: 'info' });
  if (S.oven.series.some((s) => s.narrow)) msgs.push({ code: 'far_narrow', level: 'info' });
  const clipped = S.oven.series.filter((s) => s.zeroFrac > 0.01);
  if (clipped.length || S.oven.darks.some((d) => d.zeroFrac > 0.01)) msgs.push({ code: hasDark ? 'dark_clipped' : 'dark_clipped_nodark', files: clipped.map((s) => tLabel(s.tC)) });
  const sp = S.oven.result?.fit.strayPct;
  const spHits = sp ? CH.map((c, k) => (sp[c] > 1 ? `${num(Math.round(sp[c] * 10) / 10, 1)} % ${L('en', 'in')} ${chName(k)}` : '')).filter(Boolean) : [];
  if (spHits.length) msgs.push({ code: 'stray_light', vars: { a: spHits.join(', ') } });
  const worst = S.oven.series.reduce((a, s) => (s.hotCount.reduce((x, y) => x + y, 0) > a.n ? { n: s.hotCount.reduce((x, y) => x + y, 0), area: 3 * s.w * s.h } : a), { n: 0, area: 1 });
  if (worst.n) msgs.push({ code: 'hot_pixels', vars: { n: worst.n, a: num((100 * worst.n) / worst.area, 2) }, level: 'info' });
  msgs.push(...plateauMoved());
  return msgs;
}
const STEP1_VIEW = [...FILE_CODES, 'dark_clipped', 'dark_clipped_nodark', 'plateau_outside', 'plateau_not_flat', 'cal_one_channel', 'bayer_order', 'oven_fit_missing', 'preset_mismatch', 'cal_bad', 'name_unread', 'name_unread_T', 'name_unread_texp'];
function renderStage1() {
  const el = document.getElementById('wr-s1');
  if (!el) return;
  const ov = ovenItems(), darks = darkItems();
  const temps = new Set(ov.filter((it) => it.tC !== undefined).map((it) => it.tC)).size;
  const miss = ovenMissing();
  const ft = fileTable(() => true, 'wr-ft1');
  // the plateau preview comes to the view only when one of its series has a warning (DESIGN §8)
  const plBad = S.oven.series.findIndex((s) => plateauMsgs(s).some((m) => m.level !== 'info'));
  if (plBad >= 0 && !plateauMsgs(S.oven.series[S.oven.cur]).some((m) => m.level !== 'info')) S.oven.cur = plBad;
  const plView = plBad >= 0;
  const all = [...loadMsgsFor(1), ...ft.msgs, ...ovenImageMsgs(), ...(miss.length ? [{ code: 'oven_fit_missing', vars: { n: miss.length } } as Msg] : [])];
  const { v, o } = splitMsgs(all, STEP1_VIEW);
  const p = preset();
  const cam = cameraOf(S.items);
  const mismatch = cam && !cam.cam.includes(p.id) && !p.id.includes(cam.cam) && !S.custom;
  const respMsgs: Msg[] = [];
  if (mismatch) respMsgs.push({ code: 'preset_mismatch', vars: { a: p.id, b: cam!.cam } });
  if (p.allowRemoveIr && !S.custom && !S.irOn) respMsgs.push({ code: 'no_ir_cut' });
  const exOven = S.o.examples.findIndex((x) => /^horno/.test(x.file)), exSyn = S.o.examples.findIndex((x) => /sintetica/.test(x.file));
  el.innerHTML = `${stepHead(1, L('Calibración con el horno', 'Calibration with the furnace'), 'wr-st1', st1(), L('Mide cuánta señal da cada canal de la cámara con el horno a temperatura conocida.', 'Measures how much signal each camera channel gives with the furnace at a known temperature.'))}
  <div class="wr-drop" id="wr-drop">
    <h3 class="wr-droplab">${L('Fotos del horno y sus oscuros', 'Furnace photos and their darks')}</h3>
    <p class="cluster">${fileBtn('wr-o-dir', L('Elegir carpeta', 'Choose folder'), '', true)} ${fileBtn('wr-o-files', L('Elegir archivos o .zip', 'Choose files or .zip'), '.tif,.tiff,.zip,.csv,.pfs,.json')}</p>
    <p class="meta">${L('Oscuros, fotos con el lente tapado y el mismo tiempo de exposición.', 'Darks, photos with the lens covered and the same exposure time.')}</p>
  </div>
  ${ov.length ? `<p class="wr-read" id="wr-o-read">${L(`${temps} ${temps === 1 ? 'temperatura' : 'temperaturas'} · ${ov.length} ${ov.length === 1 ? 'imagen' : 'imágenes'} · ${darks.length} ${darks.length === 1 ? 'oscuro' : 'oscuros'}`, `${temps} ${temps === 1 ? 'temperature' : 'temperatures'} · ${ov.length} ${ov.length === 1 ? 'image' : 'images'} · ${darks.length} ${darks.length === 1 ? 'dark' : 'darks'}`)}</p>` : ''}
  ${miss.length ? ft.html : ''}
  ${msgBox(v)}
  ${plView ? `<div class="wr-result" id="wr-plateau-view">${plateauBlock()}</div>` : ''}
  ${pbtn('wr-oven-fit', L('Calcular la calibración', 'Compute the calibration'), !S.oven.series.length, L('Primero carguen las fotos del horno.', 'Load the furnace photos first.'))}
  <div class="wr-result" id="wr-s2"></div>
  ${optsBlock(1, [
    msgBox(o),
    osec(L('Archivos', 'Files'), `${miss.length ? '' : ft.html}${ft.pfs}${S.items.length ? `<p>${btn('wr-clear', L('Vaciar la tabla', 'Clear the table'))}</p>` : ''}`),
    S.oven.series.length && !plView ? osec(L('Zona plana', 'Flat zone'), plateauBlock()) : '',
    S.oven.result ? osec(L('Resultado detallado', 'Detailed result'), '<div id="wr-fit"></div>') : '',
    S.oven.result?.tail ? osec(L('Corregir la respuesta sobre 700 nm', 'Correct the response above 700 nm'), '<div id="wr-tail-box"></div>') : '',
    osec(L('Supuestos', 'Assumptions'), `<div class="wr-fields">
      ${numIn('wr-dT', S.sigmaT, 'min="0" max="50" step="1"', L('Incertidumbre del termopar (K)', 'Thermocouple uncertainty (K)'))}
      <label class="control"><span class="control__head">${L('Emisividad de la cavidad', 'Cavity emissivity')} <output id="wr-eps-out">${num(S.eps, 2)}</output></span><input type="range" id="wr-eps" aria-label="${L('Emisividad de la cavidad', 'Cavity emissivity')}" min="0.80" max="1" step="0.01" value="${S.eps}"></label>
      ${chk('wr-known', S.known, L('La emisividad de la cavidad y la transmisión de las ventanas son conocidas', 'The cavity emissivity and the window transmission are known'))}</div>`),
    S.oven.series.length ? osec(L('Oscuro', 'Dark'), '<div id="wr-dark"></div>') : '',
    osec(L('Respuesta de la cámara', 'Camera response'), `<div class="wr-grid2"><div class="wr-controls">
      ${sel('wr-preset', PRESETS.map((x) => [x.id, x.id === 'a2A1920-160ucPRO' ? L('Cámara del laboratorio', 'Lab camera') : L('Otra cámara', 'Other camera')] as [string, string]), S.presetId, L('Cámara', 'Camera'))}
      ${sel('wr-basis', [['photon', L('por fotón', 'per photon')], ['energy', L('por energía', 'per energy')]], S.basis, L('La curva se lee', 'The curve is read'))}
      ${p.allowRemoveIr && !S.custom ? `${chk('wr-ir', S.irOn, L('Corte infrarrojo', 'Infrared cut'))}<div class="wr-row">${numIn('wr-ircut', S.irCutNm, 'min="550" max="900" step="1"', L('Corte (nm)', 'Cut (nm)'))}${numIn('wr-irw', S.irWidthNm, 'min="1" max="60" step="1"', L('Ancho (nm)', 'Width (nm)'))}</div>` : ''}
      ${sel('wr-sp', [['none', L('ninguno', 'none')], ['sp700', L('pasa-cortos 700 nm', 'short-pass 700 nm')]], S.sp ? 'sp700' : 'none', L('Filtro delante del lente', 'Filter in front of the lens'))}
      <p class="cluster">${fileBtn('wr-resp', L('Cargar otra curva CSV', 'Load another CSV curve'), '.csv,text/csv')}${S.custom ? ` ${btn('wr-resp-clear', L('Volver a la curva de la cámara', 'Back to the camera curve'))}` : ''}</p>
      ${msgBox(respMsgs)}</div>
      <figure><div id="wr-psi" class="wr-plot"></div><figcaption>${L('Respuesta de los tres canales contra la longitud de onda λ.', 'Response of the three channels against the wavelength λ.')}</figcaption></figure></div>`),
    osec(L('Campo plano', 'Flat field'), '<div id="wr-flat"></div>'),
    osec(L('Otra forma de calibrar', 'Another way to calibrate'), `<p class="wr-tabs" role="group" aria-label="${L('Forma de calibrar', 'Way to calibrate')}">${(['oven', 'sphere', 'file'] as const).map((k) => `<button type="button" class="wr-tab" data-calpath="${k}" aria-pressed="${S.calPath === k}">${k === 'oven' ? L('Horno', 'Furnace') : k === 'sphere' ? L('Esfera integradora', 'Integrating sphere') : L('Archivo', 'File')}</button>`).join('')}</p>
      <div id="wr-calpath"></div>
      <p>${btn('wr-cal-save', L('Guardar la calibración (.json)', 'Save the calibration (.json)'), !S.cal)}</p>`),
    osec(L('Revisar una imagen suelta', 'Check a single image'), `<p>${L('Muestra saturados y ceros de una foto antes de seguir tomando.', 'Shows saturated pixels and zeros of a photo before taking more.')}</p>
      <p class="cluster">${fileBtn('wr-q-file', L('Elegir un TIFF', 'Choose a TIFF'), '.tif,.tiff,image/tiff')} ${fileBtn('wr-q-dir', L('Elegir una carpeta', 'Choose a folder'), '', true)}</p><div id="wr-quick"></div>`),
    exOven >= 0 || exSyn >= 0 ? osec(L('Ejemplo', 'Example'), `${exOven >= 0 ? `<p>${btn(`wr-ex-${exOven}`, L('Cargar solo el horno del laboratorio', 'Load only the lab furnace'))} <span class="meta">${kB(S.o.examples[exOven].bytes)}</span></p>` : ''}
      ${exSyn >= 0 ? `<p>${btn(`wr-ex-${exSyn}`, L('Cargar la llama sintética, con su respuesta conocida', 'Load the synthetic flame, with its known answer'))} <span class="meta">${kB(S.o.examples[exSyn].bytes)}</span></p>` : ''}`) : '',
  ])}`;
  const ps = psiNow();
  const m = Math.max(...ps.psi.R);
  plot('wr-psi', { xlabel: L('longitud de onda λ (nm)', 'wavelength λ (nm)'), ylabel: L('respuesta', 'response'), height: 220 }, CH.map((c, k) => ({ kind: 'line', x: ps.lamNm, y: Array.from(ps.psi[c], (v) => v / m), cls: chCls[k], label: c })), { xlim: [380, 1000] });
  if (S.calPath === 'sphere') renderSphere(); else if (S.calPath === 'file') renderCalFile();
  renderDark(); drawPlateau(); renderFit(); renderFlat(); renderQuick();
  renderStage2();
  renderNav();
}
function fileTable(filter: (it: Item) => boolean, id: string): { html: string; msgs: Msg[]; pfs: string } {
  const list = S.items.filter(filter);
  if (!list.length) return { html: '', msgs: [], pfs: '' };
  const msgs: Msg[] = [];
  let rewritten = false;
  const rows = list.map((it) => {
    const s = it.fi;
    let state = '';
    if (it.status === 'pending') state = L('en espera', 'waiting');
    else if (it.status === 'reading') state = L('leyendo', 'reading');
    else if (it.status === 'error') { msgs.push({ code: it.err?.code ?? 'not_tiff', files: [it.path] }); state = `<em>${L('no se pudo leer', 'could not be read')}</em>`; }
    else if (s) {
      const satCh = s.satRaw.map((v, k) => (v > 0 ? CH_LETTER[k] : '')).filter(Boolean);
      state = satCh.length ? L(`saturado en ${satCh.join(', ')}`, `saturated in ${satCh.join(', ')}`) : L('leído', 'read');
      if (it.role === 'ignore') state = L('ignorado', 'ignored');
      if (s.software && !/pylon|basler/i.test(s.software)) { if (S.example) rewritten = true; else msgs.push({ code: 'rewritten', vars: { program: s.software.replace(/\0/g, '').trim() }, files: [it.path] }); }
      if (s.encoding === 'u8') msgs.push({ code: 'u8_gamma', files: [it.path] });
    }
    const readable = it.status !== 'error';
    const needT = readable && it.role === 'oven' && it.tC === undefined, needE = readable && (it.role === 'oven' || it.role === 'dark') && it.texpUs === undefined;
    if (needT || needE) msgs.push({ code: needT && needE ? 'name_unread' : needT ? 'name_unread_T' : 'name_unread_texp', vars: { file: it.path }, level: 'info' });
    const roleSel = `<select data-id="${it.id}" data-f="role" aria-label="${L('Rol de', 'Role of')} ${esc(it.path)}">${ROLES.map((r) => `<option value="${r}"${r === it.role ? ' selected' : ''}>${roleName(r)}</option>`).join('')}</select>`;
    const tIn = `<input type="number" inputmode="decimal" step="any" data-id="${it.id}" data-f="tC" value="${it.tC ?? ''}" aria-label="${L('Temperatura del horno en °C de', 'Furnace temperature in °C of')} ${esc(it.path)}"${needT ? ' aria-invalid="true" class="wr-missing"' : ''}>`;
    const eff = it.role === 'flame' && it.texpUs === undefined && fin(S.flame.texpUs) ? ` placeholder="${S.flame.texpUs}"` : '';
    const eIn = `<input type="number" inputmode="decimal" step="any" data-id="${it.id}" data-f="texpUs" value="${it.texpUs ?? ''}"${eff} aria-label="${L('Tiempo de exposición en µs de', 'Exposure time in µs of')} ${esc(it.path)}"${needE ? ' aria-invalid="true" class="wr-missing"' : ''}>`;
    return [`<span class="wr-path">${esc(it.path)}</span>`, roleSel, it.role === 'oven' ? tIn : '', eIn, s ? `${s.width} × ${s.height}` : '', state];
  });
  if (rewritten) msgs.push({ code: 'rewritten_example', level: 'info' });
  const pfsRows = [...new Map(list.filter((it) => it.pfs).map((it) => [it.path.split('/').slice(0, -1).join('/'), it.pfs!])).entries()];
  const keys = ['Gain', 'BlackLevel', 'PixelFormat', 'ExposureTime'];
  return {
    html: `<div class="wr-files-box" id="${id}">${tableHtml([L('Archivo', 'File'), L('Rol', 'Role'), L('Horno (°C)', 'Furnace (°C)'), L('Exposición (µs)', 'Exposure (µs)'), L('Tamaño', 'Size'), L('Estado', 'Status')], rows, 'wr-files')}</div>`,
    msgs,
    pfs: pfsRows.length ? tableHtml([L('Carpeta', 'Folder'), ...keys], pfsRows.map(([d, p]) => [`<span class="wr-path">${esc(d)}</span>`, ...keys.map((k) => esc(p[k] ?? '–'))])) : '',
  };
}
function seriesTabs(): string {
  return `<p class="wr-tabs" role="group" aria-label="${L('Serie del horno', 'Furnace series')}">${S.oven.series.map((s, k) => `<button type="button" class="wr-tab" data-series="${k}" aria-pressed="${k === S.oven.cur}">${tLabel(s.tC)}</button>`).join('')}</p>`;
}
function renderDark() {
  const el = document.getElementById('wr-dark');
  if (!el || !S.oven.series.length) return;
  const mode = darkModeNow();
  const rows = S.oven.series.map((s) => { const sg = s.signals[0]; return [tLabel(s.tC), num(s.texpUs[0]), CH.map((c) => num(sg.strip[c], 1)).join(' / '), pct(s.zeroFrac, 1)]; });
  el.innerHTML = `${sel('wr-darkmode', [['auto', L('automático', 'automatic')], ['intercept', L('pendiente con el tiempo de exposición', 'slope against exposure time')], ['strip', L('anillo fuera de la abertura', 'ring outside the aperture')], ['files', L('fotos oscuras', 'dark photos')]], S.darkMode, L(`Cómo se trata el oscuro, ahora ${mode === 'files' ? 'fotos oscuras' : mode === 'strip' ? 'anillo' : 'pendiente'}`, `Dark subtraction (now: ${mode === 'files' ? 'dark photos' : mode === 'strip' ? 'ring' : 'slope'}`))}
  ${tableHtml([L('Serie', 'Series'), L('Exposición más corta (µs)', 'Shortest exposure (µs)'), L('Nivel fuera de la abertura', 'Level outside the aperture') + ' R / G / B', L('Ceros', 'Zeros')], rows)}
  ${S.oven.darks.length ? tableHtml([L('Oscuro', 'Dark'), L('Exposición (µs)', 'Exposure (µs)'), L('Nivel', 'Level') + ' R / G / B', L('Ceros', 'Zeros')], S.oven.darks.map((d) => [esc(S.items.find((it) => it.id === d.id)?.path ?? d.id), num(d.texpUs), CH.map((c) => num(d.level[c], 1)).join(' / '), pct(d.zeroFrac, 1)])) : ''}`;
}
/** Preview of the flat zone with its controls, in the view with a warning and in Options otherwise. */
function plateauBlock(): string {
  const s = S.oven.series[S.oven.cur];
  if (!s) return '';
  const pl = S.oven.plateaus[String(s.tC)];
  return `${seriesTabs()}
  <div class="wr-grid2"><figure><canvas id="wr-pcv" class="wr-preview" tabindex="0" aria-label="${L('Foto del horno. Un clic pone el centro de la zona plana, las flechas la mueven y más y menos cambian el radio.', 'Furnace photo. A click sets the centre of the flat zone, the arrows move it and plus and minus change the radius.')}"></canvas>
    <figcaption>${L(`${tLabel(s.tC)}, canal ${chName(S.oven.view)}. Círculo continuo, zona plana. Punteado, abertura.`, `${tLabel(s.tC)}, ${chName(S.oven.view)} channel. Solid circle, flat zone. Dotted, aperture.`)}</figcaption></figure>
  <div class="wr-controls">
    <p class="seg" role="group" aria-label="${L('Canal de la vista previa', 'Preview channel')}">${[0, 1, 2].map((c) => `<button type="button" data-view="${c}" aria-pressed="${S.oven.view === c}">${chName(c)}</button>`).join('')}</p>
    <label class="control"><span class="control__head">${L('Centro x (píxeles)', 'Centre x (pixels)')} <output>${num(pl.cx, 1)}</output></span><input type="range" id="wr-pcx" aria-label="${L('Centro x', 'Centre x')}" min="0" max="${s.w - 1}" step="0.5" value="${pl.cx}"></label>
    <label class="control"><span class="control__head">${L('Centro y (píxeles)', 'Centre y (pixels)')} <output>${num(pl.cy, 1)}</output></span><input type="range" id="wr-pcy" aria-label="${L('Centro y', 'Centre y')}" min="0" max="${s.h - 1}" step="0.5" value="${pl.cy}"></label>
    <label class="control"><span class="control__head">${L('Radio (píxeles)', 'Radius (pixels)')} <output>${num(pl.radius, 1)}</output></span><input type="range" id="wr-pr" aria-label="${L('Radio', 'Radius')}" min="3" max="${Math.round(s.aperture.r)}" step="0.5" value="${pl.radius}"></label>
    <p>${btn('wr-preset-pl', L('Volver a la zona automática', 'Back to the automatic zone'))}</p>
    <p class="lab__readout" id="wr-pl-read" aria-live="polite"></p>
  </div></div>
  <div class="wr-msgs" id="wr-pl-msgs"></div>`;
}
function drawPlateau() {
  const s = S.oven.series[S.oven.cur];
  const cv = document.getElementById('wr-pcv') as HTMLCanvasElement | null;
  if (!s || !cv) return;
  const pl = S.oven.plateaus[String(s.tC)];
  const g = drawPreview(cv, s.preview, S.oven.view, true, 1 / Math.max(1e-6, Math.max(...Array.from({ length: 50 }, (_, i) => s.preview.v[(Math.floor(s.preview.v.length / 3 / 50) * i) * 3 + S.oven.view] || 0), 0.05)));
  const st = s.preview.step;
  circle(g, s.aperture.cx / st, s.aperture.cy / st, s.aperture.r / st, '#f0b000', true);
  circle(g, pl.cx / st, pl.cy / st, pl.radius / st, '#00d0ff', false, 2);
  const flat = s.prof?.flat ?? { R: s.plateau.flatness.R, G: s.plateau.flatness.G, B: s.plateau.flatness.B, RG: s.plateau.flatRG, BG: s.plateau.flatBG };
  const ok = Math.max(flat.R, flat.G, flat.B) <= 0.0101 && Math.max(flat.RG, flat.BG) <= 0.00505;
  const rd = document.getElementById('wr-pl-read');
  if (rd) rd.textContent = L(`Zona plana de ${num(pl.radius, 1)} píxeles. Plana dentro de ±1 %, ${ok ? 'sí' : 'no'}.`, `Flat zone of ${num(pl.radius, 1)} pixels. ${ok ? 'Flat' : 'Not flat'} to within ±1 %.`);
  const m = document.getElementById('wr-pl-msgs');
  if (m) m.innerHTML = msgHtml(plateauMsgs(s));
}
let plTimer = 0;
function movePlateau(dcx: number, dcy: number, dr: number, abs?: { cx: number; cy: number }) {
  const s = S.oven.series[S.oven.cur];
  if (!s) return;
  const pl = S.oven.plateaus[String(s.tC)];
  if (abs) { pl.cx = abs.cx; pl.cy = abs.cy; } else { pl.cx += dcx; pl.cy += dcy; pl.radius = Math.max(3, Math.min(s.aperture.r, pl.radius + dr)); }
  invalidate(1, 'oven');
  clearTimeout(plTimer);
  drawPlateau();
  plTimer = window.setTimeout(async () => {
    try { s.prof = await ask({ type: 'oven.plateau', tC: s.tC, cy: pl.cy, cx: pl.cx, radius: pl.radius }); } catch { /* keeps the proposal profile */ }
    keepFocus(() => { const st = document.getElementById('wr-st1'); if (st) st.innerHTML = subs(st1().t); });
    drawPlateau();
    for (const [id, v] of [['wr-pcx', pl.cx], ['wr-pcy', pl.cy], ['wr-pr', pl.radius]] as [string, number][]) { const i = document.getElementById(id) as HTMLInputElement | null; if (i && document.activeElement !== i) i.value = String(v); const o = i?.parentElement?.querySelector('output'); if (o) o.textContent = num(v, 1); }
  }, 180);
}
/** Temperature the camera reads at each furnace temperature with the calibration in use (K_R/K_G and ψ of S.cal). */
let readCache: { cal: unknown; res: unknown; rows: { tC: number; TK: number; Tread: number }[] } | null = null;
function bisectT(f: (T: number) => number, a = 700, b = 3000): number {
  let fa = f(a);
  const fb = f(b);
  if (!(fa * fb <= 0)) return NaN;
  for (let i = 0; i < 70; i++) { const m = (a + b) / 2, fm = f(m); if (fa * fm <= 0) b = m; else { a = m; fa = fm; } }
  return (a + b) / 2;
}
function ovenReadings(): { tC: number; TK: number; Tread: number }[] {
  const r = S.oven.result, c = S.cal;
  if (!r || !c || c.mode !== 'oven') return [];
  if (readCache && readCache.cal === c && readCache.res === r) return readCache.rows;
  const ps = psiNow();
  const band = makeBand(ps.lamNm, ps.psi, 'gl', emLaw());
  const q = c.K.R / c.K.G;
  const rows = r.fit.ratioTest.map((x) => ({ tC: x.tC, TK: x.TK, Tread: bisectT((T) => Math.log((q * bandX(band, 'R', T)) / bandX(band, 'G', T)) - Math.log(x.rMeas)) })).sort((a, b) => a.tC - b.tC);
  readCache = { cal: c, res: r, rows };
  return rows;
}
/** Result of step 1, the calibration line and its one test (DESIGN §2). Id wr-s2 kept for the tests. */
function renderStage2() {
  const el = document.getElementById('wr-s2');
  if (!el) return;
  const c = S.cal;
  if (!c) { el.innerHTML = ''; el.hidden = true; return; }
  el.hidden = false;
  const line = c.mode === 'file' ? L(`Calibración cargada desde ${esc(c.psi.name || L('un archivo', 'a file'))}.`, `Calibration loaded from ${esc(c.psi.name || 'a file')}.`)
    : c.mode === 'sphere' ? L('Calibración con la esfera integradora.', 'Calibration with the integrating sphere.')
    : (c.calibrated ?? []).length === 1 ? L('Solo un canal calibrado. La temperatura necesita dos.', 'Only one channel calibrated. Temperature needs two.')
    : c.tested === false ? L('Calibración sin probar. Carguen al menos dos temperaturas.', 'Calibration not tested. Load at least two temperatures.')
    : !c.consistent ? L('La calibración no pasa la prueba.', 'The calibration does not pass the test.')
    : c.absolute ? L('Calibración absoluta, da T y f_s en ppm.', 'Absolute calibration, gives T and f_s in ppm.') : L('Calibración relativa. T absoluta y f_s relativa.', 'Relative calibration. Absolute T and relative f_s.');
  const extra: Msg[] = [];
  if (S.flame.sensorMismatch) extra.push({ code: 'sensor_mismatch', vars: { a: `${c.sensor.width} × ${c.sensor.height}`, b: (() => { const f = flameItems()[0]; const s = sensorOf(f); return `${s.width} × ${s.height}`; })() } });
  let test = '';
  if (c.mode === 'oven' && S.oven.result && c.tested !== false) {
    const rows = c.tail?.applied ? [] : ovenReadings();
    const worst = rows.length && rows.every((x) => fin(x.Tread)) ? rows.reduce((a, x) => Math.max(a, Math.abs(x.Tread - x.TK)), 0) : NaN;
    const four = S.oven.series.length >= 4 && S.tailOn;
    const tl = c.tail?.applied && S.oven.result.tail?.ok ? S.oven.result.tail : null;
    const table = tl
      ? tableHtml([L('Temperatura dejada fuera', 'Temperature left out'), L('Error en una llama a 1800 K', 'Error in a flame at 1800 K')], tl.loo.filter((l) => tl.criterion.from.includes(l.tC)).map((l) => [tLabel(l.tC), `${sgn(l.dToutK, 1)} K`]), 'wr-ratio-t', L('Prueba, se calibra sin cada temperatura y se mira el error en la llama.', 'Test, calibrate without each temperature and look at the error in the flame.'))
      : tableHtml([L('Horno (termopar)', 'Furnace (thermocouple)'), L('Leída por la cámara', 'Read by the camera'), L('Diferencia', 'Difference')], rows.map((x) => [tLabel(x.tC), fin(x.Tread) ? tLabel(Math.round(x.Tread - T0C)) : '–', fin(x.Tread) ? `${sgn(x.Tread - x.TK)} K` : '–']), 'wr-ratio-t', L('Prueba, la cámara debe leer la temperatura del horno.', 'Test, the camera must read the furnace temperature.'));
    const w = tl ? tl.ratioTest.afterK : worst;
    test = `<div id="wr-ratio">${table}
      ${c.consistent && fin(w) ? check(true, L(`Pasa. La diferencia máxima es ${num(w)} K.`, `Passes. The largest difference is ${num(w)} K.`)) : check(false, esc(msgText({ code: four ? 'ratio_test_fail_all' : 'ratio_test_fail', vars: { a: fin(w) ? Math.round(w) : '–' } }).text), 'ratio_test_fail')}</div>`;
  } else if (c.mode === 'oven') test = '<div id="wr-ratio"></div>';
  el.innerHTML = `<p class="wr-calline"><span id="wr-st2">${subs(line)}</span></p>${test}${msgBox(extra)}`;
}
function renderFit() {
  const r = S.oven.result;
  const fe = document.getElementById('wr-fit');
  if (r && fe) {
    const f = r.fit;
    const msgs: Msg[] = [];
    if (f.calibrated.length === 1) msgs.push({ code: 'cal_one_channel', vars: { channel: CH.indexOf(f.calibrated[0]) } });
    if (!f.order.ok) msgs.push({ code: 'bayer_order' });
    const hitsBy = CH.map((c) => r.tails.filter((t) => t[c] > 0.1).map((t) => Math.round(t.tC + T0C)));
    if (hitsBy.some((h) => h.length)) msgs.push({ code: r.ratioWorst > 50 || !f.ratio.RG.tested ? 'ir_tail' : 'ir_tail_ok', vars: { a: hitsBy.every((h) => h.length === r.tails.length) ? L('los tres canales', 'the three channels') : hitsBy.map((h, k) => (h.length ? chName(k) : '')).filter(Boolean).join(L(' y ', ' and ')) }, level: 'info' });
    const tempsAll = [...new Set(f.channels.R.perT.map((p) => p.tC))];
    const rows = tempsAll.map((tC) => [tLabel(tC), ...CH.map((c) => { const p = f.channels[c].perT.find((x) => x.tC === tC); return p ? `${sig(p.K)} ± ${sig(p.sK, 2)}` : '–'; })]);
    const lin = CH.map((c) => Math.max(...f.channels[c].perT.map((p) => (fin(p.linMaxPct) ? p.linMaxPct : 0)), 0));
    for (const s of S.oven.series) CH.forEach((c, k) => { const p = f.channels[c].perT.find((x) => x.tC === s.tC); if (p && p.linMaxPct > 1) msgs.push({ code: 'nonlinear', vars: { channel: k, a: Math.round(p.linMaxPct * 10) / 10 } }); });
    const rowsLoo = f.ratioTest.map((x) => [tLabel(x.tC), tLabel(Math.round(x.Tpred - T0C)), `${sgn(x.dT)} K`]);
    const imgRows: string[][] = [];
    for (const s of S.oven.series) s.texpUs.forEach((t, i) => {
      const sg = s.signals[i];
      const idx = f.signals.findIndex((x) => x.tC === s.tC && x.texpUs === t);
      imgRows.push([tLabel(s.tC), num(t), CH.map((c) => pct(sg.S[c] / S.oven.ceil12, 0)).join(' / '), CH.map((c) => (idx >= 0 ? (f.accepted[c][idx] ? L('sí', 'yes') : L('no', 'no')) : '–')).join(' / ')]);
    });
    const blueOut = !!S.cal?.tail?.applied || !(S.cal?.calibrated ?? []).includes('B');
    fe.innerHTML = `${tableHtml([L('Temperatura', 'Temperature'), L('Sensibilidad rojo', 'Red sensitivity'), L('Sensibilidad verde', 'Green sensitivity'), L('Sensibilidad azul', 'Blue sensitivity')], rows)}
    ${f.ratio.RG.tested ? `<figure><div id="wr-kt" class="wr-plot"></div><figcaption>${L('Sensibilidad de cada canal en cada temperatura, sobre la que se usa.', 'Sensitivity of each channel at each temperature, over the one used.')}</figcaption></figure>` : ''}
    <p>${f.ratio.RG.tested ? L(`Dispersión de la razón de sensibilidades rojo/verde entre temperaturas, ${num(f.ratio.RG.dispersion * 100, 2)} %.`, `Scatter of the red/green sensitivity ratio across temperatures, ${num(f.ratio.RG.dispersion * 100, 2)} %.`) : ''}
    ${L(`Respuesta lineal con el tiempo de exposición. Desviación máxima R ${num(lin[0], 1)} %, G ${num(lin[1], 1)} %, B ${num(lin[2], 1)} %.`, `Linear response with exposure time. Largest departure R ${num(lin[0], 1)} %, G ${num(lin[1], 1)} %, B ${num(lin[2], 1)} %.`)}${blueOut ? ` ${L('El azul no se usa en T ni en f_s.', 'Blue is not used for T or f_s.').replace(/f_s/, 'f<sub>s</sub>')}` : ''}</p>
    ${rowsLoo.length ? `<h4>${S.cal?.tail?.applied ? L('Dejando fuera cada temperatura, sin corregir sobre 700 nm', 'Leaving each temperature out, without the correction above 700 nm') : L('Dejando fuera cada temperatura', 'Leaving each temperature out')}</h4>${tableHtml([L('Horno', 'Furnace'), L('Leída', 'Read'), L('Diferencia', 'Difference')], rowsLoo)}` : ''}
    <h4>${L('Imágenes del horno', 'Furnace images')}</h4>${tableHtml([L('Serie', 'Series'), L('Exposición (µs)', 'Exposure (µs)'), L('Uso de la escala', 'Use of full scale') + ' R / G / B', L('Entra', 'Used') + ' R / G / B'], imgRows)}
    ${msgBox(msgs)}`;
    const kt: Series[] = [];
    CH.forEach((c, k) => {
      const ch = f.channels[c];
      kt.push({ kind: 'points', x: ch.perT.map((p) => p.tC), y: ch.perT.map((p) => p.K / ch.K), cls: chCls[k], label: c });
      ch.perT.forEach((p) => kt.push({ kind: 'line', x: [p.tC, p.tC], y: [(p.K - (fin(p.sK) ? p.sK : 0)) / ch.K, (p.K + (fin(p.sK) ? p.sK : 0)) / ch.K], cls: chCls[k] }));
    });
    if (f.ratio.RG.tested) plot('wr-kt', { xlabel: L('horno (°C)', 'furnace (°C)'), ylabel: L('sensibilidad relativa', 'relative sensitivity'), height: 220 }, kt);
  }
  const tb = document.getElementById('wr-tail-box');
  if (r && tb) tb.innerHTML = tailBlock(r);
}
/** Correction of the camera response above 700 nm with the furnace (tailcal.ts), on by default. */
function tailBlock(r: OvenResult): string {
  const t: TailCal | null = r.tail;
  if (!t) return '';
  const intro = `${chk('wr-tail-on', S.tailOn, L('Corregir la respuesta sobre 700 nm', 'Correct the response above 700 nm'))}
  <p>${L('Ajusta la parte infrarroja de la respuesta de la cámara con el horno. Necesita tres temperaturas o más.', 'Fits the infrared part of the camera response with the furnace. It needs three temperatures or more.')}</p>`;
  if (!t.ok) return `${intro}${msgBox([{ code: 'tail_few', level: 'info' }])}`;
  const msgs: Msg[] = [];
  if (!S.tailOn) msgs.push({ code: 'tail_off', level: 'info' });
  else if (t.criterion.pass) msgs.push({ code: 'tail_ok', level: 'info' });
  else msgs.push({ code: 'tail_fail', vars: { a: num(t.criterion.worstPct, 1) } });
  if (t.temps.length === 3) msgs.push({ code: 'tail_three', level: 'info' });
  const rows = t.temps.map((tC) => {
    const l = t.loo.find((x) => x.tC === tC)!, q = t.ratio.find((x) => x.tC === tC)!;
    return [tLabel(tC), `${sgn(q.dTflameNominalK)} K`, `${sgn(l.dToutK, 1)} K`];
  });
  const range = S.o.tailRange ? (getLang() === 'es' ? S.o.tailRange.es : S.o.tailRange.en) : '';
  return `${intro}
  <p id="wr-tail-fit">${L(`La respuesta del rojo sobre 700 nm se multiplica por ${num(t.s, 3)} ± ${num(t.u.s, 3)}.`, `The red response above 700 nm is multiplied by ${num(t.s, 3)} ± ${num(t.u.s, 3)}.`)}</p>
  ${tableHtml([L('Temperatura dejada fuera', 'Temperature left out'), L('Error en una llama a 1800 K, antes', 'Error in a flame at 1800 K, before'), L('después', 'after')], rows, 'wr-tail-table')}
  <p id="wr-tail-crit">${t.criterion.pass ? L(`Pasa, peor diferencia ${num(t.ratioTest.afterK)} K.`, `Passes, worst difference ${num(t.ratioTest.afterK)} K.`) : L(`No pasa, peor diferencia ${num(t.ratioTest.afterK)} K.`, `Does not pass, worst difference ${num(t.ratioTest.afterK)} K.`)}${range && S.tailOn ? ` ${L(`Aun con la corrección, T puede quedar corrida ${range}.`, `Even with the correction, T can be off by ${range}.`)}` : ''}</p>
  ${msgBox(msgs)}`;
}
function renderFlat() {
  const el = document.getElementById('wr-flat');
  if (!el) return;
  const flats = S.items.filter((it) => it.role === 'flat');
  const v = S.flat.vignette;
  el.innerHTML = `${flats.length ? `<p>${btn('wr-flat-build', L(`Construir el mapa con ${flats.length} ${flats.length === 1 ? 'imagen' : 'imágenes'}`, `Build the map from ${flats.length} ${flats.length === 1 ? 'image' : 'images'}`))}</p>` : `<p class="meta">${L('Carguen fotos con el rol campo plano.', 'Load photos with the flat field role.')}</p>`}
  ${v ? `<div class="wr-grid2"><figure><canvas id="wr-vig" class="wr-preview"></canvas><figcaption>${L('Viñeteo del verde.', 'Green vignetting.')}</figcaption></figure><figure><div id="wr-vigp" class="wr-plot"></div><figcaption>${L('Perfil por el centro, por canal.', 'Profile through the centre, per channel.')}</figcaption></figure></div>` : ''}`;
  if (v) {
    const cv = document.getElementById('wr-vig') as HTMLCanvasElement | null;
    if (cv) drawGrey(cv, Float32Array.from(v.G, (x) => (x - 0.8) / 0.25), v.bw, v.bh, false);
    const row = Math.floor(v.bh / 2);
    plot('wr-vigp', { xlabel: L('bloque', 'block'), ylabel: 'K/K₀', height: 200 }, (['R', 'G', 'B'] as const).map((c, k) => ({ kind: 'line', x: Array.from({ length: v.bw }, (_, i) => i), y: Array.from({ length: v.bw }, (_, i) => v[c][row * v.bw + i]), cls: chCls[k] })));
  }
}
function renderSphere() {
  const el = document.getElementById('wr-calpath');
  if (!el) return;
  const imgs = S.items.filter((it) => it.role === 'sphere');
  const darks = S.items.filter((it) => it.role === 'dark' && /(esfera|sphere)/i.test(it.path));
  const r = S.sphere.result;
  const units: [string, string][] = [['radiance_nm', 'W m⁻² sr⁻¹ nm⁻¹'], ['radiance_um', 'W m⁻² sr⁻¹ µm⁻¹'], ['irradiance_nm', L('W m⁻² nm⁻¹ (irradiancia)', 'W m⁻² nm⁻¹ (irradiance)')], ['irradiance_um', L('W m⁻² µm⁻¹ (irradiancia)', 'W m⁻² µm⁻¹ (irradiance)')], ['uW_cm2_nm', 'µW cm⁻² nm⁻¹'], ['counts', L('cuentas relativas', 'relative counts')]];
  let res = '';
  if (r && r.K) {
    const K = r.K as Rgb<number>, rsd = r.rsd as Rgb<number>, near = r.near as Rgb<number>;
    const msgs: Msg[] = [];
    CH.forEach((c, k) => { if (near[c] > 0.05) msgs.push({ code: 'sphere_near_ceiling', vars: { channel: k, a: Math.round(near[c] * 1000) / 10 } }); });
    for (const c of (r.short as Ch[])) msgs.push({ code: 'sphere_short', vars: { channel: CH.indexOf(c) } });
    res = `${tableHtml([L('Canal', 'Channel'), L('Sensibilidad', 'Sensitivity'), L('Dispersión en el disco', 'Scatter in the disc')], CH.map((c) => [near[c] > 0.05 ? `${c} <em>${L('dudoso', 'doubtful')}</em>` : c, sig(K[c], 4), pct(rsd[c], 1)]))}${msgBox(msgs)}`;
  } else if (r && r.msgs) res = msgBox(r.msgs as Msg[]);
  el.innerHTML = `<p class="meta">${L(`${imgs.length} ${imgs.length === 1 ? 'imagen' : 'imágenes'} de esfera y ${darks.length} ${darks.length === 1 ? 'oscuro' : 'oscuros'}.`, `${imgs.length} sphere ${imgs.length === 1 ? 'image' : 'images'} and ${darks.length} ${darks.length === 1 ? 'dark' : 'darks'}.`)} ${S.sphere.spectrumName ? L(`Espectro ${esc(S.sphere.spectrumName)}.`, `Spectrum ${esc(S.sphere.spectrumName)}.`) : ''}</p>
  <div class="wr-fields">
    ${numIn('wr-tsph', fin(S.sphere.tSphUs) ? S.sphere.tSphUs : imgs[0]?.texpUs ?? NaN, 'min="1" step="any"', L('Exposición de la esfera (µs)', 'Sphere exposure (µs)'))}
    <div>${fileBtn('wr-spec', L('Elegir el espectro CSV', 'Choose the spectrum CSV'), '.csv,.txt,text/csv')}</div>
    ${sel('wr-sunit', units, S.sphere.valUnit, L('Unidad del espectro', 'Spectrum unit'))}
    ${sel('wr-slam', [['nm', 'nm'], ['um', 'µm'], ['A', 'Å']], S.sphere.lamUnit, L('Unidad de la longitud de onda', 'Wavelength unit'))}
    ${sel('wr-nd', [['', L('elijan una opción', 'choose an option')], ['none', L('sin filtro', 'no filter')], ['od', L('densidad óptica', 'optical density')], ['curve', L('curva CSV', 'CSV curve')]], S.sphere.nd, L('Filtro neutro', 'Neutral density filter'))}
    ${S.sphere.nd === 'od' ? numIn('wr-od', S.sphere.od, 'min="0" max="6" step="0.01"', L('Densidad óptica', 'Optical density')) : S.sphere.nd === 'curve' ? `<div>${fileBtn('wr-ndc', L('Curva de transmisión CSV', 'Transmission CSV'), '.csv,.txt')} <span class="meta">${esc(S.sphere.ndCurveName)}</span></div>` : ''}
    ${sel('wr-fill', [['median', L('mediana', 'median')], ['radial', L('radial', 'radial')]], S.sphere.fill, L('Relleno fuera del disco', 'Fill outside the disc'))}
  </div>
  <p>${btn('wr-sph-run', L('Calcular con la esfera', 'Compute with the sphere'), !imgs.length)}</p>
  ${res}`;
}
function renderCalFile() {
  const el = document.getElementById('wr-calpath');
  if (!el) return;
  const c = S.cal?.mode === 'file' ? S.cal : null;
  el.innerHTML = `<p>${L('Carguen una calibración .json guardada antes con esta herramienta.', 'Load a .json calibration saved earlier with this tool.')}</p>
  <p>${fileBtn('wr-calin', L('Elegir la calibración .json', 'Choose the calibration .json'), '.json,application/json')}</p>
  ${c ? `<p class="meta">${L('Sensibilidad', 'Sensitivity')} R, G, B = ${sig(c.K.R, 4)}, ${sig(c.K.G, 4)}, ${sig(c.K.B, 4)}</p>` : ''}
  ${msgBox(S.calFileMsgs)}`;
}

// ============================================================================ step 2 · flame images (stage 3)
function st3(): Pill {
  if (S.stale[3]) return pStale();
  const err = S.flame.msgs.find((m) => m.level === 'error');
  if (err) {
    if (err.code === 'burner_needed') return BURNER_STOP();
    if (err.code === 'scale_needed') return pStop('la escala', 'the scale');
    if (err.code === 'name_unread_flame') return pStop('el tiempo de exposición', 'the exposure time');
    return pStop('un dato', 'a value');
  }
  if (!flameItems().length) return pTodo();
  const p = S.flame.prepared;
  if (p && !S.flame.burnerOk) return { k: 'todo', t: L('Pendiente, confirmen el quemador', 'To do, confirm the burner') };
  if (p) return median(p.halvesRms) > 0.1 ? pWarn('mitades distintas', 'halves differ') : pOk(`${p.nz} alturas`, `${p.nz} heights`);
  return pTodo();
}
/** With the flame t_exp typed above, the missing name is a note that says which value is used (GB3r3-estudiante-05). */
function stage3Msgs(): Msg[] {
  if (!fin(S.flame.texpUs)) return S.flame.msgs;
  const had = S.flame.msgs.some((m) => m.code === 'name_unread_flame' && m.level !== 'error');
  const rest = S.flame.msgs.filter((m) => !(m.code === 'name_unread_flame' && m.level !== 'error'));
  return had ? [...rest, { code: 'texp_typed', vars: { a: num(S.flame.texpUs, Number.isInteger(S.flame.texpUs) ? 0 : 1) }, level: 'info' }] : rest;
}
/** Shift test of R and B on the ruler, for run.json. */
function shiftRecord() {
  const t = S.flame.shift;
  if (!t) return null;
  return { verdict: t.verdict, why: t.why, pattern: t.pattern, k: t.k, kUnc: t.kUnc, pred: t.pred, meas: t.meas, unc: t.unc, axes: t.axes, nEdge: t.nEdge, tiles: t.tiles };
}
/** Readout of the shift test of R and B on the ruler, in Options of step 2. */
function shiftHtml(): string {
  const t = S.flame.shift;
  if (!t) return `<p class="lab__readout" id="wr-shift">${L('No se pudo leer la regla para la prueba.', 'The ruler could not be read for the test.')}</p>`;
  if (t.verdict === 'not_applicable') return `<p class="lab__readout" id="wr-shift" data-verdict="${t.verdict}">${L('Esta imagen no necesita centrar los canales.', 'This image does not need the channel centring.')}</p>`;
  const verdict = t.verdict === 'activate' ? L('La regla dice que conviene centrar los canales rojo y azul.', 'The ruler says the red and blue channels should be centred.')
    : t.verdict === 'keep_off' ? L('La regla dice que no conviene centrar los canales rojo y azul.', 'The ruler says the red and blue channels should not be centred.')
    : L('La regla no decide. Dejen el centrado apagado.', 'The ruler does not decide. Leave the centring off.');
  return `<p class="lab__readout" id="wr-shift" data-verdict="${t.verdict}">${verdict} ${L(`Medido ${num(t.k, 2)} ± ${num(2 * t.kUnc, 2)} veces el corrimiento supuesto.`, `Measured ${num(t.k, 2)} ± ${num(2 * t.kUnc, 2)} times the assumed shift.`)}</p>`;
}
const STEP2_VIEW = [...FILE_CODES, 'burner_needed', 'upside_down', 'scale_needed', 'tilted', 'tilted_1', 'tilt_warn', 'tilt_warn_1', 'tilt_warn_one', 'axis_spread', 'axis_edge', 'axis_ambiguous', 'axis_low_signal', 'axis_none', 'crop_outside', 'edge_signal', 'edge_signal_file', 'frames_mixed_size', 'frames_mixed_size_1', 'sensor_mismatch', 'method_mismatch', 'encoding_mismatch', 'transposed', 'orient_first', 'bg_texp', 'dark_texp', 'name_unread_flame', 'need_stage', 'no_frames', 'scale_none', 'odd_width', 'pfs_differs'];
function renderStage3() {
  const el = document.getElementById('wr-s3');
  if (!el) return;
  const fl = flameItems();
  const p = S.flame.prepared;
  const timed = fl.some((it) => it.name.timeS !== undefined);
  const nb = S.items.filter((it) => it.role === 'background').length, nru = S.items.filter((it) => it.role === 'ruler').length;
  const ft = fileTable((it) => it.role === 'flame' || it.role === 'background' || it.role === 'ruler', 'wr-ft3');
  const linFail = !!S.oven.result && CH.some((c) => S.oven.result!.fit.channels[c].perT.some((x) => x.linMaxPct > 1));
  const all = [...loadMsgsFor(3), ...stage3Msgs().filter((m) => m.code !== 'rows_saturated').map((m) => (m.code === 'texp_outside_cal' && !linFail ? { ...m, level: 'info' as const } : m))];
  const { v, o } = splitMsgs(all, [...STEP2_VIEW, ...(linFail ? ['texp_outside_cal'] : [])]);
  const near = (codes: string[]) => v.filter((m) => codes.includes(m.code));
  const rest = v.filter((m) => !['burner_needed', 'upside_down', 'scale_needed'].includes(m.code));
  const frames = fl.map((it) => `<li class="wr-frame${it.include ? '' : ' wr-off'}"><label><input type="checkbox" data-incl="${it.id}"${it.include ? ' checked' : ''}> <canvas data-thumb="${it.id}" width="1" height="1"></canvas><span class="wr-path">${esc(it.path.split('/').pop()!)}</span>${it.include ? '' : `<span class="meta">${it.why === 'tilted' ? L('fuera, inclinado', 'out, tilted') : it.why === 'axis' ? L('fuera, eje corrido', 'out, axis off') : it.why === 'average' ? L('fuera, en el promedio', 'out, in the average') : it.why && it.why !== 'excluded' ? esc(it.why) : L('fuera', 'out')}</span>`}</label></li>`).join('');
  const halves = p ? median(p.halvesRms) : NaN;
  el.innerHTML = `${stepHead(2, L('Imágenes de la llama', 'Flame images'), 'wr-st3', st3(), L('Prepara las fotos de la llama con su tiempo de exposición, su escala y el quemador abajo.', 'Sets up the flame photos with their exposure time, their scale and the burner at the bottom.'))}
  <div class="wr-drop">
    <h3 class="wr-droplab">${L('Fotos de la llama, del fondo y de la regla', 'Flame, background and ruler photos')}</h3>
    <p class="cluster">${fileBtn('wr-f-dir', L('Elegir carpeta', 'Choose folder'), '', true)} ${fileBtn('wr-f-files', L('Elegir archivos o .zip', 'Choose files or .zip'), '.tif,.tiff,.zip,.json')}</p>
  </div>
  ${fl.length ? `<p class="wr-read" id="wr-f-read">${L(`${fl.length} ${fl.length === 1 ? 'cuadro' : 'cuadros'} de llama · ${nb} fondo · ${nru} regla`, `${fl.length} flame ${fl.length === 1 ? 'frame' : 'frames'} · ${nb} background · ${nru} ruler`)}</p>
  <div class="wr-inputs">
    <div class="wr-pair">${numIn('wr-texp', S.flame.texpUs, 'min="1" step="any"', L('Tiempo de exposición (µs)', 'Exposure time (µs)'))}${chk('wr-texpok', S.flame.texpConfirmed, L('Confirmo que es el de la toma', 'This is the exposure of the photos'))}</div>
    <div class="wr-pair">${numIn('wr-mmpx', S.flame.mmPerPx, 'min="0" step="any"', L('Escala (mm por píxel)', 'Scale (mm per pixel)'))}${chk('wr-scaleok', S.flame.scaleConfirmed, L('Confirmo la escala', 'Scale confirmed'))}</div>
    ${msgBox(near(['scale_needed']))}
    ${S.flame.rulerPreview ? `<div class="wr-ruler-box"><p class="meta">${L('Hagan clic en dos marcas de la regla.', 'Click two marks on the ruler.')}</p>
      <canvas id="wr-rcv" class="wr-preview wr-ruler" tabindex="0" aria-label="${L('Foto de la regla. Hagan clic en dos marcas.', 'Ruler photo. Click two marks.')}"></canvas>
      <div class="wr-row">${numIn('wr-rmm', S.flame.rulerMm, 'min="0" step="any"', L('Distancia entre las marcas (mm)', 'Distance between the marks (mm)'))} ${btn('wr-rclear', L('Borrar las marcas', 'Clear the marks'))}</div></div>` : ''}
    <div class="wr-pair">${numIn('wr-zb', S.flame.zBurner, 'step="any"', L('Altura de la fila de abajo sobre el quemador (mm)', 'Height of the bottom row above the burner (mm)'))}
    ${sel('wr-quarter', [['0', L('como se cargó', 'as loaded')], ['1', L('90° a la derecha', '90° clockwise')], ['3', L('90° a la izquierda', '90° counter-clockwise')], ['2', '180°']], String(S.flame.quarter), L('Girar', 'Rotate'))}</div>
  </div>` : ''}
  ${msgBox(rest)}
  ${pbtn('wr-prep', L('Preparar las imágenes', 'Prepare the images'), !fl.length, L('Primero carguen las fotos de la llama.', 'Load the flame photos first.'))}
  ${p ? `<div class="wr-result wr-grid2"><figure><canvas id="wr-fcv" class="wr-preview"></canvas><figcaption>${L('Imagen orientada. Línea continua, eje.', 'Oriented image. Solid line, axis.')}</figcaption></figure>
    <div>${chk('wr-burnerok', S.flame.burnerOk, L('El quemador queda abajo', 'The burner is at the bottom'))}
    ${msgBox(near(['burner_needed', 'upside_down']))}
    <p>${L(`${p.nz} alturas de ${num(p.zMm[0], 0)} a ${num(p.zMm[p.nz - 1], 0)} mm, ${p.nr} radios.`, `${p.nz} heights from ${num(p.zMm[0], 0)} to ${num(p.zMm[p.nz - 1], 0)} mm, ${p.nr} radii.`)}</p>
    ${halves > 0.1 ? check(false, L(`Las dos mitades difieren ${num(100 * halves, 0)} %. Revisen el giro, o elijan una mitad en Opciones del paso 3.`, `The two halves differ by ${num(100 * halves, 0)} %. Check the rotation, or pick one half in step 3 Options.`), 'halves') : check(true, L(`Las dos mitades se parecen, diferencia ${num(100 * halves, 0)} %.`, `The two halves look alike, ${num(100 * halves, 0)} % difference.`))}
    <p>${btn('wr-reco', L('Calcular los pasos 3, 4 y 5 de una vez', 'Compute steps 3, 4 and 5 at once'))}</p></div></div>` : ''}
  ${optsBlock(2, [
    msgBox(o),
    fl.length ? osec(L('Cuadros', 'Frames'), `${timed ? `<div class="wr-row">${sel('wr-times', [['repeat', L('repeticiones, se promedian', 'repeats, averaged')], ['instants', L('instantes distintos, se usa uno', 'different instants, one is used')]], S.flame.times, L('Tiempos en los nombres', 'Times in the names'))}${S.flame.times === 'instants' ? sel('wr-instant', fl.map((it) => [it.id, it.path.split('/').pop()!] as [string, string]), S.flame.instant, L('Instante', 'Instant')) : ''}</div>` : ''}<ul class="wr-frames">${frames}</ul>`) : '',
    osec(L('Archivos', 'Files'), ft.html + ft.pfs),
    fl.length ? osec(L('Geometría', 'Geometry'), `<div class="wr-fields">
      ${sel('wr-sensq', [['0', L('sin girar', 'no rotation')], ['1', L('90° a la derecha', '90° clockwise')], ['3', L('90° a la izquierda', '90° counter-clockwise')], ['2', '180°']], String(S.flame.sensorQ), L('Posición del sensor', 'Sensor orientation'))}
      ${chk('wr-mirror', S.flame.mirror, L('Espejo izquierda derecha', 'Left-right mirror'))}
      ${numIn('wr-fine', S.flame.fineDeg, 'min="-10" max="10" step="0.1"', L('Rotación fina (°)', 'Fine rotation (°)'))}
      ${chk('wr-axisauto', S.flame.axisAuto, L('Eje automático', 'Automatic axis'))}
      ${!S.flame.axisAuto ? numIn('wr-axis', S.flame.axisManual, 'step="0.05"', L('Eje (columna)', 'Axis (column)')) : ''}
      ${numIn('wr-hw', S.flame.halfWidth, 'min="4" step="1"', L('Semiancho de búsqueda del eje (píxeles)', 'Axis search half-width (pixels)'))}
      ${numIn('wr-nr', S.flame.nr, 'min="4" step="1"', L('Radios por lado', 'Radii per side'))}
      ${numIn('wr-rtop', S.flame.rowTop, 'min="0" step="1"', L('Primera fila', 'First row'))}
      ${numIn('wr-rbot', S.flame.rowBottom, 'min="0" step="1"', L('Última fila', 'Last row'))}
      ${numIn('wr-rph', S.flame.rowsPerHeight, 'min="1" max="32" step="1"', L('Filas por altura', 'Rows per height'))}
      ${numIn('wr-burner', S.flame.burnerRow, 'min="0" step="1"', L('Fila del quemador', 'Burner row'))}
      ${numIn('wr-sunc', S.flame.scaleUnc, 'min="0" max="20" step="0.1"', L('Incertidumbre de la escala (%)', 'Scale uncertainty (%)'))}</div>`) : '',
    osec(L('Cámara', 'Camera'), `<div class="wr-fields">
      ${sel('wr-bin', [['2x2', L('2 × 2 (por defecto)', '2 × 2 (default)')], ['1x1', L('sin agrupar', 'no grouping')]], S.group ? '2x2' : '1x1', L('Agrupar píxeles', 'Group pixels'))}
      ${chk('wr-centre', S.centre, L('Centrar los canales rojo y azul', 'Centre the red and blue channels'))}</div>
      ${S.flame.rulerPreview ? shiftHtml() : ''}`),
    p ? osec(L('Mitades', 'Halves'), `${heightSlider('wr-h3b')}<figure><div id="wr-halves" class="wr-plot"></div><figcaption>${L('Mitad derecha continua y mitad izquierda punteada, a la altura elegida.', 'Right half solid and left half dotted, at the chosen height.')}</figcaption></figure>`) : '',
    p ? osec(L('Descarga', 'Download'), `<p>${btn('wr-npz', L('Descargar las imágenes preparadas (.npz)', 'Download the prepared images (.npz)'))}</p>`) : '',
  ])}`;
  for (const it of fl) {
    const c = el.querySelector(`canvas[data-thumb="${it.id}"]`) as HTMLCanvasElement | null;
    if (c && it.thumb) drawThumb(c, it.thumb, S.flame.quarter, S.flame.mirror);
  }
  drawRuler();
  if (p) drawFlame();
  renderNav();
}
/** Frame thumbnail with the rotation chosen, like the preview (GB3r3-estudiante-18). */
function drawThumb(c: HTMLCanvasElement, th: NonNullable<Item['thumb']>, quarter: number, mirror: boolean) {
  const src = document.createElement('canvas');
  src.width = th.w; src.height = th.h;
  const sg = src.getContext('2d')!; const im = sg.createImageData(th.w, th.h); im.data.set(th.rgba); sg.putImageData(im, 0, 0);
  const q = ((quarter % 4) + 4) % 4, swap = q % 2 === 1;
  c.width = swap ? th.h : th.w; c.height = swap ? th.w : th.h;
  const g = c.getContext('2d')!;
  g.translate(c.width / 2, c.height / 2); g.rotate((q * Math.PI) / 2); if (mirror) g.scale(-1, 1);
  g.drawImage(src, -th.w / 2, -th.h / 2);
}
function median(a: number[]) { const b = a.filter(fin).sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)] : NaN; }
function heightSlider(id: string): string {
  const p = S.flame.prepared!;
  return `<label class="control"><span class="control__head">${L('Altura', 'Height')} <output>${num(p.zMm[S.flame.height], 1)} mm</output></span><input type="range" id="${id}" class="wr-height" aria-label="${L('Altura', 'Height')}" min="0" max="${p.nz - 1}" step="1" value="${S.flame.height}"></label>`;
}
function drawRuler() {
  const cv = document.getElementById('wr-rcv') as HTMLCanvasElement | null;
  const pv = S.flame.rulerPreview;
  if (!cv || !pv) return;
  // the ruler photo was taken with the same turned camera, so it is shown with the rotation of the flame
  const src = document.createElement('canvas');
  drawPreview(src, pv, 3, false, 1.2);
  const V = rulerView(pv);
  cv.width = V.W; cv.height = V.H;
  const g = cv.getContext('2d')!;
  g.save(); g.translate(V.W / 2, V.H / 2); g.rotate((V.q * Math.PI) / 2); if (S.flame.mirror) g.scale(-1, 1);
  g.drawImage(src, -pv.w / 2, -pv.h / 2); g.restore();
  const pts = S.flame.rulerPts.map(([x, y]) => V.fwd(x / pv.step, y / pv.step));
  for (const [u, v] of pts) circle(g, u, v, 4, '#00d0ff', false, 2);
  if (pts.length === 2) line(g, pts[0][0], pts[0][1], pts[1][0], pts[1][1], '#00d0ff');
}
/**
 * Display of the ruler preview with the flame rotation (S.flame.quarter, S.flame.mirror), as drawThumb does.
 * fwd maps a preview pixel of the file to the canvas, inv maps a canvas pixel back. The marks are kept in file
 * pixels, so the distance between them, and the scale, do not depend on the rotation.
 */
function rulerView(pv: Planes) {
  const q = ((S.flame.quarter % 4) + 4) % 4, m = S.flame.mirror, swap = q % 2 === 1;
  const W = swap ? pv.h : pv.w, H = swap ? pv.w : pv.h;
  const c = [1, 0, -1, 0][q], s = [0, 1, 0, -1][q];
  const fwd = (x: number, y: number): [number, number] => {
    const px = (m ? -1 : 1) * (x - pv.w / 2), py = y - pv.h / 2;
    return [W / 2 + c * px - s * py, H / 2 + s * px + c * py];
  };
  const inv = (u: number, v: number): [number, number] => {
    const a = u - W / 2, b = v - H / 2;
    return [pv.w / 2 + (m ? -1 : 1) * (c * a + s * b), pv.h / 2 - s * a + c * b];
  };
  return { q, W, H, fwd, inv };
}
function drawFlame() {
  const p = S.flame.prepared;
  const cv = document.getElementById('wr-fcv') as HTMLCanvasElement | null;
  if (!p || !cv) return;
  const pv = p.preview;
  const g = drawGrey(cv, pv.v, pv.w, pv.h);
  const st = pv.step;
  line(g, p.axis / st, 0, p.axis / st, pv.h, '#00d0ff', false, 1.5);
  line(g, (p.axis - p.nr) / st, 0, (p.axis - p.nr) / st, pv.h, '#f0b000', true);
  line(g, (p.axis + p.nr) / st, 0, (p.axis + p.nr) / st, pv.h, '#f0b000', true);
  line(g, 0, p.rows[0] / st, pv.w, p.rows[0] / st, '#f0b000', true);
  line(g, 0, p.rows[1] / st, pv.w, p.rows[1] / st, '#f0b000', true);
  const k = S.flame.height;
  const nb = Math.max(1, S.flame.rowsPerHeight);
  const yk = p.rows[1] - (k + 0.5) * nb;
  line(g, 0, yk / st, pv.w, yk / st, '#ff4060', false, 1);
  const ser: Series[] = [];
  for (let c = 0; c < 3; c++) {
    ser.push({ kind: 'line', x: p.rMm, y: p.right[c][k], cls: chCls[c] });
    ser.push({ kind: 'line', x: p.rMm, y: p.left[c][k], cls: chCls[c], dash: true });
  }
  plot('wr-halves', { xlabel: 'r (mm)', ylabel: L('imagen', 'image'), height: 220 }, ser);
}

// ============================================================================ step 3 · Abel inversion (stage 4)
const methodLong = (m: string) => ({ onion: L('pelado de cebolla sin suavizado', 'onion peeling without smoothing'), tikhonov: L('pelado de cebolla con Tikhonov', 'onion peeling with Tikhonov'), sat: 'SAT', annabel: L('red neuronal', 'neural network') } as Record<string, string>)[m] ?? m;
function abelMsgs(): Msg[] {
  const p = S.flame.prepared, o = S.abel.out;
  const msgs: Msg[] = [...S.abel.annabel.msgs];
  if (o && o.overReg && p) {
    const atCorner = S.abel.outMethod === 'tikhonov' && !!S.abel.lcurve && Math.abs(Math.log10(o.lambdaRel / S.abel.lcurve.lambda)) < 0.03;
    const rows = o.overRows ?? [];
    const zs = rows.map((k) => p.zMm[k]);
    const v = { n: rows.length || o.overReg, a: zs.length ? num(Math.min(...zs), 1) : '–', b: zs.length ? num(Math.max(...zs), 1) : '–' };
    msgs.push(atCorner ? { code: 'over_regularised_corner', vars: v, level: 'info' } : { code: 'over_regularised', vars: v });
  }
  const lc = S.abel.lcurve;
  if (lc && S.abel.method === 'tikhonov' && Math.abs(Math.log10(S.abel.lambdaRel / lc.lambda)) < 0.03 && (lc.index <= 1 || lc.index >= lc.lambdas.length - 2)) msgs.push({ code: 'lcurve_edge', vars: { a: sig(lc.lambda, 2) } });
  if (o && o.excluded.length && o.excluded.length !== p?.satRows.length) msgs.push({ code: 'rows_saturated', vars: { n: o.excluded.length } });
  else if (p && p.satRows.length) msgs.push({ code: 'rows_saturated', vars: { n: p.satRows.length } });
  return msgs;
}
const STEP3_VIEW = ['over_regularised', 'over_regularised_1', 'lcurve_edge', 'rows_saturated', 'rows_saturated_1', 'sat_not_spd', 'tikhonov_not_spd', 'lambda_out_of_range', 'webgpu_lost', 'worker_failed', 'need_stage', 'no_projections'];
function st4(): Pill {
  if (S.stale[4]) return pStale();
  if (!S.flame.prepared) return pTodo();
  if (S.abel.annabel.running) return pBusy(L('Pendiente, la red neuronal está entrenando', 'To do, the neural network is training'));
  if (!S.abel.out) return pTodo();
  if (splitMsgs(abelMsgs(), STEP3_VIEW).v.length) return pWarn();
  return pOk();
}
function renderStage4() {
  const el = document.getElementById('wr-s4');
  if (!el) return;
  const p = S.flame.prepared, o = S.abel.out, A = S.abel.annabel;
  const lr = Math.log10(S.abel.lambdaRel);
  const { v, o: om } = splitMsgs(abelMsgs(), STEP3_VIEW);
  const m = S.abel.method;
  const smooth = m === 'tikhonov' ? ` · ${S.abel.lambdaAuto ? L('Suavizado automático', 'Automatic smoothing') : S.abel.lamFlat ? L(`Suavizado ${sig(S.abel.lambdaRel, 2)} fijo, la curva L no tiene esquina`, `Smoothing ${sig(S.abel.lambdaRel, 2)} fixed, the L-curve has no corner`) : L(`Suavizado ${sig(S.abel.lambdaRel, 2)}, elegido a mano`, `Smoothing ${sig(S.abel.lambdaRel, 2)}, set by hand`)}` : '';
  const symOpts: [string, string][] = m === 'sat' || m === 'annabel' ? [['both', L('las dos mitades en el ajuste', 'both halves in the fit')], ['average', L('las dos mitades promediadas', 'both halves averaged')], ['right', L('solo la derecha', 'right only')], ['left', L('solo la izquierda', 'left only')]] : [['average', L('las dos mitades promediadas', 'both halves averaged')], ['right', L('solo la derecha', 'right only')], ['left', L('solo la izquierda', 'left only')]];
  el.innerHTML = `${stepHead(3, L('Inversión de Abel', 'Abel inversion'), 'wr-st4', st4(), L('Pasa de la luz sumada en la línea de visión a la emisión en cada radio.', 'Goes from the light summed along the line of sight to the emission at each radius.'))}
  ${p ? `<p class="wr-method">${L('Método', 'Method')} ${methodLong(m)}${smooth}</p>` : ''}
  ${pbtn('wr-abel', m === 'annabel' ? L('Entrenar la red neuronal', 'Train the neural network') : L('Invertir', 'Invert'), !p || !S.flame.burnerOk || A.running, !p ? L('Primero preparen las imágenes en el paso 2.', 'Prepare the images in step 2 first.') : !S.flame.burnerOk ? L('Primero confirmen el quemador en el paso 2.', 'Confirm the burner in step 2 first.') : '', A.running ? ` ${btn('wr-ann-stop', L('Detener', 'Stop'))}` : '')}
  ${m === 'annabel' && (A.running || A.tick) ? '<p class="meta" id="wr-ann-prog" aria-live="polite"></p>' : ''}
  ${msgBox(v)}
  ${o && p ? `<div class="wr-result"><figure><div id="wr-reproj" class="wr-plot"></div><figcaption>${L('Imagen medida (fina) y reconstruida (gruesa) a la altura elegida.', 'Measured image (thin) and reconstructed image (thick) at the chosen height.')}</figcaption></figure>
    ${heightSlider('wr-h4')}
    ${v.some((x) => x.code.startsWith('over_regularised')) ? '' : check(true, L('La reconstrucción sigue a la imagen medida. Lo que queda es ruido.', 'The reconstruction follows the measured image. What is left is noise.'))}</div>` : ''}
  ${optsBlock(3, [
    msgBox(om),
    m === 'tikhonov' ? osec(L('Suavizado', 'Smoothing'), `<label class="control"><span class="control__head">${L('Suavizado', 'Smoothing')} <output>${sig(S.abel.lambdaRel, 2)}</output></span><input type="range" id="wr-lam" aria-label="${L('Suavizado', 'Smoothing')}" min="${Math.log10(LAMBDA_REL_MIN)}" max="${Math.log10(LAMBDA_REL_MAX)}" step="0.05" value="${lr}"></label>
      ${chk('wr-lam-auto', S.abel.lambdaAuto, L('Automático con la curva L', 'Automatic with the L-curve'))}
      <p>${L('Cuánto se alisa el perfil al invertir. Más suavizado quita ruido cerca del eje y borra detalles finos. Menos suavizado deja más detalle y más ruido.', 'How much the profile is smoothed when inverting. More smoothing removes noise near the axis and erases fine detail. Less smoothing keeps more detail and more noise.')}</p>
      <p class="cluster">${btn('wr-lc', L('Calcular la curva L', 'Compute the L-curve'), !p)}${S.abel.lcurve ? ` ${btn('wr-lc-use', L(`Volver al automático, ${sig(S.abel.lcurve.lambda, 2)}`, `Back to automatic, ${sig(S.abel.lcurve.lambda, 2)}`))}` : ''}</p>
      ${S.abel.lcurve ? `<figure><div id="wr-lcp" class="wr-plot"></div><figcaption>${L('Cada punto es un suavizado. El punto naranjo es la esquina, el equilibrio que usa el modo automático.', 'Each point is one smoothing value. The orange point is the corner, the balance the automatic mode uses.')}</figcaption></figure>` : ''}`) : '',
    osec(L('Método', 'Method'), `<p class="seg" role="group" aria-label="${L('Método', 'Method')}">${(['tikhonov', 'onion', 'sat', 'annabel'] as const).map((x) => `<button type="button" data-method="${x}" aria-pressed="${m === x}">${methodLong(x)}</button>`).join('')}</p>
      ${m === 'sat' ? `<div class="wr-row">${numIn('wr-seg', S.abel.segments, 'min="3" max="60" step="1"', L('Segmentos', 'Segments'))}${numIn('wr-satl', S.abel.satLambda, 'min="0" step="any"', L('Suavizado SAT', 'SAT smoothing'))}</div>` : ''}
      ${m === 'annabel' ? annabelPanel() : ''}`),
    osec(L('Mitad', 'Half'), sel('wr-sym', symOpts, m === 'annabel' ? 'both' : S.abel.sym === 'both' && m !== 'sat' ? 'average' : S.abel.sym, L('Usar', 'Use'))),
    o && p ? osec(L('Gráficos', 'Plots'), `<div class="wr-grid2"><figure><div id="wr-resid" class="wr-plot"></div><figcaption>${L('Diferencia entre la imagen medida y la reconstruida, en el verde. Punteada, diferencia entre mitades.', 'Difference between the measured and the reconstructed image, on green. Dotted, difference between halves.')}</figcaption></figure>
      <figure><div id="wr-jr" class="wr-plot"></div><figcaption>${L('Emisión por radio a la altura elegida.', 'Emission per radius at the chosen height.')}</figcaption></figure></div>
      <figure class="wr-maps3">${[0, 1, 2].map((c) => `<div><canvas id="wr-jmap-${c}" class="wr-map"></canvas><span class="meta">${chName(c)}</span></div>`).join('')}<figcaption>${L('Emisión de cada canal en toda la llama.', 'Emission of each channel over the whole flame.')}</figcaption></figure>
      <p class="lab__readout" id="wr-abel-read" aria-live="polite"></p>`) : '',
    p ? osec(L('Comparar métodos', 'Compare methods'), `<p>${L('Pelado de cebolla, Tikhonov y SAT a la misma altura, en el verde.', 'Onion peeling, Tikhonov and SAT at the same height, on green.')} ${btn('wr-cmp', L('Comparar a esta altura', 'Compare at this height'))}</p>
      ${S.abel.compare ? `<figure><div id="wr-cmpp" class="wr-plot"></div><figcaption>${L('Gris, pelado de cebolla. Continua, Tikhonov. Punteada, SAT.', 'Grey, onion peeling. Solid, Tikhonov. Dotted, SAT.')}</figcaption></figure>` : ''}`, 'wr-compare') : '',
  ])}`;
  drawAbel();
  renderAnnabelProgress();
  renderNav();
}
/** A duration in words, "menos de N min" (GB3r2-adv-18). */
function minText(sec: number): string {
  const m = sec / 60;
  if (!(m >= 1)) return L('menos de 1 min', 'under 1 min');
  return L(`menos de ${num(Math.ceil(m))} min`, `under ${num(Math.ceil(m))} min`);
}
function annabelPanel(): string {
  const A = S.abel.annabel;
  const est = A.estimate ? L(`${minText(A.estimate.sPerIter * A.iterations)} para ${num(A.iterations)} iteraciones.`, `${minText(A.estimate.sPerIter * A.iterations)} for ${num(A.iterations)} iterations.`) : '';
  return `<p>${L('Entrena una red neuronal en este navegador.', 'Trains a neural network in this browser.')}${S.gpu === false ? ` ${L('Este navegador no tiene tarjeta gráfica disponible y será más lento.', 'This browser has no graphics card available and will be slower.')}` : ''}</p>
  <div class="wr-row">${numIn('wr-ann-it', A.iterations, 'min="100" max="100000" step="100"', L('Iteraciones', 'Iterations'))}${numIn('wr-seed', S.res.seed, 'min="0" step="1"', L('Semilla', 'Seed'))}</div>
  <p>${btn('wr-ann-est', L('Estimar el tiempo', 'Estimate the time'), !S.flame.prepared || A.running)}</p>
  <p class="meta" id="wr-ann-est-t">${est}</p>`;
}
function renderAnnabelProgress() {
  const e = document.getElementById('wr-ann-prog');
  const t = S.abel.annabel.tick;
  if (e && t) e.textContent = L(`Iteración ${num(t.iter)} de ${num(t.total)}${t.etaS !== null ? `, faltan unos ${num(t.etaS / 60, 1)} min` : ''}.`, `Iteration ${num(t.iter)} of ${num(t.total)}${t.etaS !== null ? `, about ${num(t.etaS / 60, 1)} min left` : ''}.`);
}
function drawAbel() {
  const p = S.flame.prepared, o = S.abel.out;
  const lc = S.abel.lcurve;
  if (lc && document.getElementById('wr-lcp')) {
    plot('wr-lcp', { xlabel: L('diferencia con la imagen (log)', 'difference with the image (log)'), ylabel: L('rugosidad del perfil (log)', 'profile roughness (log)'), height: 220 }, [
      { kind: 'line', x: lc.rho.map(Math.log), y: lc.eta.map(Math.log), cls: 's-data' },
      { kind: 'points', x: [Math.log(lc.rho[lc.index])], y: [Math.log(lc.eta[lc.index])], cls: 's-warn' },
    ]);
  }
  if (!p || !o) return;
  const k = S.flame.height;
  const ser: Series[] = [];
  for (let c = 0; c < 3; c++) {
    ser.push({ kind: 'line', x: p.rMm, y: p.right[c][k], cls: `${chCls[c]} wr-thin` });
    ser.push({ kind: 'line', x: p.rMm, y: p.left[c][k], cls: `${chCls[c]} wr-thin`, dash: true });
    ser.push({ kind: 'line', x: p.rMm, y: o.reproj[c][k], cls: `${chCls[c]} wr-thick` });
  }
  plot('wr-reproj', { xlabel: 'r (mm)', ylabel: L('imagen', 'image'), height: 220 }, ser);
  const diff = Array.from(p.right[1][k], (v, i) => v - p.left[1][k][i]);
  plot('wr-resid', { xlabel: 'r (mm)', ylabel: L('diferencia', 'difference'), height: 200 }, [
    { kind: 'line', x: p.rMm, y: o.resid[1][k], cls: 's-chG wr-thick' }, { kind: 'line', x: p.rMm, y: diff, cls: 's-muted', dash: true },
    { kind: 'line', x: [0, p.rMm[p.nr - 1]], y: [0, 0], cls: 's-muted' }]);
  plot('wr-jr', { xlabel: 'r (mm)', ylabel: L('emisión (W m⁻³ sr⁻¹)', 'emission (W m⁻³ sr⁻¹)'), height: 220 }, [0, 1, 2].map((c) => ({ kind: 'line', x: p.rMm, y: o.J[c][k], cls: chCls[c] })));
  for (let c = 0; c < 3; c++) {
    const cv = document.getElementById(`wr-jmap-${c}`) as HTMLCanvasElement | null;
    if (!cv) continue;
    const f = new Float64Array(p.nz * p.nr);
    let mx = 0;
    for (let kk = 0; kk < p.nz; kk++) for (let i = 0; i < p.nr; i++) { const v = o.J[c][kk][i]; f[kk * p.nr + i] = v; if (fin(v) && v > mx) mx = v; }
    drawMap(cv, f, p.nz, p.nr, (['ch_R', 'ch_G', 'ch_B'] as const)[c], 0, mx || 1);
  }
  const r = o.resid[1][k];
  const n3 = Math.max(1, Math.floor(p.nr / 5));
  const rms = (a: ArrayLike<number>, i0: number, i1: number) => { let s = 0, n = 0; for (let i = i0; i < i1; i++) if (fin(a[i])) { s += a[i] ** 2; n++; } return n ? Math.sqrt(s / n) : NaN; };
  const ax = rms(r, 0, n3), ed = rms(r, p.nr - n3, p.nr);
  const rd = document.getElementById('wr-abel-read');
  if (rd) rd.textContent = fin(ax) && fin(ed) && ed > 0 ? L(`A ${num(p.zMm[k], 1)} mm la diferencia cerca del eje es ${num(ax / ed, 1)} veces la del borde.`, `At ${num(p.zMm[k], 1)} mm the difference near the axis is ${num(ax / ed, 1)} times the one at the edge.`) : '';
  const cmp = S.abel.compare;
  if (cmp && document.getElementById('wr-cmpp')) plot('wr-cmpp', { xlabel: 'r (mm)', ylabel: L('emisión del verde', 'green emission'), height: 220 }, [
    { kind: 'line', x: p.rMm, y: cmp.onion, cls: 's-muted' }, { kind: 'line', x: p.rMm, y: cmp.tikhonov, cls: 's-chG' }, { kind: 'line', x: p.rMm, y: cmp.sat, cls: 's-chG', dash: true }]);
}

// ============================================================================ step 4 · T and f_s (stage 5)
function tfsMsgs(): Msg[] {
  const t = S.tfs.out, p = S.flame.prepared;
  const d = calDecision();
  const msgs: Msg[] = [];
  if (d.state === 'uncalibrated' && S.abel.out) msgs.push({ code: 'uncalibrated' });
  if (t) {
    if (t.ambiguous) msgs.push({ code: 'ratio_ambiguous', vars: { n: t.ambiguous } });
    if (t.outside) msgs.push({ code: 'ratio_outside', vars: { n: t.outside }, level: 'info' });
    let neg = 0, edge = 0;
    for (const f of t.flags) { if (f & 1) neg++; if (f & 2) edge++; }
    if (neg) msgs.push({ code: 'fs_negative', vars: { n: neg } });
    if (edge) msgs.push({ code: 'T_at_grid_edge', vars: { n: edge }, level: 'info' });
    if (t.chem.some((x) => x)) msgs.push({ code: 'chemilum', level: 'info' });
    if (d.state === 'uncalibrated') msgs.push({ code: 'chem_off_uncal', level: 'info' });
    const Ts = usedSorted(t.T), fsv = fsShown();
    const T95 = pctOf(Ts, 0.95), F95 = fsv && fsv.unit === 'ppm' ? pctOf(usedSorted(fsv.v), 0.95) : NaN;
    if (fsv && p) {
      const Fu = usedSorted(fsv.v).filter((x) => x > 0), f95 = pctOf(Fu, 0.95);
      let fmax = -Infinity, over = 0;
      for (let q = 0; q < t.mask.length; q++) { if (t.mask[q] !== 1 || !fin(fsv.v[q])) continue; if (fsv.v[q] > 3 * f95) over++; if (fsv.v[q] > fmax) fmax = fsv.v[q]; }
      if (fin(f95) && f95 > 0 && (fmax > 5 * f95 || over > 0.01 * Fu.length)) msgs.push({ code: 'fs_outliers' });
    }
    if ((fin(T95) && T95 > 2400) || (fin(F95) && F95 > 50)) msgs.push(fin(F95) ? { code: 'implausible', vars: { a: num(T95), b: sig(F95, 2) } } : { code: 'implausible_T', vars: { a: num(T95) } });
    const dd = S.stale[6] && S.tfs.dec ? S.tfs.dec : d;
    if (dd.provisional && dd.fs !== 'no') msgs.push({ code: !S.flame.texpConfirmed || !S.flame.scaleConfirmed ? 'fs_provisional' : 'fs_provisional_cal' });
  }
  if (S.tfs.path === 'net') {
    msgs.push(...S.tfs.net.msgs);
    const r = S.tfs.net.result;
    if (t && r?.xcheck && !r.xcheck.agrees) msgs.push({ code: 'net_xcheck_bad', vars: { a: xcheckBad(r.xcheck) } });
    if (t && r?.bounds?.warnT) msgs.push({ code: 'net_bounds_T', level: 'info' });
    if (t && r?.bounds?.warnFs && d.fs === 'ppm') msgs.push({ code: 'net_bounds_fs', level: 'info' });
  }
  return msgs;
}
const STEP4_VIEW = ['uncalibrated', 'ratio_ambiguous', 'fs_negative', 'fs_outliers', 'implausible', 'implausible_T', 'fs_provisional', 'fs_provisional_cal', 'net_xcheck_bad', 'net_two_channels', 'net_no_signal', 'webgpu_lost', 'worker_failed', 'no_emission'];
function st5(): Pill {
  if (S.stale[5]) return pStale();
  if (!S.abel.out) return pTodo();
  if (S.tfs.net.running) return pBusy(L('Pendiente, la red neuronal está entrenando', 'To do, the neural network is training'));
  if (!S.tfs.out) return pTodo();
  const v = splitMsgs(tfsMsgs(), STEP4_VIEW).v;
  if (v.some((m) => m.code === 'fs_provisional' || m.code === 'fs_provisional_cal') && v.length === 1) return pWarn('f_s provisional', 'f_s provisional');
  if (v.length) return pWarn();
  return pOk();
}
function renderStage5() {
  const el = document.getElementById('wr-s5');
  if (!el) return;
  const t = S.tfs.out, p = S.flame.prepared;
  const { v, o } = splitMsgs(tfsMsgs(), STEP4_VIEW);
  const N = S.tfs.net;
  let result = '';
  if (t && p) {
    const T = usedSorted(t.T), fs = fsShown();
    const F = fs ? usedSorted(fs.v) : [];
    const fsLine = !fs ? L('Sin f_s con esta calibración.', 'No f_s with this calibration.') : fs.unit === 'ppm' ? L(`f_s mediana ${sig(pctOf(F, 0.5), 2)} ppm, percentil 95 de ${sig(pctOf(F, 0.95), 2)} ppm.`, `Median f_s ${sig(pctOf(F, 0.5), 2)} ppm, 95th percentile ${sig(pctOf(F, 0.95), 2)} ppm.`) : L('f_s relativa, sin ppm con esta calibración.', 'Relative f_s, no ppm with this calibration.');
    result = `<div class="wr-result"><p class="wr-big">${L(`T mediana ${num(pctOf(T, 0.5))} K, entre ${num(pctOf(T, 0.05))} y ${num(pctOf(T, 0.95))} K.`, `Median T ${num(pctOf(T, 0.5))} K, between ${num(pctOf(T, 0.05))} and ${num(pctOf(T, 0.95))} K.`)}<br>${subs(fsLine)}</p>
    ${v.length ? '' : check(true, L('Valores dentro de lo esperado para una llama con hollín.', 'Values within what is expected for a sooting flame.'))}</div>`;
  }
  const isNet = S.tfs.path === 'net';
  el.innerHTML = `${stepHead(4, L('T y f_s', 'T and f_s'), 'wr-st5', st5(), L('Calcula la temperatura T con la razón rojo/verde y la fracción de volumen de hollín f_s con la intensidad.', 'Computes the temperature T from the red/green ratio and the soot volume fraction f_s from the intensity.'))}
  ${pbtn('wr-tfs', isNet ? L('Calcular T y f_s con la red neuronal', 'Compute T and f_s with the neural network') : L('Calcular T y f_s', 'Compute T and f_s'), !S.abel.out || N.running, !S.abel.out ? L('Primero inviertan en el paso 3.', 'Invert in step 3 first.') : '', N.running ? ` ${btn('wr-net-stop', L('Detener', 'Stop'))}` : '')}
  ${isNet && (N.running || N.tick) ? `<progress id="wr-net-bar" class="wr-progress" max="${N.tick ? N.tick.total : N.iterations}" value="${N.tick ? N.tick.iter : 0}" aria-label="${L('Avance del entrenamiento', 'Training progress')}"></progress><p class="meta" id="wr-net-prog" aria-live="polite">${N.tick ? netProgressText() : ''}</p>` : ''}
  ${msgBox(v)}
  ${result}
  ${optsBlock(4, [
    msgBox(o),
    osec(L('Camino', 'Path'), `${sel('wr-path', [['ratio', L('razón rojo/verde', 'red/green ratio')], ['fit3', L('ajuste con los tres canales', 'fit with the three channels')], ['net', L('red neuronal', 'neural network')]], S.tfs.path, L('Cómo se calculan T y f_s', 'How T and f_s are computed').replace(/f_s/, 'f<sub>s</sub>'))}
      ${isNet ? netPanel() : ''}
      ${isNet && t && N.result ? `<p class="lab__readout" id="wr-net-xc">${netReading(N.result)}</p>` : ''}
      ${t && isNet && N.result ? `<figure class="wr-maps3 wr-netmaps"><div><canvas id="wr-net-tcf" class="wr-map"></canvas><span class="meta">${L('T directa', 'direct T')}</span></div><div><canvas id="wr-net-dt" class="wr-map"></canvas><span class="meta">${L('diferencia', 'difference')}</span></div><figcaption>${L(`T del cálculo directo y su diferencia con la red, de 0 a ${num(netDiff().max)} K.`, `T of the direct calculation and its difference with the network, from 0 to ${num(netDiff().max)} K.`)}</figcaption></figure>` : ''}`),
    osec(L('Canales', 'Channels'), `<div class="wr-fields">${S.tfs.path === 'ratio' ? sel('wr-pair', [['RG', L('rojo/verde', 'red/green')], ['BG', L('azul/verde', 'blue/green')], ['BR', L('azul/rojo', 'blue/red')]], S.tfs.pair, L('Par de canales para T', 'Channel pair for T')) : ''}
      ${S.tfs.path === 'ratio' ? sel('wr-fsch', [['RG', L('rojo y verde', 'red and green')], ['R', L('solo rojo', 'red only')], ['G', L('solo verde', 'green only')], ['RGB', L('rojo, verde y azul', 'red, green and blue')], ['B', L('solo azul', 'blue only')]], S.tfs.fsChannel, L('Canales para f<sub>s</sub>', 'Channels for f<sub>s</sub>')) : ''}</div>
      <p>${L('El azul se usa solo con una calibración que lo valide, por ejemplo con la esfera integradora.', 'Blue is used only with a calibration that validates it, for example with the integrating sphere.')}</p>`),
    osec(L('Ley de E(m)', 'E(m) law'), `${sel('wr-em', [['group', L('de referencia', 'reference')], ['coeffs', L('a + b ln²λ + c lnλ/λ, λ en µm', 'a + b ln²λ + c lnλ/λ, λ in µm')], ['constant', L('constante', 'constant')]], S.tfs.em, 'E(m)')}
      ${S.tfs.em === 'coeffs' ? `<div class="wr-row">${numIn('wr-ema', S.tfs.a, 'step="any"', 'a')}${numIn('wr-emb', S.tfs.b, 'step="any"', 'b')}${numIn('wr-emc', S.tfs.c, 'step="any"', 'c')}</div>` : ''}
      ${S.tfs.em === 'constant' ? numIn('wr-eme', S.tfs.E, 'min="0.01" max="1" step="0.01"', 'E') : ''}`),
    osec(L('Máscaras', 'Masks'), `<div class="wr-fields">
      ${numIn('wr-sig', S.tfs.sigFrac * 100, 'min="0" max="50" step="0.5"', L('Señal mínima (% del máximo)', 'Minimum signal (% of the maximum)'))}
      ${numIn('wr-island', S.tfs.islandFrac * 100, 'min="0" max="50" step="1"', L('Islas sueltas (% del área mayor)', 'Small isolated regions (% of the largest area)'))}
      ${numIn('wr-zcut', S.tfs.zCut, 'step="any"', L('Quitar la quimioluminiscencia bajo la altura (mm)', 'Remove chemiluminescence below the height (mm)'))}
      ${numIn('wr-chem', S.tfs.chemThr * 100, 'min="0" max="200" step="1"', L('Umbral de quimioluminiscencia (%)', 'Chemiluminescence threshold (%)'))}</div>`),
    osec(L('Gráficos', 'Plots'), `<figure><div id="wr-tab" class="wr-plot"></div><figcaption>${L('Tabla de temperatura contra la razón de canales.', 'Table of temperature against the channel ratio.')}</figcaption></figure>
      ${t && S.tfs.path === 'fit3' ? `<figure><canvas id="wr-chi" class="wr-map"></canvas><figcaption>${L('Calidad del ajuste con tres canales.', 'Quality of the three-channel fit.')}</figcaption></figure>` : ''}
      ${t && p ? `<p class="lab__readout">${losReading(t, p)}</p>` : ''}`),
  ])}`;
  drawTable();
  const nr = N.result;
  if (t && p && isNet && nr) {
    const a = document.getElementById('wr-net-tcf') as HTMLCanvasElement | null, b = document.getElementById('wr-net-dt') as HTMLCanvasElement | null;
    const { d, max } = netDiff();
    const ext = mapExtent(p);
    if (a) { drawMap(a, nr.Tcf, p.nz, p.nr, 't', S.res.Tmin, S.res.Tmax); a.style.aspectRatio = `${ext.aspect}`; }
    if (b) { drawMap(b, d, p.nz, p.nr, 'fs', 0, max); b.style.aspectRatio = `${ext.aspect}`; }
  }
  if (t && S.tfs.path === 'fit3') {
    const cv = document.getElementById('wr-chi') as HTMLCanvasElement | null;
    const vv = Array.from(t.chi2rel).filter(fin).sort((x, y) => x - y);
    if (cv && p) drawMap(cv, t.chi2rel, p.nz, p.nr, 'fs', 0, vv.length ? vv[Math.floor(0.99 * (vv.length - 1))] : 1);
  }
  renderNav();
}
/** |T direct − T network| per pixel and the top of its scale, the 95th percentile and never under the warning limit. */
function netDiff(): { d: Float64Array; max: number } {
  const t = S.tfs.out, nr = S.tfs.net.result;
  if (!t || !nr) return { d: new Float64Array(0), max: XCHECK_WARN_P90_K };
  const d = Float64Array.from(t.T, (x, q) => (fin(x) && fin(nr.Tcf[q]) ? Math.abs(x - nr.Tcf[q]) : NaN));
  const v = Array.from(d).filter(fin).sort((a, b) => a - b);
  const p95 = v.length ? v[Math.floor(0.95 * (v.length - 1))] : 0;
  return { d, max: Math.max(XCHECK_WARN_P90_K, Math.ceil(p95 / 10) * 10) };
}
/** Panel of the neural network option, iterations and estimated time. */
function netPanel(): string {
  const N = S.tfs.net, e = N.estimate;
  const est = e ? L(`${minText(e.sPerIter * N.iterations)} para ${num(N.iterations)} iteraciones.`, `${minText(e.sPerIter * N.iterations)} for ${num(N.iterations)} iterations.`) : '';
  return `<p>${L('Entrena una red neuronal en este navegador.', 'Trains a neural network in this browser.')}${S.gpu === false ? ` ${L('Este navegador no tiene tarjeta gráfica disponible y será más lento.', 'This browser has no graphics card available and will be slower.')}` : ''}</p>
  ${numIn('wr-net-it', N.iterations, 'min="10" max="100000" step="100"', L('Iteraciones', 'Iterations'))}
  <p>${btn('wr-net-est', L('Estimar el tiempo', 'Estimate the time'), !S.abel.out || N.running)}</p>
  <p class="meta" id="wr-net-est-t">${est}</p>`;
}
function netProgressText(): string {
  const t = S.tfs.net.tick;
  if (!t) return '';
  return L(`Iteración ${num(t.iter)} de ${num(t.total)}${t.etaS !== null && t.iter < t.total ? `, faltan unos ${num(Math.max(t.etaS, 0), 0)} s` : ''}.`, `Iteration ${num(t.iter)} of ${num(t.total)}${t.etaS !== null && t.iter < t.total ? `, about ${num(Math.max(t.etaS, 0), 0)} s left` : ''}.`);
}
function renderNetProgress() {
  const t = S.tfs.net.tick;
  const bar = document.getElementById('wr-net-bar') as HTMLProgressElement | null;
  if (bar && t) { bar.hidden = false; bar.max = t.total; bar.value = t.iter; }
  const e = document.getElementById('wr-net-prog');
  if (e) e.textContent = netProgressText();
}
function xcheckBad(x: XCheck): string {
  const parts: string[] = [];
  if (x.bad.includes('T_median')) parts.push(L(`T difiere ${num(x.TMedianAbsDiffK)} K en la mediana`, `T differs by ${num(x.TMedianAbsDiffK)} K in the median`));
  if (x.bad.includes('T_p90')) parts.push(L(`${num(x.TP90AbsDiffK)} K en el percentil 90`, `${num(x.TP90AbsDiffK)} K at the 90th percentile`));
  if (x.bad.includes('fs')) parts.push(L(`f_s difiere en un factor ${num(Math.exp(x.fsMedianAbsLogRatio), 2)}`, `f_s differs by a factor ${num(Math.exp(x.fsMedianAbsLogRatio), 2)}`));
  return parts.join(L(' y ', ' and ')) + '.';
}
/** The comparison of the network with the direct calculation, after training. */
function netReading(r: NetOut): string {
  const tr = L(`La red se entrenó ${num(r.completed)} de ${num(r.planned)} iteraciones en ${num(r.elapsedS, 1)} s. Mejor resultado en la iteración ${num(r.bestIter)}.`, `The network trained ${num(r.completed)} of ${num(r.planned)} iterations in ${num(r.elapsedS, 1)} s. Best result at iteration ${num(r.bestIter)}.`);
  const x = r.xcheck;
  if (!x) return `${tr} ${L('Ningún píxel sirve para comparar con el cálculo directo.', 'No pixel is suitable for comparing with the direct calculation.')}`;
  return `${tr} ${L(`Comparada con el cálculo directo en ${num(x.nPixels)} píxeles, T difiere ${num(x.TMedianAbsDiffK)} K en la mediana y ${num(x.TP90AbsDiffK)} K en el percentil 90. ${x.agrees ? 'Coinciden.' : ''}`, `Compared with the direct calculation on ${num(x.nPixels)} pixels, T differs by ${num(x.TMedianAbsDiffK)} K in the median and ${num(x.TP90AbsDiffK)} K at the 90th percentile. ${x.agrees ? 'They agree.' : ''}`)}`;
}
function losReading(t: TfsOut, p: Prepared): string {
  let a = 0, b = 0, n = 0;
  for (let q = 0; q < t.T.length; q++) if (t.mask[q] === 1 && fin(t.T[q]) && fin(t.Tlos[q])) { a += t.T[q]; b += t.Tlos[q]; n++; }
  void p;
  if (!n) return '';
  const d = (b - a) / n;
  return d >= 0 ? L(`La razón sin invertir da en promedio ${num(d)} K más.`, `The ratio without inversion gives ${num(d)} K more on average.`) : L(`La razón sin invertir da en promedio ${num(-d)} K menos.`, `The ratio without inversion gives ${num(-d)} K less on average.`);
}
function drawTable() {
  const ps = psiNow();
  const band = makeBand(ps.lamNm, ps.psi, 'gl', emLaw());
  const Tg = Float64Array.from({ length: 161 }, (_, k) => 1000 + 10 * k);
  const tab = ratioTable(band, Tg);
  const r = S.tfs.pair === 'RG' ? tab.RG : S.tfs.pair === 'BG' ? tab.BG : tab.BR;
  plot('wr-tab', { xlabel: 'T (K)', ylabel: S.tfs.pair.split('').join('/'), height: 220 }, [{ kind: 'line', x: Tg, y: r, cls: 's-data' }]);
}

// ============================================================================ step 5 · results (stage 6)
/** Comparison with the ground truth of the synthetic flame, the metric of comparar_con_verdad.py (DESIGN §6). */
function truthLine(): string {
  const tr = S.truth, t = S.tfs.out, p = S.flame.prepared;
  if (!tr || !t || !p) return '';
  const { zg, rp, T, F, nr } = tr;
  const ss = (g: Float64Array, x: number) => { let lo = 0, hi = g.length; while (lo < hi) { const m = (lo + hi) >> 1; if (g[m] < x) lo = m + 1; else hi = m; } return Math.min(Math.max(lo - 1, 0), g.length - 2); };
  const at = (V: Float64Array, z: number, r: number) => {
    const iz = ss(zg, z), ir = ss(rp, r);
    const tz = (z - zg[iz]) / (zg[iz + 1] - zg[iz]), tq = (r - rp[ir]) / (rp[ir + 1] - rp[ir]);
    const v = (a: number, b: number) => V[a * (nr + 1) + b];
    return (1 - tz) * (1 - tq) * v(iz, ir) + tz * (1 - tq) * v(iz + 1, ir) + (1 - tz) * tq * v(iz, ir + 1) + tz * tq * v(iz + 1, ir + 1);
  };
  const fs = fsShown();
  let sT = 0, nT = 0, fmax = 0;
  const Ft = new Float64Array(p.nz * p.nr);
  for (let k = 0; k < p.nz; k++) for (let i = 0; i < p.nr; i++) {
    const q = k * p.nr + i;
    Ft[q] = Math.max(0, at(F, p.zMm[k], p.rMm[i]));
    if (Ft[q] > fmax) fmax = Ft[q];
    if ((t.mask[q] & 1) === 1 && fin(t.T[q])) { const d = t.T[q] - at(T, p.zMm[k], p.rMm[i]); sT += d * d; nT++; }
  }
  if (!nT) return '';
  const eT = Math.sqrt(sT / nT);
  let eF = NaN;
  if (fs && fs.unit === 'ppm') {
    let s = 0, n = 0;
    for (let q = 0; q < Ft.length; q++) if ((t.mask[q] & 1) === 1 && Ft[q] > 0.05 * fmax && fin(fs.v[q])) { const d = (fs.v[q] / Ft[q] - 1) * 100; s += d * d; n++; }
    if (n) eF = Math.sqrt(s / n);
  }
  return `<p class="wr-truth" id="wr-truth">${fin(eF) ? L(`Llama sintética. La T se aleja ${num(eT)} K de la verdadera y f_s un ${num(eF)} %.`, `Synthetic flame. T is ${num(eT)} K off the true value and f_s is ${num(eF)} % off.`) : L(`Llama sintética. La T se aleja ${num(eT)} K de la verdadera.`, `Synthetic flame. T is ${num(eT)} K off the true value.`)}</p>`.replace(/f_s/g, 'f<sub>s</sub>');
}
function st6(): Pill {
  if (S.stale[6]) return pStale();
  if (!(S.tfs.out && S.flame.prepared && S.abel.out)) return pTodo();
  const x = S.tfs.path === 'net' ? S.tfs.net.result?.xcheck : null;
  if (x && !x.agrees) return pWarn('la red no coincide, vean el paso 4', 'the network does not match, see step 4');
  return S.res.band ? pOk() : pTodo();
}
function renderStage6() {
  const el = document.getElementById('wr-s6');
  if (!el) return;
  const t = S.tfs.out, p = S.flame.prepared;
  const fs = fsShown();
  const ok = !!(t && p && S.abel.out);
  const band = !!S.res.band;
  const runMsgs = S.loadMsgs.filter((m) => /^run_/.test(m.code));
  const eqHref = location.protocol === 'file:' ? new URL('../equations/#budget', S.o.pageUrl).href : '../equations/#budget';
  el.innerHTML = `${stepHead(5, L('Resultados', 'Results'), 'wr-st6', st6(), L('Mapas y perfiles de T y f_s, con su banda de ruido, para el informe.', 'Maps and profiles of T and f_s, with their noise band, for the report.'))}
  ${ok && p ? `<div class="wr-maps">
    <figure><div class="wr-mapax" id="wr-map-T-ax"><canvas id="wr-map-T" class="wr-map"></canvas></div><canvas id="wr-bar-T" class="wr-cbar"></canvas><p class="wr-barlab"><span>${num(S.res.Tmin)}</span><span>T (K)</span><span>${num(S.res.Tmax)}</span></p><figcaption>${L(`T(r, z) en K. Altura sobre el quemador ${num(p.zMm[0], 0)} a ${num(p.zMm[p.nz - 1], 0)} mm.`, `T(r, z) in K. Height above the burner ${num(p.zMm[0], 0)} to ${num(p.zMm[p.nz - 1], 0)} mm.`)}</figcaption></figure>
    ${fs ? `<figure><div class="wr-mapax" id="wr-map-fs-ax"><canvas id="wr-map-fs" class="wr-map"></canvas></div><canvas id="wr-bar-fs" class="wr-cbar"></canvas><p class="wr-barlab"><span>0</span><span>f<sub>s</sub> (${fs.unit})</span><span>${sig(S.res.fsMax, 2)}</span></p><figcaption>${fs.unit === 'ppm' ? L('f<sub>s</sub>(r, z) en ppm.', 'f<sub>s</sub>(r, z) in ppm.') : L('f<sub>s</sub>(r, z) relativa.', 'Relative f<sub>s</sub>(r, z).')}</figcaption></figure>` : ''}
  </div>
  <div class="wr-row">${[0, 1, 2].map((i) => numIn(`wr-hz${i}`, fin(S.res.heights[i]) ? +p.zMm[S.res.heights[i]].toFixed(1) : NaN, 'step="any"', i === 0 ? L('Alturas de los perfiles (mm)', 'Profile heights (mm)') : `<span class="sr-only">${L(`Altura ${i + 1} (mm)`, `Height ${i + 1} (mm)`)}</span>`)).join('')}</div>` : ''}
  ${pbtn('wr-band', L('Calcular perfiles con banda de ruido', 'Compute profiles with noise band'), !ok, L('Primero calculen T y f_s en el paso 4.', 'Compute T and f_s in step 4 first.').replace(/f_s/, 'f<sub>s</sub>')).replace('wr-btn--primary', band ? 'wr-btn--box' : 'wr-btn--primary')}
  ${ok ? `<div class="wr-grid2"><figure><div id="wr-profT" class="wr-plot"></div><figcaption>${L('T(r) a las alturas elegidas.', 'T(r) at the chosen heights.')}</figcaption></figure>
    ${fs ? `<figure><div id="wr-profF" class="wr-plot"></div><figcaption>${L('f<sub>s</sub>(r) a las mismas alturas.', 'f<sub>s</sub>(r) at the same heights.')}</figcaption></figure>` : ''}</div>
    ${band ? check(true, L(`Banda calculada con ${S.res.nBand} repeticiones.`, `Band computed with ${S.res.nBand} repetitions.`)) : ''}
    ${truthLine()}
    <p class="wr-act"><button class="wr-btn ${band ? 'wr-btn--primary' : 'wr-btn--box'}" type="button" id="wr-dl-zip">${L('Descargar todo (.zip)', 'Download all (.zip)')}</button> ${btn('wr-dl-csv', 'CSV')} ${btn('wr-dl-png-T', L('PNG de T', 'T PNG'))} ${btn('wr-dl-png-fs', L('PNG de f<sub>s</sub>', 'f<sub>s</sub> PNG'), !fs)}</p>
    <p class="meta">${L('CSV, PNG y registro para el informe.', 'CSV, PNGs and log for the report.')}</p>` : ''}
  ${optsBlock(5, [
    ok ? osec(L('Escala de los mapas', 'Map scale'), `<div class="wr-row">${numIn('wr-tmin', S.res.Tmin, 'step="10"', L('T mínima (K)', 'Minimum T (K)'))}${numIn('wr-tmax', S.res.Tmax, 'step="10"', L('T máxima (K)', 'Maximum T (K)'))}${fs ? numIn('wr-fsmax', +S.res.fsMax.toPrecision(3), 'min="0" step="any"', L(`f<sub>s</sub> máxima (${fs.unit})`, `Maximum f<sub>s</sub> (${fs.unit})`)) : ''}</div>`) : '',
    osec(L('Banda', 'Band'), `<div class="wr-row">${numIn('wr-nband', S.res.nBand, 'min="5" max="200" step="1"', L('Repeticiones', 'Repetitions'))}${numIn('wr-seed6', S.res.seed, 'min="0" step="1"', L('Semilla', 'Seed'))}</div>`),
    ok ? osec(L('Mapa de línea de visión', 'Line-of-sight map'), `<figure><div class="wr-mapax" id="wr-map-los-ax"><canvas id="wr-map-los" class="wr-map"></canvas></div><figcaption>${L('T sin invertir. Muestra lo que se leería sin la inversión de Abel.', 'T without inversion. It shows what would be read without the Abel inversion.')}</figcaption></figure>`) : '',
    osec(L('Más descargas', 'More downloads'), `<p class="cluster">${getLang() === 'es' ? `${btn('wr-dl-csvpt', 'CSV con punto decimal', !ok)} ` : ''}${btn('wr-dl-run', L('Registro del cálculo (.json)', 'Log of the calculation (.json)'), !S.items.length)}</p>
      ${ok ? `<p class="cluster wr-npy">${L('Mapas sueltos (.npy)', 'Single maps (.npy)')} ${Object.keys(npyFields()).map((k) => `<button class="wr-btn wr-btn--quiet" type="button" data-npy="${k}">${k}</button>`).join(' ')}</p>` : ''}`),
    osec(L('Retomar un cálculo guardado', 'Resume a saved calculation'), `<p>${L('Rearma los ajustes de un registro .json y comprueba que los archivos sean los mismos.', 'Restores the settings of a .json log and checks that the files are the same.')}</p><p>${fileBtn('wr-run-in', L('Elegir el registro .json', 'Choose the .json log'), '.json,application/json')}</p>${msgBox(runMsgs)}`),
    osec(L('Límites del método', 'Limits of the method'), `<p>${L('Hay errores que la banda no incluye.', 'There are errors the band does not include.')} <a data-eq>${L('Están en Ecuaciones.', 'They are in Equations.')}</a></p>`),
  ])}`;
  const eq = el.querySelector('a[data-eq]') as HTMLAnchorElement | null;
  if (eq) eq.href = eqHref;
  if (ok) drawResults();
  renderNav();
}

function drawResults() {
  const t = S.tfs.out!, p = S.flame.prepared!;
  const fs = fsShown();
  const T = document.getElementById('wr-map-T') as HTMLCanvasElement | null;
  if (T) drawMap(T, t.T, p.nz, p.nr, 't', S.res.Tmin, S.res.Tmax);
  let out = 0;
  for (let q = 0; q < t.T.length; q++) if (t.mask[q] === 1 && fin(t.T[q]) && (t.T[q] < S.res.Tmin || t.T[q] > S.res.Tmax)) out++;
  S.res.Tout = out;
  const bt = document.getElementById('wr-bar-T') as HTMLCanvasElement | null; if (bt) drawBar(bt, 't');
  const los = document.getElementById('wr-map-los') as HTMLCanvasElement | null;
  if (los) drawMap(los, t.Tlos, p.nz, p.nr, 't', S.res.Tmin, S.res.Tmax);
  if (fs) {
    const F = document.getElementById('wr-map-fs') as HTMLCanvasElement | null;
    if (F) drawMap(F, fs.v, p.nz, p.nr, 'fs', 0, S.res.fsMax);
    const bf = document.getElementById('wr-bar-fs') as HTMLCanvasElement | null; if (bf) drawBar(bf, 'fs');
  }
  const ext = mapExtent(p);
  for (const id of ['wr-map-T', 'wr-map-fs', 'wr-map-los']) {
    const c = document.getElementById(id) as HTMLCanvasElement | null;
    if (c) c.style.aspectRatio = `${ext.aspect}`;
    const box = document.getElementById(`${id}-ax`);
    // a tall narrow flame stays under 30rem high, a wide one under 15rem wide
    if (box) { box.style.width = `min(100% - 3.5rem, 15rem, ${(30 * ext.aspect).toFixed(3)}rem)`; mapAxes(box, ext, tickFmt); }
  }
  const b = S.res.band;
  const sT: Series[] = [], sF: Series[] = [];
  const dash = [false, true, false];
  S.res.heights.forEach((k, j) => {
    if (!fin(k) || k < 0 || k >= p.nz) return;
    const sl = (a: ArrayLike<number>) => Array.from({ length: p.nr }, (_, i) => a[k * p.nr + i]);
    const cls = ['s-data', 's-physics', 's-learned'][j] ?? 's-data';
    if (b) sT.push({ kind: 'band', x: p.rMm, y: sl(b.Tlo), y2: sl(b.Thi), cls: 's-muted' });
    sT.push({ kind: 'line', x: p.rMm, y: sl(t.T), cls, dash: dash[j], label: `${num(p.zMm[k], 1)} mm` });
    if (fs) {
      if (b) sF.push({ kind: 'band', x: p.rMm, y: sl(b.fslo).map((v) => v * fs.factor), y2: sl(b.fshi).map((v) => v * fs.factor), cls: 's-muted' });
      sF.push({ kind: 'line', x: p.rMm, y: sl(fs.v), cls, dash: dash[j], label: `${num(p.zMm[k], 1)} mm` });
    }
  });
  plot('wr-profT', { xlabel: 'r (mm)', ylabel: 'T (K)', height: 240, legend: true }, sT);
  if (fs) plot('wr-profF', { xlabel: 'r (mm)', ylabel: `fₛ (${fs.unit})`, height: 240, legend: true }, sF, fin(S.res.fsMax) && S.res.fsMax > 0 ? { ylim: [0, 2 * S.res.fsMax] } : {});
}

// ============================================================================ events
function onInput(e: Event) {
  const t = e.target as HTMLInputElement | HTMLSelectElement;
  const id = t.id, v = (t as HTMLInputElement).value, n = parseFloat(String(v).replace(',', '.'));
  const val = fin(n) ? n : NaN;
  if (t.dataset.incl) {
    // stage 3 frame list, include or exclude one frame (GB3 fix, the checkbox had no handler)
    const it = S.items.find((x) => x.id === t.dataset.incl);
    const on = (t as HTMLInputElement).checked;
    if (!it || it.include === on) return;
    it.include = on; it.why = on ? '' : 'excluded';
    invalidate(3, 'flame');
    renderAll();
    return;
  }
  if (t.dataset.id && t.dataset.f) {
    const it = S.items.find((x) => x.id === t.dataset.id);
    if (!it) return;
    if (t.dataset.f === 'role') {
      // act on change only, so the arrow keys can walk the list of roles (GB3r1-web-01)
      if (e.type !== 'change') return;
      const was = it.role;
      it.role = t.value as Role;
      const calRole = (r: Role) => r === 'oven' || r === 'dark' || r === 'flat' || r === 'sphere';
      harmonise().then(() => {
        // a flame, background or ruler role does not touch the calibration (GB3r1-fisico-18)
        if (calRole(was) || calRole(it.role)) invalidate(1, 'files');
        if (['flame', 'background', 'ruler'].includes(was) || ['flame', 'background', 'ruler'].includes(it.role)) invalidate(3, 'flame');
        renderAll();
        if (it.role === 'oven' || it.role === 'dark' || was === 'oven' || was === 'dark') analyseOven();
        if (it.role === 'flame' || was === 'flame') loadFlame();
        if (it.role === 'ruler') { S.flame.rulerId = it.id; loadRuler(); }
        else if (was === 'ruler' && S.flame.rulerId === it.id) {
          // the ruler was set to another role, so the next ruler in the table takes its place (or none)
          const next = S.items.find((x) => x.role === 'ruler' && x.status === 'ok');
          S.flame.rulerId = next?.id ?? ''; S.flame.rulerPts = [];
          if (next) loadRuler(); else { S.flame.rulerPreview = null; S.flame.shift = null; renderStage3(); }
        }
      });
      return;
    }
    if (t.dataset.f === 'tC') { it.tC = fin(val) ? val : undefined; it.tFrom = 'typed'; }
    if (t.dataset.f === 'texpUs') { it.texpUs = fin(val) ? val : undefined; it.texpFrom = 'typed'; if (it.role === 'flame' && fin(val)) S.flame.texpUs = val; }
    if (e.type === 'change') { if (['oven', 'dark', 'flat', 'sphere'].includes(it.role)) invalidate(1, 'files'); else invalidate(3, 'flame'); renderSoon(); if (it.role === 'oven' || it.role === 'dark') analyseOven(); }
    return;
  }
  const map: Record<string, () => void> = {
    'wr-preset': () => { S.presetId = v as Preset['id']; S.presetManual = true; S.custom = null; invalidate(1, 'psi'); },
    'wr-basis': () => { S.basis = v as 'photon' | 'energy'; invalidate(1, 'psi'); },
    'wr-sp': () => { S.sp = v === 'sp700'; invalidate(1, 'psi'); },
    'wr-ir': () => { S.irOn = (t as HTMLInputElement).checked; invalidate(1, 'psi'); },
    'wr-ircut': () => { if (fin(val)) { S.irCutNm = val; invalidate(1, 'psi'); } },
    'wr-irw': () => { if (fin(val)) { S.irWidthNm = val; invalidate(1, 'psi'); } },
    'wr-dT': () => { if (fin(val)) { S.sigmaT = Math.min(50, Math.max(0, val)); invalidate(1, 'oven'); } },
    'wr-eps': () => { S.eps = val; const o = document.getElementById('wr-eps-out'); if (o) o.textContent = num(val, 2); invalidate(1, 'oven'); },
    'wr-tail-on': () => { S.tailOn = (t as HTMLInputElement).checked; if (S.oven.result) setCalFromOven(); renderAll(); },
    'wr-known': () => { S.known = (t as HTMLInputElement).checked; if (S.cal?.mode === 'oven') { S.cal.emissivityKnown = S.known; S.cal.absolute = S.known; } invalidate(5, 'cal'); },
    'wr-darkmode': () => { S.darkMode = v as typeof S.darkMode; invalidate(1, 'oven'); },
    'wr-pcx': () => movePlateau(0, 0, 0, { cx: val, cy: S.oven.plateaus[String(S.oven.series[S.oven.cur].tC)].cy }),
    'wr-pcy': () => movePlateau(0, 0, 0, { cx: S.oven.plateaus[String(S.oven.series[S.oven.cur].tC)].cx, cy: val }),
    'wr-pr': () => { const pl = S.oven.plateaus[String(S.oven.series[S.oven.cur].tC)]; movePlateau(0, 0, val - pl.radius); },
    'wr-tsph': () => { S.sphere.tSphUs = val; },
    'wr-sunit': () => { S.sphere.valUnit = v as typeof S.sphere.valUnit; },
    'wr-slam': () => { S.sphere.lamUnit = v as typeof S.sphere.lamUnit; },
    'wr-nd': () => { S.sphere.nd = v as typeof S.sphere.nd; },
    'wr-od': () => { S.sphere.od = val; },
    'wr-fill': () => { S.sphere.fill = v as 'median' | 'radial'; },
    'wr-texp': () => { S.flame.texpUs = val; invalidate(3, 'flame'); },
    'wr-texpok': () => { S.flame.texpConfirmed = (t as HTMLInputElement).checked; invalidate(5, 'cal'); },
    'wr-bin': () => { S.group = v === '2x2'; reopts(); },
    'wr-centre': () => { S.centre = (t as HTMLInputElement).checked; reopts(); },
    'wr-times': () => { timesSet = true; S.flame.times = v as 'repeat' | 'instants'; if (!S.flame.instant) S.flame.instant = flameItems()[0]?.id ?? ''; invalidate(3, 'flame'); },
    'wr-instant': () => { S.flame.instant = v; invalidate(3, 'flame'); },
    // GB3 fix, the per-frame axis and asymmetry depend on the orientation, so they are measured again
    'wr-sensq': () => { S.flame.sensorQ = +v; invalidate(3, 'flame'); void loadFlame(); },
    'wr-quarter': () => { S.flame.quarter = +v; S.flame.burnerOk = false; invalidate(3, 'flame'); void loadFlame(); },
    'wr-burnerok': () => { S.flame.burnerOk = (t as HTMLInputElement).checked; if (S.flame.burnerOk) S.flame.msgs = S.flame.msgs.filter((m) => m.code !== 'burner_needed'); },
    'wr-mirror': () => { S.flame.mirror = (t as HTMLInputElement).checked; invalidate(3, 'flame'); void loadFlame(); },
    'wr-fine': () => { S.flame.fineDeg = fin(val) ? val : 0; invalidate(3, 'flame'); },
    'wr-axisauto': () => { S.flame.axisAuto = (t as HTMLInputElement).checked; if (!S.flame.axisAuto && !fin(S.flame.axisManual) && S.flame.prepared) S.flame.axisManual = +S.flame.prepared.axis.toFixed(2); invalidate(3, 'flame'); },
    'wr-axis': () => { S.flame.axisManual = val; invalidate(3, 'flame'); },
    'wr-hw': () => { S.flame.halfWidth = val; invalidate(3, 'flame'); },
    'wr-nr': () => { S.flame.nr = val; invalidate(3, 'flame'); },
    'wr-rtop': () => { S.flame.rowTop = val; invalidate(3, 'flame'); },
    'wr-rbot': () => { S.flame.rowBottom = val; invalidate(3, 'flame'); },
    'wr-rph': () => { if (fin(val) && val >= 1) { S.flame.rowsPerHeight = Math.round(val); invalidate(3, 'flame'); } },
    'wr-burner': () => { S.flame.burnerRow = val; invalidate(3, 'flame'); },
    'wr-zb': () => { S.flame.zBurner = fin(val) ? val : 0; invalidate(3, 'flame'); },
    'wr-mmpx': () => { S.flame.mmPerPx = val; invalidate(3, 'flame'); },
    'wr-sunc': () => { S.flame.scaleUnc = val; },
    'wr-scaleok': () => { S.flame.scaleConfirmed = (t as HTMLInputElement).checked; invalidate(5, 'cal'); },
    'wr-rmm': () => { S.flame.rulerMm = val; rulerScale(); },
    'wr-sym': () => { S.abel.sym = v as Sym; invalidate(4, 'abel'); },
    // a value set by the student turns the automatic smoothing off (DESIGN §4, C4), drawing never fires this
    'wr-lam': () => { S.abel.lambdaRel = 10 ** val; S.abel.lambdaAuto = false; S.abel.lamFlat = false; const o = t.parentElement?.querySelector('output'); if (o) o.textContent = sig(S.abel.lambdaRel, 2); invalidate(4, 'abel'); },
    'wr-lam-auto': () => { S.abel.lambdaAuto = (t as HTMLInputElement).checked; S.abel.lamFlat = false; if (S.abel.lambdaAuto && S.abel.lcurve && S.abel.lcFor === S.flame.prepared) { S.abel.lambdaRel = S.abel.lcurve.lambda; } invalidate(4, 'abel'); },
    'wr-seg': () => { if (fin(val)) { S.abel.segments = Math.round(val); invalidate(4, 'abel'); } },
    'wr-satl': () => { if (fin(val)) { S.abel.satLambda = val; invalidate(4, 'abel'); } },
    'wr-ann-it': () => { if (fin(val)) S.abel.annabel.iterations = Math.max(100, Math.round(val)); },
    'wr-seed': () => { if (fin(val)) S.res.seed = Math.round(val); },
    'wr-seed6': () => { if (fin(val)) { S.res.seed = Math.round(val); S.res.band = null; } },
    'wr-path': () => { S.tfs.path = asTfsPath(v); invalidate(5, 'tfs'); if (S.tfs.path === 'net' && S.abel.out && !S.tfs.net.estimate && !S.tfs.net.running) void estimateNet(); },
    'wr-net-it': () => { if (fin(val)) { S.tfs.net.iterations = Math.max(10, Math.round(val)); S.tfs.net.msgs = S.tfs.net.msgs.filter((m) => m.code !== 'net_slow'); } },
    'wr-pair': () => { S.tfs.pair = v as 'RG' | 'BG' | 'BR'; invalidate(5, 'tfs'); },
    'wr-em': () => { S.tfs.em = v as typeof S.tfs.em; invalidate(5, 'tfs'); },
    'wr-ema': () => { S.tfs.a = val; invalidate(5, 'tfs'); }, 'wr-emb': () => { S.tfs.b = val; invalidate(5, 'tfs'); }, 'wr-emc': () => { S.tfs.c = val; invalidate(5, 'tfs'); }, 'wr-eme': () => { S.tfs.E = val; invalidate(5, 'tfs'); },
    'wr-sig': () => { if (fin(val)) { S.tfs.sigFrac = val / 100; invalidate(5, 'tfs'); } },
    'wr-island': () => { if (fin(val) && val >= 0) { S.tfs.islandFrac = val / 100; invalidate(5, 'tfs'); } },
    'wr-zcut': () => { S.tfs.zCut = val; invalidate(5, 'tfs'); },
    'wr-chem': () => { if (fin(val)) { S.tfs.chemThr = val / 100; invalidate(5, 'tfs'); } },
    'wr-fsch': () => { S.tfs.fsChannel = asFsMode(v); invalidate(5, 'tfs'); },
    'wr-tmin': () => { if (fin(val)) { S.res.Tmin = val; S.res.Tmanual = true; drawResults(); } },
    'wr-tmax': () => { if (fin(val)) { S.res.Tmax = val; S.res.Tmanual = true; drawResults(); } },
    'wr-fsmax': () => { if (fin(val) && val > 0) { S.res.fsMax = val; drawResults(); } },
    'wr-nband': () => { if (fin(val)) S.res.nBand = Math.max(5, Math.min(200, Math.round(val))); },
  };
  if (/^wr-hz\d$/.test(id) && S.flame.prepared) {
    const p = S.flame.prepared, j = +id.slice(-1);
    if (fin(val)) { let best = 0; for (let k = 0; k < p.nz; k++) if (Math.abs(p.zMm[k] - val) < Math.abs(p.zMm[best] - val)) best = k; S.res.heights[j] = best; } else S.res.heights[j] = NaN;
    if (e.type === 'change') drawResults();
    return;
  }
  if (t.classList.contains('wr-height')) {
    S.flame.height = +v;
    // the two height sliders (step 3 view and step 2 Options) write the same state
    for (const h of Array.from(document.querySelectorAll<HTMLInputElement>('input.wr-height'))) { if (h !== t) h.value = v; const o = h.parentElement?.querySelector('output'); if (o && S.flame.prepared) o.textContent = `${num(S.flame.prepared.zMm[S.flame.height], 1)} mm`; }
    drawFlame(); drawAbel();
    return;
  }
  const f = map[id];
  if (!f) return;
  f();
  const live = new Set(['wr-eps', 'wr-lam', 'wr-pcx', 'wr-pcy', 'wr-pr']);
  if (id === 'wr-lam' && e.type === 'change') { renderAll(); return; }
  if (e.type === 'change' || (t.tagName === 'SELECT') || (t as HTMLInputElement).type === 'checkbox') {
    if (!live.has(id) && !/^wr-(tmin|tmax|fsmax)$/.test(id)) { if (e.type === 'change' && (t as HTMLInputElement).type === 'number') renderSoon(); else renderAll(); }
  } else if (live.has(id) && id === 'wr-eps') { /* output updated in place */ }
}
/** New load options for the Worker, then every file is read again (grouping or R and B centring changed). */
function reopts() {
  ask({ type: 'opts', group: S.group, bayer: S.bayer, centre: S.centre }).then(async () => { await scanItems(S.items); await harmonise(); invalidate(1, 'files'); renderAll(); await analyseOven(); await loadFlame(); if (S.flame.rulerId) await loadRuler(); });
}
function rulerScale() {
  const pts = S.flame.rulerPts;
  if (pts.length !== 2 || !fin(S.flame.rulerMm)) return;
  const px = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) * S.flame.rulerScale;
  if (px > 0) { S.flame.mmPerPx = S.flame.rulerMm / px; invalidate(3, 'flame'); }
}
async function onClick(e: MouseEvent) {
  const t = (e.target as HTMLElement).closest('button, canvas') as HTMLElement | null;
  if (!t) return;
  const id = t.id;
  // the buttons of a group keep the focus after their section is drawn again (GB3r1-web-01)
  if (t.dataset.calpath) { S.calPath = t.dataset.calpath as typeof S.calPath; keepFocus(renderStage1); return; }
  if (t.dataset.series !== undefined) { S.oven.cur = +t.dataset.series; keepFocus(() => { renderPlateau(); renderFit(); }); return; }
  if (t.dataset.view !== undefined) { S.oven.view = +t.dataset.view; keepFocus(renderPlateau); return; }
  if (t.dataset.method) { S.abel.method = t.dataset.method as typeof S.abel.method; if (S.abel.method === 'annabel' || S.abel.method === 'sat') S.abel.sym = 'both'; else if (S.abel.sym === 'both') S.abel.sym = 'average'; keepFocus(renderStage4);
    // without a graphics card the time is measured at once, so it is in sight before training (GB1 decision, GB3r3-adv-13)
    if (S.abel.method === 'annabel' && !S.gpu && !S.abel.annabel.estimate && S.flame.prepared && !S.abel.annabel.running) void estimateAnnabel();
    return; }
  if (t.dataset.npy) { const f = npyFields()[t.dataset.npy]; if (f) save(new Blob([f as BlobPart]), `${t.dataset.npy}.npy`); return; }
  if (/^wr-ex-\d+$/.test(id)) { await loadExample(S.o.examples[+id.slice(6)]); return; }
  if (id === 'wr-pcv') {
    const s = S.oven.series[S.oven.cur];
    const r = t.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * s.preview.w * s.preview.step, y = ((e.clientY - r.top) / r.height) * s.preview.h * s.preview.step;
    movePlateau(0, 0, 0, { cx: Math.round(x * 2) / 2, cy: Math.round(y * 2) / 2 });
    return;
  }
  if (id === 'wr-rcv' && S.flame.rulerPreview) {
    const pv = S.flame.rulerPreview, r = t.getBoundingClientRect(), V = rulerView(pv);
    const [x, y] = V.inv(((e.clientX - r.left) / r.width) * V.W, ((e.clientY - r.top) / r.height) * V.H);
    if (S.flame.rulerPts.length >= 2) S.flame.rulerPts = [];
    S.flame.rulerPts.push([x * pv.step, y * pv.step]);
    rulerScale();
    keepFocus(renderStage3);
    return;
  }
  const actions: Record<string, () => unknown> = {
    'wr-reco': recommended,
    'wr-clear': async () => { await clearTable(); renderAll(); },
    'wr-resp-clear': () => { S.custom = null; invalidate(1, 'psi'); renderAll(); },
    'wr-preset-pl': () => { const s = S.oven.series[S.oven.cur]; S.oven.plateaus[String(s.tC)] = { cy: s.plateau.cy, cx: s.plateau.cx, radius: s.plateau.radius }; s.prof = undefined; invalidate(1, 'oven'); renderPlateau(); },
    'wr-oven-fit': fitOvenNow,
    'wr-flat-build': buildFlat,
    'wr-sph-run': fitSphere,
    'wr-cal-save': downloadCal,
    'wr-prep': prepareFlameNow,
    'wr-npz': downloadNpz,
    'wr-rclear': () => { S.flame.rulerPts = []; renderStage3(); },
    'wr-abel': abelClick,
    'wr-ann-stop': () => S.abel.annabel.ctrl?.abort(),
    'wr-net-stop': () => S.tfs.net.ctrl?.abort(),
    'wr-net-est': estimateNet,
    'wr-ann-est': estimateAnnabel,
    'wr-lc': runLcurve,
    'wr-lc-use': () => { if (S.abel.lcurve) { S.abel.lambdaRel = S.abel.lcurve.lambda; S.abel.lambdaAuto = true; S.abel.lamFlat = false; invalidate(4, 'abel'); renderAll(); } },
    'wr-cmp': compareMethods,
    'wr-tfs': runTfsNow,
    'wr-band': runBandNow,
    'wr-dl-csv': () => save(new Blob([csvText()], { type: 'text/csv' }), `${FN.results()}_${stamp()}.csv`),
    'wr-dl-csvpt': () => save(new Blob([csvText(true)], { type: 'text/csv' }), `${FN.resultsPt()}_${stamp()}.csv`),
    'wr-dl-png-T': async () => { const b = await pngOf('T'); if (b) save(b, `${FN.map()}_T_${theme()}_${stamp()}.png`); },
    'wr-dl-png-fs': async () => { const b = await pngOf('fs'); if (b) save(b, `${FN.map()}_fs_${theme()}_${stamp()}.png`); },
    'wr-dl-run': () => save(new Blob([JSON.stringify(runRecord(), null, 1)], { type: 'application/json' }), `run_${stamp()}.json`),
    'wr-dl-zip': downloadAll,
  };
  const a = actions[id];
  if (a) await a();
}
async function onFiles(e: Event) {
  const t = e.target as HTMLInputElement;
  if (t.type !== 'file' || !t.files?.length) return;
  const id = t.id;
  if (id === 'wr-q-file' || id === 'wr-q-dir') { await quickReview(filesFromInput(t)); return; }
  if (id === 'wr-o-dir' || id === 'wr-o-files' || id === 'wr-f-dir' || id === 'wr-f-files') { S.loadFrom = id.startsWith('wr-f') ? 3 : 1; const f = filesFromInput(t); await loadFiles(f, f.length === 1 ? f[0].path : L(`${f.length} archivos`, `${f.length} files`)); return; }
  if (id === 'wr-run-in') { await loadRun(t.files[0], t.files[0].name); return; }
  if (id === 'wr-calin') { await loadCalFile(t.files[0]); return; }
  if (id === 'wr-spec') { S.sphere.spectrum = await t.files[0].text(); S.sphere.spectrumName = t.files[0].name; renderStage1(); return; }
  if (id === 'wr-ndc') { S.sphere.ndCurve = await t.files[0].text(); S.sphere.ndCurveName = t.files[0].name; renderStage1(); return; }
  if (id === 'wr-resp') {
    try { const r = parseResponseCsv(await t.files[0].text(), t.files[0].name); S.custom = { name: t.files[0].name, lamNm: r.lamNm, eta: r.eta }; invalidate(1, 'psi'); }
    catch { S.loadMsgs.push({ code: 'response_bad' }); }
    renderAll();
  }
}
function onKey(e: KeyboardEvent) {
  const t = e.target as HTMLElement;
  if (t.id === 'wr-pcv') {
    const step = e.shiftKey ? 5 : 1;
    const m: Record<string, [number, number, number]> = { ArrowLeft: [-step, 0, 0], ArrowRight: [step, 0, 0], ArrowUp: [0, -step, 0], ArrowDown: [0, step, 0], '+': [0, 0, 0.5], '=': [0, 0, 0.5], '-': [0, 0, -0.5] };
    const d = m[e.key];
    if (d) { e.preventDefault(); movePlateau(d[0], d[1], d[2]); }
  }
}

// ============================================================================ mount
export function mount(root: HTMLElement, o: MountOptions) {
  S.o = o;
  setLang(o.lang);
  worker = new Worker(new URL('./pyro.worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (ev: MessageEvent<FromWorker>) => {
    const m = ev.data, p = pending.get(m.req);
    if (!p) return;
    if (m.kind === 'progress') { p.prog?.(m.done, m.total); return; }
    pending.delete(m.req);
    if (m.kind === 'ok') p.res(m.data); else p.rej({ code: m.code, detail: m.detail });
  };
  worker.onerror = () => { for (const [, p] of pending) p.rej({ code: 'worker_failed', detail: {} }); pending.clear(); };
  // helper Workers for ANNAbel on the CPU, created now and idle until used (§11.5)
  try { helpers = Array.from({ length: helperCount() }, () => defaultHelperFactory()); } catch { helpers = []; }
  ask({ type: 'opts', group: S.group, bayer: S.bayer, centre: S.centre });
  // five steps, the result of step 1 keeps id wr-s2 inside wr-s1 (DESIGN, correspondence table)
  root.innerHTML = `<div class="wr-top"><nav class="wr-steps" id="wr-nav" aria-label="${L('Pasos', 'Steps')}"></nav><p class="wr-busy" id="wr-busy" role="status" aria-live="polite" hidden></p></div>
  <p class="sr-only" id="wr-announce" role="status" aria-live="polite"></p><p class="sr-only" id="wr-alert" role="alert"></p>
  <section class="wr-bar" id="wr-s0"></section>
  ${[1, 3, 4, 5, 6].map((k) => `<section class="wr-stage" id="wr-s${k}"></section>`).join('')}`;
  // the Options fold keeps its state when a step is drawn again (toggle does not bubble, so capture)
  root.addEventListener('toggle', (e) => { const d = e.target as HTMLDetailsElement; if (d?.classList?.contains('wr-opts')) S.ui.open[+d.id.replace('wr-opt-', '')] = d.open; }, true);
  root.addEventListener('input', onInput);
  root.addEventListener('change', (e) => { const t = e.target as HTMLInputElement; if (t.type === 'file') onFiles(e); else onInput(e); });
  root.addEventListener('click', (e) => { onClick(e as MouseEvent); });
  root.addEventListener('keydown', onKey);
  root.addEventListener('pointerdown', () => { pointerDown = true; });
  window.addEventListener('pointerup', () => { pointerDown = false; if (pendingRender) setTimeout(() => { if (pendingRender && !pointerDown) { pendingRender = false; renderAll(); } }, 0); });
  root.addEventListener('dragover', (e) => { const d = (e.target as HTMLElement).closest('.wr-drop'); if (d) { e.preventDefault(); d.classList.add('wr-over'); S.loadFrom = d.closest('#wr-s3') ? 3 : 1; } });
  root.addEventListener('dragleave', (e) => { (e.target as HTMLElement).closest('.wr-drop')?.classList.remove('wr-over'); });
  root.addEventListener('drop', async (e) => {
    const d = (e.target as HTMLElement).closest('.wr-drop');
    if (!d) return;
    e.preventDefault(); d.classList.remove('wr-over');
    const out: InFile[] = [];
    const items = Array.from(e.dataTransfer?.items ?? []);
    const entries = items.map((it) => (it as DataTransferItem & { webkitGetAsEntry?: () => FileSystemEntry | null }).webkitGetAsEntry?.() ?? null);
    if (entries.some(Boolean)) { for (const en of entries) if (en) await walkEntry(en, '', out); }
    else for (const f of Array.from(e.dataTransfer?.files ?? [])) out.push({ path: f.name, blob: f });
    await loadFiles(out, out.length === 1 ? out[0].path : L(`${out.length} archivos`, `${out.length} files`));
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (S.tfs.out) drawResults(); drawAbel(); });
  new MutationObserver(() => { if (S.tfs.out) drawResults(); drawAbel(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  renderAll();
  webgpuAvailable().then((g) => { S.gpu = g; renderStage0(); });
  void cssVar; void msgText;
}
