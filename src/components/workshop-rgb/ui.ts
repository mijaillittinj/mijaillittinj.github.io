/**
 * Interface of the RGB pyrometry tool (WEB_SPEC §4), plain TypeScript without a framework so the
 * offline single-file copy can be built from the same source with esbuild. `mount` fills the container
 * that Tool.astro writes, one section per stage, each with its status line and its [Calcular] button.
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
import { buildStage2Problem, initStage2, trainStage2, finishStage2, Stage2LocalEngine, STAGE2_DEFAULTS, XCHECK_WARN_K, XCHECK_WARN_P90_K, XCHECK_WARN_FS_LOG, type KernelTable, type XCheck, type BoundReport, type Stage2Problem } from '../../lib/workshop-rgb/annabel/stage2.ts';
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

export interface ExampleSet { file: string; bytes: number; sha256: string; camera: string; available: boolean; preset?: { texpUs: number; mmPerPx: number; zBurnerMm: number } }
export interface MountOptions { lang: Lang; version: string; commit: string; source?: string; built?: string; base: string; examples: ExampleSet[]; pageUrl: string; biases: { key: string; es: string; en: string }[]; ovenFlameBias?: { es: string; en: string }; tailBias?: { es: string; en: string } }

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
  abel: { method: 'tikhonov' as AbelMethod | 'annabel', sym: 'average' as Sym, lambdaRel: 0.1, segments: 12, satLambda: 1e-4, lcurve: null as null | { lambdas: number[]; rho: number[]; eta: number[]; lambda: number; index: number }, out: null as AbelOut | null, outMethod: '' as string, compare: null as null | Record<string, Float64Array>,
    annabel: { running: false, ctrl: null as AbortController | null, tick: null as TrainTick | null, estimate: null as null | { path: string; sPerIter: number }, iterations: ITER_LONG, result: null as TrainResult | null, msgs: [] as Msg[] } },
  tfs: { path: 'ratio' as TfsPath, net: { running: false, ctrl: null as AbortController | null, tick: null as TrainTick | null, estimate: null as null | { path: string; sPerIter: number }, iterations: NET_ITERATIONS, result: null as NetOut | null, msgs: [] as Msg[] }, pair: 'RG' as 'RG' | 'BG' | 'BR', em: 'group' as 'group' | 'coeffs' | 'constant', a: 0.284, b: 0.272, c: 0.057, E: 0.3, sigFrac: 0.02, islandFrac: 0.05, zCut: NaN, chemThr: 0.1, fsChannel: 'RG' as FsMode, out: null as TfsOut | null, dec: null as CalDecision | null },
  res: { Tmin: 1400, Tmax: 2100, Tmanual: false, Tout: 0, bandHow: '' as '' | 'oven' | 'noise' | 'nocal', fsMax: NaN, heights: [] as number[], band: null as null | { Tlo: Float64Array; Thi: Float64Array; fslo: Float64Array; fshi: Float64Array }, seed: 42, nBand: 30, view: 'T' as string },
  stale: {} as Record<number, string>,
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
const roleName = (r: Role) => ({ oven: L('horno', 'oven'), dark: L('oscuro', 'dark'), flat: L('campo plano', 'flat field'), sphere: L('esfera', 'sphere'), flame: L('llama', 'flame'), background: L('fondo', 'background'), ruler: L('regla', 'ruler'), ignore: L('ignorar', 'ignore') } as Record<Role, string>)[r];
const subs = (t: string) => t.replace(/f_s/g, 'f<sub>s</sub>').replace(/E_G\/E_R/g, 'E<sub>G</sub>/E<sub>R</sub>').replace(/K_j/g, 'K<sub>j</sub>');
const statusLine = (id: string, text: string) => `<p class="wr-status"><span class="label">${L('Estado', 'Status')}</span> <em id="${id}">${subs(text)}</em></p>`;
const fin = (x: number) => Number.isFinite(x);
/** Number with its sign, +342 or −29. */
const sgn = (x: number, d = 0) => (fin(x) && x > 0 ? `+${num(x, d)}` : num(x, d));
/** Drift of K_R/K_G between the lowest and the highest oven temperature, the figure the pages give. */
function krkgDrift(f: OvenResult['fit']): number {
  const R = f.channels.R.perT, G = f.channels.G.perT;
  if (R.length < 2 || G.length < 2) return NaN;
  const q = (i: number) => R[i].K / (G.find((g) => g.tC === R[i].tC)?.K ?? NaN);
  return Math.abs(q(R.length - 1) / q(0) - 1);
}
const ovenBias = () => (S.o.ovenFlameBias ? (getLang() === 'es' ? S.o.ovenFlameBias.es : S.o.ovenFlameBias.en) : '');

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
const REASON: Record<string, () => string> = {
  psi: () => L('cambió la respuesta espectral', 'the spectral response changed'),
  cal: () => L('cambió la calibración', 'the calibration changed'),
  files: () => L('cambiaron los archivos', 'the files changed'),
  oven: () => L('cambió un ajuste del horno', 'an oven setting changed'),
  flame: () => L('cambiaron las imágenes o la geometría de la llama', 'the flame images or geometry changed'),
  abel: () => L('cambió la inversión', 'the inversion changed'),
  tfs: () => L('cambió un ajuste de temperatura', 'a temperature setting changed'),
};
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
  if (added.some((it) => it.role === 'sphere') && !added.some((it) => it.role === 'oven')) { S.calPath = 'sphere'; const s = added.find((it) => it.role === 'sphere'); if (s?.texpUs) S.sphere.tSphUs = s.texpUs; }
  renderAll();
  announce(added.some((it) => it.role === 'flame') && !added.some((it) => it.role === 'oven' || it.role === 'sphere') ? 3 : 1);
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

async function loadExample(ex: ExampleSet) {
  if (!ex.available) { S.loadMsgs = [{ code: 'example_missing' }]; renderStage0(); return; }
  S.busy = L('Bajando el juego de ejemplo', 'Downloading the example set');
  renderBusy();
  try {
    const r = await fetch(`${S.o.base}teaching/workshop-rgb-pyrometry/files/${ex.file}`);
    if (!r.ok) throw new Error('http');
    const blob = await r.blob();
    S.busy = '';
    const cleared = S.items.length > 0;
    if (cleared) await clearTable();
    await loadFiles([{ path: ex.file, blob }], ex.file, ex.file);
    if (cleared) { S.loadMsgs.unshift({ code: 'example_cleared', level: 'info' }); renderAll(); }
    // the documented case of the set, so the result compares with its ground truth (GB3r2-estudiante-03)
    if (ex.preset) {
      const pr = ex.preset;
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
/** Uncertainty of T at 1800 K from the oven σ_T, through K_R/K_G (one ΔT common to all temperatures). */
function dTfromOven(): number {
  const c = S.cal;
  // with the tail calibrated, the parameter and the thermocouple together (deterministic, tailcal.ts)
  if (c?.mode === 'oven' && c.tail?.applied) return c.tail.uFlameK;
  if (!c || c.mode !== 'oven' || !fin(c.Tcal) || !fin(c.sigmaT)) return NaN;
  const p = psiNow();
  const band = makeBand(p.lamNm, p.psi, 'gl', emLaw());
  const lnq = (T: number) => Math.log(bandX(band, 'R', T) / bandX(band, 'G', T));
  const dlnK = Math.abs(lnq(c.Tcal + c.sigmaT) - lnq(c.Tcal - c.sigmaT)) / 2;
  const tab = ratioTable(band, Float64Array.from([1795, 1800, 1805]));
  const sens = Math.abs((Math.log(tab.RG[2]) - Math.log(tab.RG[0])) / 10);
  return dlnK / sens;
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
  if (!ids.length) { S.flame.msgs = [{ code: 'need_stage', vars: { a: '3' } }]; renderStage3(); return; }
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
    // the corner is applied at once, so Compute does not run with the old λ (GB3r2-estudiante-16)
    if (S.abel.lcurve && fin(S.abel.lcurve.lambda)) { S.abel.lambdaRel = S.abel.lcurve.lambda; invalidate(4, 'abel'); }
  } catch { /* the plot stays empty */ }
  S.busy = '';
  renderBusy();
  renderAll();
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
  S.busy = L('Midiendo 20 iteraciones de ANNAbel', 'Timing 20 ANNAbel iterations');
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
    if (!S.res.heights.length) S.res.heights = [Math.floor(p.nz * 0.25), Math.floor(p.nz * 0.5), Math.floor(p.nz * 0.75)];
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
/** What the island criterion removed, in words (director after GB3 r1). */
function islandNote(t: TfsOut): string {
  const i = t.islands;
  if (!i) return '';
  if (!i.n) return L(`No hay islas sueltas con este criterio. La zona mayor tiene ${num(i.largest)} píxeles.`, `No loose islands with this criterion. The largest region has ${num(i.largest)} pixels.`);
  return L(`Se quitaron ${num(i.n)} ${i.n === 1 ? 'isla' : 'islas'} con ${num(i.px)} píxeles en total, cada una con menos de ${num(i.frac * 100)} % de los ${num(i.largest)} píxeles de la zona mayor. Quedan como señal baja.`, `${num(i.n)} ${i.n === 1 ? 'island' : 'islands'} with ${num(i.px)} pixels in all were removed, each under ${num(i.frac * 100)} % of the ${num(i.largest)} pixels of the largest region. They count as low signal.`);
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
  if (!flameItems().length) { S.flame.msgs = [{ code: 'need_stage', vars: { a: '3' } }]; renderAll(); return; }
  if (!fin(S.flame.mmPerPx)) { S.flame.msgs = [{ code: 'scale_needed', level: 'error' }, ...S.flame.msgs.filter((m) => m.code !== 'scale_needed')]; renderAll(); announce(3); document.getElementById('wr-mmpx')?.focus(); return; }
  let autoFit = false;
  if ((!S.cal || S.stale[1]) && S.calPath === 'oven' && S.oven.series.length) { await fitOvenNow(); autoFit = true; }
  S.abel.method = 'tikhonov'; S.abel.sym = 'average'; S.tfs.path = 'ratio'; S.tfs.pair = 'RG'; S.tfs.fsChannel = 'RG'; S.tfs.sigFrac = 0.02;
  await prepareFlameNow();
  if (!S.flame.prepared) return;
  // an upside-down flame passes every test, so the student says where the burner is, now with the preview in sight (GB3r2-adv-02, r3-adv-01)
  if (!burnerGate()) return;
  await runLcurve();
  if (S.abel.lcurve) S.abel.lambdaRel = S.abel.lcurve.lambda;
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
    abel: { method: S.abel.outMethod || S.abel.method, sym: S.abel.sym, lambdaRel: S.abel.lambdaRel, reg: 'gamma2', segments: S.abel.segments, satLambda: S.abel.satLambda, annabelIterations: S.abel.annabel.iterations, annabelBest: S.abel.annabel.result?.bestIter ?? null },
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
    title: which === 'T' ? `T(r, z) · ${lab}` : `f_s(r, z) · ${lab}`, unit: which === 'T' ? 'K' : fs?.unit ?? '',
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
  S.crops = {}; S.log = []; S.flame.loaded = false; S.flame.rulerPreview = null; S.flame.rulerId = ''; S.flame.shift = null; S.flame.instant = ''; S.loadMsgs = []; S.flame.msgs = []; S.sphere.result = null; S.stale = {};
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
    Object.assign(S.abel, { method: s.abel.method, sym: s.abel.sym, lambdaRel: s.abel.lambdaRel, segments: s.abel.segments, satLambda: s.abel.satLambda });
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
  if (a) a.textContent = `${head}. ${st}.${warn ? L(` ${warn} ${warn === 1 ? 'aviso' : 'avisos'} en esta etapa.`, ` ${warn} ${warn === 1 ? 'warning' : 'warnings'} in this stage.`) : ''}`;
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
  keepFocus(() => { renderStage0(); renderStage1(); renderStage2(); renderStage3(); renderStage4(); renderStage5(); renderStage6(); });
  renderBusy();
}
function staleNote(stage: number): string {
  if (!S.stale[stage]) return '';
  const why = REASON[S.stale[stage]]?.() ?? L('cambió un ajuste anterior', 'an earlier setting changed');
  return L(`Desactualizada porque ${why}. Vuelvan a calcular esta etapa`, `Outdated because ${why}. Compute this stage again`);
}
function stateLabel(d: CalDecision): string {
  const base = d.state === 'absolute' ? L('Calibración absoluta', 'Absolute calibration') : d.state === 'relative' ? L('Calibración relativa. f_s en unidades relativas', 'Relative calibration. f_s in relative units') : d.state === 'inconsistent' ? L('Calibración no consistente', 'Inconsistent calibration') : d.state === 'untested' ? L('Calibración sin probar, una sola temperatura del horno', 'Untested calibration, a single oven temperature') : d.state === 'one_channel' ? L('Un solo canal calibrado', 'Only one channel calibrated') : L('Sin calibrar', 'Uncalibrated');
  return d.provisional && d.fs !== 'no' ? `${base}. ${L('f_s provisional', 'f_s provisional')}` : base;
}
function tableHtml(head: string[], rows: string[][], cls = ''): string {
  return `<div class="wr-tablewrap"><table class="${cls}"><thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
const numIn = (id: string, value: number, attrs: string, label: string, note = '') => `<label class="control wr-num"><span class="control__head"><span>${label}</span></span><input type="number" id="${id}" value="${fin(value) ? value : ''}" ${attrs}>${note ? `<span class="meta">${note}</span>` : ''}</label>`;
const sel = (id: string, opts: [string, string][], value: string, label: string, note = '') => `<label class="control"><span class="control__head"><span>${label}</span></span><select id="${id}">${opts.map(([v, t]) => `<option value="${v}"${v === value ? ' selected' : ''}>${t}</option>`).join('')}</select>${note ? `<span class="meta">${note}</span>` : ''}</label>`;
const chk = (id: string, on: boolean, label: string, note = '') => `<label class="wr-check"><input type="checkbox" id="${id}"${on ? ' checked' : ''}> ${label}${note ? `<span class="meta">${note}</span>` : ''}</label>`;
const btn = (id: string, text: string, disabled = false) => `<button class="btn" type="button" id="${id}"${disabled ? ' disabled' : ''}>${text}</button>`;
const fileBtn = (id: string, text: string, accept: string, dir = false) => `<label class="btn wr-filebtn">${text}<input type="file" id="${id}" ${accept ? `accept="${accept}"` : ''} ${dir ? 'webkitdirectory' : ''} multiple class="sr-only"></label>`;

// ============================================================================ stage 0
function browserChecks(): { label: string; ok: boolean | null; note?: string; title?: string }[] {
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return [
    { label: L('Cálculo en segundo plano', 'Computing in the background'), ok: !!worker, title: 'Module Workers' },
    { label: L('Huella SHA-256 de los archivos', 'SHA-256 fingerprint of the files'), ok: true, note: globalThis.crypto?.subtle ? '' : L('más lenta, igual funciona', 'slower, still works'), title: 'crypto.subtle' },
    { label: L('Abrir archivos .zip', 'Opening .zip files'), ok: typeof DecompressionStream !== 'undefined', title: 'DecompressionStream' },
    { label: L('ANNAbel con la tarjeta gráfica', 'ANNAbel on the graphics card'), ok: S.gpu, title: 'WebGPU', note: S.gpu === null ? L('comprobando', 'checking') : S.gpu ? L('ANNAbel puede correr en el navegador', 'ANNAbel can run in the browser') : L('ANNAbel queda como opción avanzada', 'ANNAbel stays an advanced option') },
    { label: L('Memoria aproximada', 'Approximate memory'), ok: mem === undefined ? null : mem >= 4, note: mem === undefined ? L('el navegador no la informa', 'the browser does not report it') : `${num(mem)} GB` },
  ];
}
function renderStage0() {
  const el = document.getElementById('wr-s0');
  if (!el) return;
  const yes = L('sí', 'yes'), no = L('no', 'no');
  const exName: Record<string, [string, string]> = {
    'horno_ejemplo.zip': [L('Horno real', 'Real oven'), L('a2A1920, cuatro temperaturas, hasta la etapa 2', 'a2A1920, four temperatures, up to stage 2')],
    'llama_septiembre_2026.zip': [L('Llama de septiembre con su esfera', 'September flame with its sphere'), L('acA2040, de punta a punta', 'acA2040, end to end')],
    'llama_sintetica_coflame.zip': [L('Llama sintética con su horno', 'Synthetic flame with its oven'), L('los seis pasos, con los valores verdaderos de T y f_s para comparar', 'the six steps, with the ground truth T and f_s to compare against')],
  };
  el.innerHTML = `
  <h2 class="row__label nonum">${L('Etapa 0 · Inicio', 'Stage 0 · Start')}</h2>
  ${statusLine('wr-st0', L('Lista', 'Ready'))}
  <h3>${L('Comprobación del navegador', 'Browser check')}</h3>
  ${tableHtml([L('Función', 'Feature'), L('Disponible', 'Available'), L('Nota', 'Note')], browserChecks().map((c) => [c.title ? `<span title="${c.title}">${c.label}</span>` : c.label, c.ok === null ? '–' : c.ok ? yes : `<em>${no}</em>`, c.note ?? '']))}
  <h3>${L('Datos de ejemplo', 'Example data')}</h3>
  <ul class="wr-plain">${S.o.examples.map((ex, k) => `<li>${btn(`wr-ex-${k}`, L(`Cargar ${exName[ex.file]?.[0] ?? ex.file}`, `Load ${exName[ex.file]?.[0] ?? ex.file}`), false)} <span class="meta">${subs(exName[ex.file]?.[1] ?? '')}, ${kB(ex.bytes)}${ex.available ? '' : L(', todavía no publicado', ', not yet published')}</span></li>`).join('')}
    <li>${fileBtn('wr-run-in', L('Cargar un registro .json', 'Load a .json log'), '.json,application/json')} <span class="meta">${L('rearma los ajustes de un cálculo anterior y comprueba los archivos', 'restores the settings of an earlier run and checks the files')}</span></li></ul>
  <div class="wr-msgs">${msgHtml(S.loadMsgs.filter((m) => /^(example_|run_)/.test(m.code)))}</div>
  <h3>${L('Revisión rápida', 'Quick check')}</h3>
  <p>${L('Elijan un TIFF o una carpeta recién copiada del computador de la cámara. La tabla muestra en segundos cuánto de la escala usa cada canal, los píxeles saturados, los ceros del oscuro y si el nombre se entiende. La guía la pide después de cada serie. Esta revisión no carga la calibración.', 'Pick a TIFF or a folder just copied from the camera computer. Within seconds the table shows how much of full scale each channel uses, the saturated pixels, the zeros of the dark frame and whether the name can be read. The lab guide asks for it after each series. This check does not load the calibration.')}</p>
  <p class="cluster">${fileBtn('wr-q-file', L('Elegir un TIFF', 'Choose a TIFF'), '.tif,.tiff,image/tiff')} ${fileBtn('wr-q-dir', L('Elegir una carpeta', 'Choose a folder'), '', true)}</p>
  <div id="wr-quick"></div>
  <h3>${L('Camino recomendado', 'Recommended path')}</h3>
  <p>${L('Con las imágenes de la calibración y de la llama cargadas y la escala escrita en la etapa 3, este botón calcula el ajuste del horno si falta y corre las etapas 3 a 6 con Tikhonov en la esquina de la curva L, la razón rojo sobre verde, las máscaras por defecto y f_s por mínimos cuadrados con el rojo y el verde, como en la presentación (el azul queda fuera). Sin escala se detiene y lo dice. Los otros métodos quedan en el bloque Comparar métodos de la etapa 4.', 'With the calibration and flame images loaded and the scale typed in stage 3, this button computes the oven fit if it is missing and runs stages 3 to 6 with Tikhonov at the corner of the L-curve, the red-to-green ratio, the default masks and f_s by least squares of red and green, as in the presentation (blue left out). Without a scale it stops and says so. The other methods stay in the Compare methods block of stage 4.').replace(/f_s/g, 'f<sub>s</sub>')}</p>
  <p>${btn('wr-reco', L('Calcular con los valores recomendados', 'Compute with the recommended values'), !flameItems().length)}</p>`;
  renderQuick();
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
      return [`<span class="wr-path">${esc(it.path)}</span>`, `<em>${it.status === 'error' ? L('no se pudo leer', 'could not be read') : L('leyendo', 'reading')}</em>`, '', '', '', '', '', ''];
    }
    if (s.encoding === 'u8') msgs.push({ code: 'u8_gamma', files: [it.path] });
    if (s.encoding === 'u12') msgs.push({ code: 'u12_ambiguous', files: [it.path] });
    if (s.encoding === 'float') msgs.push({ code: 'float_nosat', files: [it.path] });
    if (s.software && !/pylon|basler/i.test(s.software)) msgs.push({ code: 'rewritten', vars: { program: s.software.replace(/\0/g, '').trim() }, files: [it.path] });
    s.satRaw.forEach((v, k) => { if (v > 0) msgs.push({ code: 'saturated', vars: { channel: k }, files: [it.path] }); });
    s.p999.forEach((v, k) => { if (s.satRaw[k] === 0 && v >= 0.9 * s.ceil12) msgs.push({ code: 'quick_high', vars: { channel: k }, files: [it.path] }); });
    if (s.zeroFrac.some((z) => z > 0.01)) msgs.push({ code: 'quick_zeros', files: [it.path] });
    const n = it.name;
    const read: string[] = [];
    if (n.tC !== undefined) read.push(tLabel(n.tC));
    if (n.texpUs !== undefined) read.push(n.texpAs && /ms/i.test(n.texpAs) ? L(`${n.texpAs.replace('.', ',')} leído como ${num(n.texpUs)} µs`, `${n.texpAs} read as ${num(n.texpUs)} µs`) : `${num(n.texpUs)} µs`);
    if (n.role !== 'oven') read.push(roleName(n.role));
    const cam = s.model ?? n.model ?? presetFor({ width: s.width, height: s.height }).preset.id;
    return [`<span class="wr-path">${esc(it.path)}</span>`, s.p999.map((v) => pct(v / s.ceil12, 0)).join(' / '), s.satRaw.map((v) => num(v)).join(' / '), s.zeroFrac.map((z) => pct(z, 1)).join(' / '), encName(s.encoding), `${s.width} × ${s.height}`, esc(cam), read.length ? read.join(', ') : `<em>${L('no se entiende', 'not understood')}</em>`];
  });
  el.innerHTML = tableHtml([L('Archivo', 'File'), L('Percentil 99,9 (% de la escala)', '99.9th percentile (% of full scale)') + ' R / G / B', L('Saturados', 'Saturated') + ' R / G / B', L('Ceros', 'Zeros') + ' R / G / B', L('Codificación', 'Encoding'), L('Tamaño', 'Size'), L('Cámara', 'Camera'), L('Nombre', 'Name')], rows, 'wr-quick') +
    `<div class="wr-msgs">${msgHtml(msgs)}</div><p>${btn('wr-q-use', L('Usar estas imágenes en la calibración (1a.1)', 'Use these images in the calibration (1a.1)'))}</p>`;
}

// ============================================================================ stage 1
function stage1Status(): string {
  if (S.busy && (S.oven.analysing)) return esc(S.busy);
  if (staleNote(1)) return staleNote(1);
  if (S.cal) {
    const c = S.cal;
    const how = c.mode === 'oven' ? L('con el horno', 'with the oven') : c.mode === 'sphere' ? L('con la esfera', 'with the sphere') : L('desde un archivo', 'from a file');
    return c.consistent ? L(`Lista, calibración ${how}`, `Ready, calibration ${how}`) : L(`Calibración ${how} no consistente`, `Calibration ${how} inconsistent`);
  }
  if (S.calPath === 'oven') {
    if (!S.items.some((it) => it.role === 'oven')) return L('Faltan las imágenes del horno', 'The oven images are missing');
    const miss = S.items.filter((it) => it.role === 'oven' && it.status !== 'error' && (it.tC === undefined || it.texpUs === undefined)).length;
    if (miss) return L(`Faltan la temperatura o el tiempo de exposición en ${miss} ${miss === 1 ? 'archivo' : 'archivos'}`, `Temperature or exposure time missing in ${miss} ${miss === 1 ? 'file' : 'files'}`);
    if (!S.oven.series.length) return L('No hay ninguna serie del horno que se pueda usar', 'There is no usable oven series');
    const n = S.oven.series.length;
    return L(`Falta el ajuste. ${n} ${n === 1 ? 'serie' : 'series'} con su meseta`, `The fit is missing. ${n} series with their plateau`);
  }
  if (S.calPath === 'sphere') return L('Falta calcular la esfera', 'The sphere is not computed yet');
  return L('Falta el archivo de calibración', 'The calibration file is missing');
}
function renderStage1() {
  const el = document.getElementById('wr-s1');
  if (!el) return;
  const p = preset();
  const cam = cameraOf(S.items);
  const mismatch = cam && !cam.cam.includes(p.id) && !p.id.includes(cam.cam) && !S.custom;
  el.innerHTML = `
  <h2 class="row__label nonum">${L('Etapa 1 · Calibración', 'Stage 1 · Calibration')}</h2>
  ${statusLine('wr-st1', stage1Status())}
  <h3>1.0 · ${L('Respuesta espectral', 'Spectral response')}</h3>
  <div class="wr-grid2">
    <div class="wr-controls">
      ${sel('wr-preset', PRESETS.map((x) => [x.id, x.id === 'a2A1920-160ucPRO' ? `a2A1920-160ucPRO, Sony IMX392 ${L('(laboratorio)', '(lab)')}` : `acA2040-55uc, ${L('campaña de septiembre de 2026', 'September 2026 campaign')}`] as [string, string]), S.presetId, L('Curva de la cámara', 'Camera curve'))}
      <p class="meta">${S.custom ? L(`Curva cargada desde ${esc(S.custom.name)}.`, `Curve loaded from ${esc(S.custom.name)}.`) : S.presetManual ? L('Elegida a mano en este selector.', 'Chosen by hand in this selector.') + (cam ? L(` Los archivos indican la ${esc(cam.cam)} por ${cam.why}.`, ` The files point to the ${esc(cam.cam)} from ${cam.why}.`) : '') : cam ? L(`Elegida por ${cam.why}.`, `Chosen from ${cam.why}.`) : L('Por defecto, la cámara del laboratorio. Carguen las imágenes para confirmarla.', 'By default, the lab camera. Load the images to confirm it.')}
      ${p.id === 'a2A1920-160ucPRO' && !S.custom ? L(' La curva ya incluye el filtro infrarrojo de la montura y no se agrega otro corte.', ' The curve already includes the infrared filter of the mount and no further cut is added.') : ''}${p.id === 'acA2040-55uc' && !S.custom ? L(' Basler, curva de ANNAbel RGB.', ' Basler, ANNAbel RGB curve.') : ''}</p>
      ${sel('wr-basis', [['photon', L('por fotón (QE)', 'per photon (QE)')], ['energy', L('por energía (A/W)', 'per energy (A/W)')]], S.basis, L('La curva se lee', 'The curve is read'), L('Por fotón se multiplica por λ/λ<sub>ref</sub>, con λ<sub>ref</sub> = 550 nm.', 'Per photon it is multiplied by λ/λ<sub>ref</sub>, with λ<sub>ref</sub> = 550 nm.'))}
      ${p.allowRemoveIr && !S.custom ? `${chk('wr-ir', S.irOn, L('Corte infrarrojo efectivo', 'Effective infrared cut'))}
        <p class="meta">${L('Esta curva no trae el filtro infrarrojo de la cámara. Se agrega como un borde suave con 50 % de transmisión en el corte. La transmisión baja de 73 a 27 % entre un ancho antes y un ancho después del corte.', 'This curve does not include the camera infrared filter. It is added as a soft edge with 50 % transmission at the cut. The transmission falls from 73 to 27 % between one width before and one width after the cut.')}</p>
        <div class="wr-row">${numIn('wr-ircut', S.irCutNm, 'min="550" max="900" step="1"', L('Corte (nm)', 'Cut (nm)'))}${numIn('wr-irw', S.irWidthNm, 'min="1" max="60" step="1"', L('Ancho (nm)', 'Width (nm)'))}</div>
        ${!S.irOn ? msgHtml([{ code: 'no_ir_cut' }]) : ''}` : ''}
      ${sel('wr-sp', [['none', L('ninguno', 'none')], ['sp700', L('pasa-cortos 700 nm (borde de 5 nm, OD 3)', 'short-pass 700 nm (5 nm edge, OD 3)')]], S.sp ? 'sp700' : 'none', L('Filtro delante del lente', 'Filter in front of the lens'), L('Opción para quien tenga otra óptica. El laboratorio no tiene este filtro, y la cola infrarroja se calibra con el horno en 1a.7. Si hay un filtro de paso corto de 700 nm delante del lente, en el horno y en la llama, elíjanlo aquí antes de cargar el horno. La herramienta lo multiplica en τ(λ) con un borde de 5 nm en 700 nm y transmisión 10⁻³ (OD 3) sobre el borde. Con el filtro puesto y sin declararlo, la prueba de la razón puede pasar igual y T de la llama salir baja sin aviso.', 'An option for other optics. The lab has no such filter, and the infrared tail is calibrated with the oven in 1a.7. If there is a 700 nm short-pass filter in front of the lens, for the oven and the flame, choose it here before loading the oven. The tool multiplies it into τ(λ) with a 5 nm edge at 700 nm and transmission 10⁻³ (OD 3) above the edge. With the filter on and not declared, the ratio test can still pass and the flame T can come out low with no warning.'))}
      <p class="cluster">${fileBtn('wr-resp', L('Cargar otra curva CSV', 'Load another CSV curve'), '.csv,text/csv')}${S.custom ? ` ${btn('wr-resp-clear', L('Volver a la curva de la cámara', 'Back to the camera curve'))}` : ''}</p>
      ${mismatch ? msgHtml([{ code: 'preset_mismatch', vars: { a: p.id, b: cam!.cam } }]) : ''}
    </div>
    <figure><div id="wr-psi" class="wr-plot"></div><figcaption>${L(`ψ<sub>j</sub>(λ) de los tres canales, normalizada al máximo del rojo, ${S.basis === 'photon' ? 'por fotón' : 'por energía'}.`, `ψ<sub>j</sub>(λ) of the three channels, normalised to the red maximum, ${S.basis === 'photon' ? 'per photon' : 'per energy'}.`)}</figcaption></figure>
  </div>
  <p class="wr-tabs" role="group" aria-label="${L('Camino de calibración', 'Calibration path')}"><span class="label">${L('Calibrar con', 'Calibrate with')}</span>${(['oven', 'sphere', 'file'] as const).map((k) => `<button type="button" class="wr-tab" data-calpath="${k}" aria-pressed="${S.calPath === k}">${k === 'oven' ? L('1a · Horno', '1a · Oven') : k === 'sphere' ? L('1b · Esfera integradora', '1b · Integrating sphere') : L('1c · Archivo', '1c · File')}</button>`).join('')}</p>
  <div id="wr-calpath"></div>
  <p class="cluster">${btn('wr-cal-save', L('Guardar calibración .json', 'Save calibration .json'), !S.cal)}</p>`;
  const ps = psiNow();
  const m = Math.max(...ps.psi.R);
  plot('wr-psi', { xlabel: 'λ (nm)', ylabel: 'ψⱼ (λ)', height: 230 }, CH.map((c, k) => ({ kind: 'line', x: ps.lamNm, y: Array.from(ps.psi[c], (v) => v / m), cls: chCls[k], label: c })), { xlim: [380, 1000] });
  if (S.calPath === 'oven') renderOven(); else if (S.calPath === 'sphere') renderSphere(); else renderCalFile();
}
function fileTable(filter: (it: Item) => boolean, id: string): string {
  const list = S.items.filter(filter);
  if (!list.length) return '';
  const msgs: Msg[] = [];
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
      if (s.software && !/pylon|basler/i.test(s.software)) msgs.push(S.example ? { code: 'rewritten_example', level: 'info' } : { code: 'rewritten', vars: { program: s.software.replace(/\0/g, '').trim() }, files: [it.path] });
      if (s.encoding === 'u8') msgs.push({ code: 'u8_gamma', files: [it.path] });
    }
    const readable = it.status !== 'error';
    const needT = readable && it.role === 'oven' && it.tC === undefined, needE = readable && (it.role === 'oven' || it.role === 'dark') && it.texpUs === undefined;
    // say what is missing in each file (GB3r2-adv-07)
    if (needT || needE) msgs.push({ code: needT && needE ? 'name_unread' : needT ? 'name_unread_T' : 'name_unread_texp', vars: { file: it.path } });
    const tNote = it.tFrom === 'log' && it.nameT !== undefined && Math.abs(it.nameT - it.logT!) > 0.05 ? `<span class="meta"> ${L('termopar', 'thermocouple')}, ${L('consigna', 'set point')} ${tLabel(it.nameT)}</span>` : it.tFrom === 'log' ? `<span class="meta"> ${L('registro', 'log')}</span>` : '';
    const eNote = it.texpFrom === 'pfs' ? '<span class="meta"> .pfs</span>' : it.name.texpAs && /ms/i.test(it.name.texpAs) ? `<span class="meta"> ${L('leído de', 'read from')} ${esc(it.name.texpAs)}</span>` : '';
    const roleSel = `<select data-id="${it.id}" data-f="role" aria-label="${L('Rol de', 'Role of')} ${esc(it.path)}">${ROLES.map((r) => `<option value="${r}"${r === it.role ? ' selected' : ''}>${roleName(r)}</option>`).join('')}</select>`;
    const tIn = `<input type="number" inputmode="decimal" step="any" data-id="${it.id}" data-f="tC" value="${it.tC ?? ''}" aria-label="${L('Temperatura del horno en °C de', 'Oven temperature in °C of')} ${esc(it.path)}"${needT ? ' aria-invalid="true" class="wr-missing"' : ''}>`;
    const eff = it.role === 'flame' && it.texpUs === undefined && fin(S.flame.texpUs) ? ` placeholder="${S.flame.texpUs}" title="${L('Se usa el t_exp de la llama escrito en la etapa 3', 'The flame t_exp typed in stage 3 is used')}"` : '';
    const eIn = `<input type="number" inputmode="decimal" step="any" data-id="${it.id}" data-f="texpUs" value="${it.texpUs ?? ''}"${eff} aria-label="${L('Tiempo de exposición en µs de', 'Exposure time in µs of')} ${esc(it.path)}"${needE ? ' aria-invalid="true" class="wr-missing"' : ''}>`;
    return [`<span class="wr-path">${esc(it.path)}</span>`, roleSel, it.role === 'oven' ? tIn + tNote : '', eIn + eNote, s ? `${s.width} × ${s.height}` : '', s ? encName(s.encoding) : '', state];
  });
  const needs = list.some((it) => it.status !== 'error' && it.role === 'oven' && (it.tC === undefined || it.texpUs === undefined));
  const fold = list.length > 12 && !needs;
  const pfsRows = [...new Map(list.filter((it) => it.pfs).map((it) => [it.path.split('/').slice(0, -1).join('/'), it.pfs!])).entries()];
  const keys = ['Gain', 'BlackLevel', 'BalanceRatio', 'PixelFormat', 'ExposureTime', 'ReverseX', 'ReverseY', 'OffsetX', 'OffsetY'];
  return `<details class="wr-filetable" id="${id}"${fold ? '' : ' open'}><summary>${L(`Tabla de archivos (${list.length}), ordenada por rol, serie y tiempo de exposición`, `File table (${list.length}), sorted by role, series and exposure time`)}</summary>
  ${tableHtml([L('Archivo', 'File'), L('Rol', 'Role'), L('T del horno (°C)', 'Oven T (°C)'), 't<sub>exp</sub> (µs)', L('Tamaño', 'Size'), L('Codificación', 'Encoding'), L('Estado', 'Status')], rows, 'wr-files')}</details>
  ${pfsRows.length ? tableHtml([L('Carpeta', 'Folder'), ...keys], pfsRows.map(([d, p]) => [`<span class="wr-path">${esc(d)}</span>`, ...keys.map((k) => esc(p[k] ?? '–'))])) : ''}
  <div class="wr-msgs">${msgHtml(msgs)}</div>`;
}
function renderOven() {
  const el = document.getElementById('wr-calpath');
  if (!el) return;
  const r = S.oven.result;
  const counts = S.items.reduce((a, it) => { a[it.role] = (a[it.role] ?? 0) + 1; return a; }, {} as Record<string, number>);
  const nImg = (role: string, n: number) => (n === 1 ? L(`1 imagen de ${roleName(role as Role)}`, `1 ${roleName(role as Role)} image`) : L(`${n} imágenes de ${roleName(role as Role)}`, `${n} ${roleName(role as Role)} images`));
  const nSeries = new Set(S.items.filter((it) => it.role === 'oven' && it.tC !== undefined).map((it) => it.tC)).size;
  el.innerHTML = `
  <h3>1a.1 · ${L('Imágenes del horno', 'Oven images')}</h3>
  <div class="wr-drop" id="wr-drop">
    <p>${L('Arrastren aquí la carpeta de la sesión, varios TIFF o un .zip. La temperatura sale de la carpeta, como <code>horno/1000C/</code>, y el tiempo de exposición del nombre, como <code>500exp.tiff</code>. La misma tabla recibe las imágenes de la llama, el fondo, la regla, la esfera y el campo plano.', 'Drag the session folder, several TIFF files or a .zip here. The temperature comes from the folder, such as <code>horno/1000C/</code>, and the exposure time from the name, such as <code>500exp.tiff</code>. The same table takes the flame, background, ruler, sphere and flat-field images.')}</p>
    <p class="cluster">${fileBtn('wr-o-dir', L('Elegir una carpeta', 'Choose a folder'), '', true)} ${fileBtn('wr-o-files', L('Elegir archivos o un .zip', 'Choose files or a .zip'), '.tif,.tiff,.zip,.csv,.pfs,.json')} ${S.items.length ? btn('wr-clear', L('Vaciar la tabla', 'Clear the table')) : ''}</p>
  </div>
  <div class="wr-grid3">
    ${numIn('wr-dT', S.sigmaT, 'min="0" max="50" step="1"', L('Incertidumbre del termopar, ± K', 'Thermocouple uncertainty, ± K'), L('Supuesto, 5 K por defecto.', 'Assumption, 5 K by default.'))}
    <label class="control"><span class="control__head">${L('Emisividad de la cavidad ε', 'Cavity emissivity ε')} <output id="wr-eps-out">${num(S.eps, 2)}</output></span><input type="range" id="wr-eps" min="0.80" max="1" step="0.01" value="${S.eps}"><span class="meta">${L('Supuesto. Con una cavidad gris la temperatura no depende de ε.', 'Assumption. With a grey cavity the temperature does not depend on ε.')}</span></label>
    ${chk('wr-known', S.known, L('La emisividad de la cavidad y la transmisión de ventanas son conocidas', 'The cavity emissivity and the window transmission are known'), S.known ? L('Calibración absoluta, da T y f_s en ppm. Desmárquenla si hay una ventana de transmisión desconocida.', 'Absolute calibration, gives T and f_s in ppm. Untick it if there is a window of unknown transmission.').replace(/f_s/, 'f<sub>s</sub>') : L('Calibración relativa. f_s en unidades relativas. La temperatura sale igual.', 'Relative calibration. f_s in relative units. The temperature comes out the same.').replace(/f_s/, 'f<sub>s</sub>'))}
  </div>
  ${S.items.length ? `<p class="meta">${esc(S.source)} · ${Object.entries(counts).map(([k, n]) => nImg(k, n)).join(', ')} ${L(`en ${nSeries} ${nSeries === 1 ? 'serie' : 'series'} del horno`, `in ${nSeries} oven series`)}${S.log.length ? ` · registro.csv ${L('con', 'with')} ${S.log.length} ${L('filas', 'rows')}` : ''}${Object.keys(S.crops).length ? ' · recorte.json' : ''}</p>
  <p class="meta">${L('Rol de cada archivo. Horno, imagen de la abertura caliente. Oscuro, tapa del lente puesta. Campo plano, esfera o difusor que llena el cuadro. Esfera, calibración con esfera integradora. Llama, fondo y regla son de la etapa 3. Ignorar lo deja fuera.', 'Role of each file. Oven, image of the hot aperture. Dark, lens cap on. Flat field, sphere or diffuser filling the frame. Sphere, calibration with an integrating sphere. Flame, background and ruler belong to stage 3. Ignore leaves it out.')}</p>` : ''}
  ${fileTable(() => true, 'wr-ft1')}
  <div class="wr-msgs">${msgHtml(S.loadMsgs.filter((m) => !/^(example_|run_)/.test(m.code)))}</div>
  <h3>1a.2 · ${L('Oscuro', 'Dark level')}</h3><div id="wr-dark"></div>
  <h3>1a.3 · ${L('Meseta', 'Plateau')}</h3><div id="wr-plateau"></div>
  <h3>1a.4 · ${L('Píxeles marcados', 'Flagged pixels')}</h3><div id="wr-flags"></div>
  <p>${btn('wr-oven-fit', L('Calcular el ajuste del horno', 'Compute the oven fit'), !S.oven.series.length)}</p>
  <h3>1a.5 · ${L('Ajuste de K<sub>j</sub>', 'Fit of K<sub>j</sub>')}</h3><div id="wr-fit">${r ? '' : `<p class="meta">${L('Aparece después del ajuste.', 'Appears after the fit.')}</p>`}</div>
  <h3>1a.6 · ${L('Linealidad', 'Linearity')}</h3><div id="wr-lin">${r ? '' : `<p class="meta">${L('Aparece después del ajuste.', 'Appears after the fit.')}</p>`}</div>
  <h3>1a.7 · ${L('Prueba de la razón', 'Ratio test')}</h3><div id="wr-ratio">${r ? '' : `<p class="meta">${L('Aparece después del ajuste. Prueba la razón contra la temperatura del termopar y queda siempre a la vista.', 'Appears after the fit. It tests the ratio against the thermocouple temperature and always stays visible.')}</p>`}</div>
  <h3>1a.8 · ${L('Campo plano (opcional)', 'Flat field (optional)')}</h3><div id="wr-flat"></div>`;
  renderDark(); renderPlateau(); renderFlags(); renderFit(); renderFlat();
}
function seriesTabs(): string {
  return `<p class="wr-tabs" role="group" aria-label="${L('Serie del horno', 'Oven series')}"><span class="label">${L('Serie del horno', 'Oven series')}</span>${S.oven.series.map((s, k) => `<button type="button" class="wr-tab" data-series="${k}" aria-pressed="${k === S.oven.cur}">${tLabel(s.tC)}</button>`).join('')}</p>`;
}
function renderDark() {
  const el = document.getElementById('wr-dark');
  if (!el) return;
  if (!S.oven.series.length) { el.innerHTML = `<p class="meta">${L('Aparece al cargar las imágenes del horno.', 'Appears once the oven images are loaded.')}</p>`; return; }
  const hasDark = darkItems().length > 0;
  const msgs: Msg[] = [];
  const mode = darkModeNow();
  if (!hasDark) msgs.push({ code: 'dark_estimated', level: 'info' });
  if (S.oven.series.some((s) => s.narrow)) msgs.push({ code: 'far_narrow', level: 'info' });
  const clipped = S.oven.series.filter((s) => s.zeroFrac > 0.01);
  if (clipped.length) msgs.push({ code: hasDark ? 'dark_clipped' : 'dark_clipped_nodark', files: clipped.map((s) => tLabel(s.tC)) });
  if (S.oven.darks.some((d) => d.zeroFrac > 0.01)) msgs.push({ code: 'dark_clipped' });
  const sp = S.oven.result?.fit.strayPct;
  // one notice with the three channels (GB3r3-adv-07)
  const spHits = sp ? CH.map((c, k) => (sp[c] > 1 ? `${num(Math.round(sp[c] * 10) / 10, 1)} % ${L('en', 'in')} ${chName(k)}` : '')).filter(Boolean) : [];
  if (spHits.length) msgs.push({ code: 'stray_light', vars: { a: spHits.join(L(', ', ', ')) } });
  const rows = S.oven.series.map((s) => {
    const sg = s.signals[0];
    return [tLabel(s.tC), s.narrow ? L('recorte sin anillo exterior', 'crop without outer ring') : L('anillo exterior, r > 2,5 radios', 'outer ring, r > 2.5 radii'), num(s.texpUs[0]), CH.map((c) => num(sg.strip[c], 1)).join(' / '), CH.map((c) => num(sg.stripSigma[c], 2)).join(' / '), pct(s.zeroFrac, 1)];
  });
  el.innerHTML = `
  ${sel('wr-darkmode', [['auto', L('automático', 'automatic')], ['intercept', L('pendiente contra t_exp, sin oscuro', 'slope against t_exp, no dark').replace(/t_exp/, 't<sub>exp</sub>')], ['strip', L('anillo fuera de la abertura', 'outer ring outside the aperture')], ['files', L('imágenes oscuras', 'dark frames')]], S.darkMode, L('Cómo se trata el oscuro', 'How the dark level is handled'), L(`En uso ${mode === 'files' ? 'imágenes oscuras, D<sub>j</sub> = a + b·t<sub>exp</sub> por canal' : mode === 'strip' ? 'el anillo fuera de la abertura' : 'oscuro no medido, K<sub>j</sub> por pendiente'}.`, `In use ${mode === 'files' ? 'dark frames, D<sub>j</sub> = a + b·t<sub>exp</sub> per channel' : mode === 'strip' ? 'the outer ring outside the aperture' : 'dark not measured, K<sub>j</sub> from the slope'}.`))}
  ${tableHtml([L('Serie', 'Series'), L('Zona', 'Region'), L('t<sub>exp</sub> más corto (µs)', 'Shortest t<sub>exp</sub> (µs)'), L('Nivel fuera de la abertura', 'Level outside the aperture') + ' R / G / B (DN)', 'σ R / G / B (DN)', L('Ceros', 'Zeros')], rows)}
  ${S.oven.darks.length ? tableHtml([L('Oscuro', 'Dark'), 't<sub>exp</sub> (µs)', L('Nivel', 'Level') + ' R / G / B (DN)', L('Ceros', 'Zeros')], S.oven.darks.map((d) => [esc(S.items.find((it) => it.id === d.id)?.path ?? d.id), num(d.texpUs), CH.map((c) => num(d.level[c], 1)).join(' / '), pct(d.zeroFrac, 1)])) : ''}
  <p class="meta">${L('Valores en DN de 12 bits sobre el superpíxel de 2 × 2, de la exposición más corta de cada serie. El nivel fuera de la abertura sube con el tiempo de exposición por la luz difusa y la cara del horno. Luz difusa bajo 1 % no cambia nada.', 'Values in 12-bit DN on the 2 × 2 superpixel, from the shortest exposure of each series. The level outside the aperture grows with exposure time through stray light and the oven face. Stray light under 1 % changes nothing.')}</p>
  <div class="wr-msgs">${msgHtml(msgs)}</div>`;
}
function renderPlateau() {
  const el = document.getElementById('wr-plateau');
  if (!el) return;
  const s = S.oven.series[S.oven.cur];
  if (!s) { el.innerHTML = `<p class="meta">${L('Aparece al cargar las imágenes del horno.', 'Appears once the oven images are loaded.')}</p>`; return; }
  const pl = S.oven.plateaus[String(s.tC)];
  el.innerHTML = `${seriesTabs()}
  <p>${L('La meseta es el disco central de la abertura donde la imagen es casi uniforme, y de ahí sale la señal del ajuste. La imagen del horno real cae suave hacia el borde como una cúpula, por eso la meseta es solo la parte central. La herramienta agrupa la imagen en bloques de 8 × 8, toma la mediana por anillo y ajusta una parábola en r² entre el centro y 0,6 radios. Propone el disco más grande donde la intensidad ajustada queda dentro de ±1 % del centro y R/G y B/G dentro de ±0,5 %. Pueden moverla con un clic, con las flechas del teclado sobre la imagen o con los controles. Las teclas + y − cambian el radio.', 'The plateau is the central disc of the aperture where the image is nearly uniform, and the signal of the fit comes from it. The real oven image falls gently towards the edge like a dome, so the plateau is only the central part. The tool groups the image in 8 × 8 blocks, takes the median of each ring and fits a parabola in r² from the centre to 0.6 radii. It proposes the largest disc in which the fitted intensity stays within ±1 % of the centre and R/G and B/G within ±0.5 %. You can move it with a click, with the arrow keys on the image or with the controls. The + and − keys change the radius.')}</p>
  <div class="lab"><div class="lab__grid">
    <div class="lab__controls">
      <p class="seg" role="group" aria-label="${L('Canal de la vista previa', 'Preview channel')}">${[0, 1, 2].map((c) => `<button type="button" data-view="${c}" aria-pressed="${S.oven.view === c}">${chName(c)}</button>`).join('')}</p>
      <label class="control"><span class="control__head">${L('Centro x (píxeles de trabajo)', 'Centre x (working pixels)')} <output>${num(pl.cx, 1)}</output></span><input type="range" id="wr-pcx" min="0" max="${s.w - 1}" step="0.5" value="${pl.cx}"></label>
      <label class="control"><span class="control__head">${L('Centro y (píxeles de trabajo)', 'Centre y (working pixels)')} <output>${num(pl.cy, 1)}</output></span><input type="range" id="wr-pcy" min="0" max="${s.h - 1}" step="0.5" value="${pl.cy}"></label>
      <label class="control"><span class="control__head">${L('Radio (píxeles de trabajo)', 'Radius (working pixels)')} <output>${num(pl.radius, 1)}</output></span><input type="range" id="wr-pr" min="3" max="${Math.round(s.aperture.r)}" step="0.5" value="${pl.radius}"></label>
      <p class="meta">${L('Un píxel de trabajo son 2 × 2 píxeles del sensor. Las coordenadas son de la imagen cargada, que en el ejemplo es un recorte.', 'One working pixel is 2 × 2 sensor pixels. Coordinates are those of the loaded image, a crop in the example.')}</p>
      <p>${btn('wr-preset-pl', L('Volver a la propuesta', 'Back to the proposal'))}</p>
      <p class="lab__readout" id="wr-pl-read" aria-live="polite"></p>
    </div>
    <div class="lab__plots">
      <figure><canvas id="wr-pcv" class="wr-preview" tabindex="0" aria-label="${L('Vista previa de la imagen más brillante sin saturar. Las flechas mueven la meseta, más y menos cambian el radio.', 'Preview of the brightest unsaturated image. Arrows move the plateau, plus and minus change the radius.')}"></canvas>
      <figcaption>${L(`${num(s.previewTexp)} µs. Escala de grises con gamma 2,2 solo para mirar, canal ${chName(S.oven.view)}. Círculo continuo, meseta. Círculo punteado, abertura detectada. Marcas. Píxeles saturados en <span class="wr-mk wr-mk--sat">rojo</span>, píxeles bajo el ruido dentro de la abertura en <span class="wr-mk wr-mk--low">azul</span>, píxeles calientes en <span class="wr-mk wr-mk--hot">amarillo</span>.`, `${num(s.previewTexp)} µs. Grey scale with gamma 2.2 for viewing only, ${chName(S.oven.view)} channel. Solid circle, plateau. Dotted circle, detected aperture. Marks. Saturated pixels in <span class="wr-mk wr-mk--sat">red</span>, pixels below the noise inside the aperture in <span class="wr-mk wr-mk--low">blue</span>, hot pixels in <span class="wr-mk wr-mk--hot">yellow</span>.`)}</figcaption></figure>
      <figure><div id="wr-profI" class="wr-plot"></div><figcaption>${L('Mediana por anillo de los bloques de 8 × 8 sobre el centro, por canal. Banda gris ±1 %. Línea vertical, radio de la meseta.', 'Ring median of the 8 × 8 blocks over the centre, per channel. Grey band ±1 %. Vertical line, plateau radius.')}</figcaption></figure>
      <figure><div id="wr-profR" class="wr-plot"></div><figcaption>${L('R/G y B/G de los bloques sobre el centro. Banda gris ±0,5 %.', 'R/G and B/G of the blocks over the centre. Grey band ±0.5 %.')}</figcaption></figure>
    </div>
  </div></div>
  <div class="wr-msgs" id="wr-pl-msgs"></div>`;
  drawPlateau();
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
  const prof = s.prof?.profile ?? s.plateau.profile;
  const rho = prof.r.map((v) => v / s.aperture.r);
  const xr = pl.radius / s.aperture.r;
  const vline = (lo: number, hi: number): Series => ({ kind: 'line', x: [xr, xr], y: [lo, hi], cls: 's-muted', dash: true });
  plot('wr-profI', { xlabel: L('r / radio de la abertura', 'r / aperture radius'), ylabel: L('I / I centro', 'I / I centre'), height: 230 }, [
    { kind: 'band', x: [0, 1.2], y: [0.99, 0.99], y2: [1.01, 1.01], cls: 's-muted' },
    ...CH.map((c, k) => ({ kind: 'points' as const, x: rho, y: prof.I[c], cls: chCls[k], label: c })), vline(0.85, 1.05)], { xlim: [0, 1.2], ylim: [0.85, 1.05] });
  plot('wr-profR', { xlabel: L('r / radio de la abertura', 'r / aperture radius'), ylabel: L('razón / centro', 'ratio / centre'), height: 230 }, [
    { kind: 'band', x: [0, 1.2], y: [0.995, 0.995], y2: [1.005, 1.005], cls: 's-muted' },
    { kind: 'points', x: rho, y: prof.RG, cls: 's-chR', label: 'R/G' }, { kind: 'points', x: rho, y: prof.BG, cls: 's-chB', label: 'B/G' }, vline(0.97, 1.03)], { xlim: [0, 1.2], ylim: [0.97, 1.03] });
  const flat = s.prof?.flat ?? { R: s.plateau.flatness.R, G: s.plateau.flatness.G, B: s.plateau.flatness.B, RG: s.plateau.flatRG, BG: s.plateau.flatBG };
  const ok = Math.max(flat.R, flat.G, flat.B) <= 0.0101 && Math.max(flat.RG, flat.BG) <= 0.00505;
  const rd = document.getElementById('wr-pl-read');
  if (rd) rd.innerHTML = L(`Meseta de ${num(pl.radius, 1)} píxeles de trabajo (${num(pl.radius * 2, 0)} del sensor), ${num(xr, 2)} radios de la abertura. Variación de borde a centro ${pct(flat.R, 2)}, ${pct(flat.G, 2)} y ${pct(flat.B, 2)} en R, G y B, R/G ${pct(flat.RG, 2)} y B/G ${pct(flat.BG, 2)}. ${ok ? 'Cumple ±1 % y ±0,5 %.' : 'No cumple ±1 % y ±0,5 %.'}`, `Plateau of ${num(pl.radius, 1)} working pixels (${num(pl.radius * 2, 0)} on the sensor), ${num(xr, 2)} aperture radii. Edge to centre variation ${pct(flat.R, 2)}, ${pct(flat.G, 2)} and ${pct(flat.B, 2)} in R, G and B, R/G ${pct(flat.RG, 2)} and B/G ${pct(flat.BG, 2)}. ${ok ? 'It meets ±1 % and ±0.5 %.' : 'It does not meet ±1 % and ±0.5 %.'}`);
  const msgs: Msg[] = [];
  const off = Math.hypot(pl.cx - s.aperture.cx, pl.cy - s.aperture.cy);
  if (off + pl.radius > s.aperture.r) msgs.push({ code: 'plateau_outside' });
  if (off > 2) msgs.push({ code: 'plateau_off_centre', vars: { a: Math.round(off * 10) / 10 } });
  if (!ok) msgs.push({ code: 'plateau_not_flat', vars: { a: Math.round(1000 * Math.max(flat.R, flat.G, flat.B)) / 10 } });
  const cs = S.oven.series.map((x) => { const it = S.items.find((i) => i.id === x.plateauId); const c = it?.crop ?? { x0: 0, y0: 0 }; const sc = it?.fi?.scale ?? 1; return { tC: x.tC, x: c.x0 + sc * x.aperture.cx, y: c.y0 + sc * x.aperture.cy }; });
  for (let i = 1; i < cs.length; i++) { const d = Math.hypot(cs[i].x - cs[i - 1].x, cs[i].y - cs[i - 1].y); if (d > 3) msgs.push({ code: 'plateau_moved', vars: { a: Math.round(d), b: `${tLabel(cs[i - 1].tC)} ${L('y', 'and')} ${tLabel(cs[i].tC)}` } }); }
  const m = document.getElementById('wr-pl-msgs');
  if (m) m.innerHTML = msgHtml(msgs);
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
    keepFocus(() => { const st = document.getElementById('wr-st1'); if (st) st.innerHTML = stage1Status(); });
    drawPlateau();
    for (const [id, v] of [['wr-pcx', pl.cx], ['wr-pcy', pl.cy], ['wr-pr', pl.radius]] as [string, number][]) { const i = document.getElementById(id) as HTMLInputElement | null; if (i && document.activeElement !== i) i.value = String(v); const o = i?.parentElement?.querySelector('output'); if (o) o.textContent = num(v, 1); }
  }, 180);
}
function renderFlags() {
  const el = document.getElementById('wr-flags');
  if (!el) return;
  if (!S.oven.series.length) { el.innerHTML = `<p class="meta">${L('Aparece al cargar las imágenes del horno.', 'Appears once the oven images are loaded.')}</p>`; return; }
  const msgs: Msg[] = [];
  const worst = S.oven.series.reduce((a, s) => (s.hotCount.reduce((x, y) => x + y, 0) > a.n ? { n: s.hotCount.reduce((x, y) => x + y, 0), area: 3 * s.w * s.h } : a), { n: 0, area: 1 });
  if (worst.n) msgs.push({ code: 'hot_pixels', vars: { n: worst.n, a: num((100 * worst.n) / worst.area, 2) }, level: 'info' });
  const r = S.oven.result;
  const rows: string[][] = [];
  for (const s of S.oven.series) s.texpUs.forEach((t, i) => {
    const sg = s.signals[i];
    const idx = r ? r.fit.signals.findIndex((x) => x.tC === s.tC && x.texpUs === t) : -1;
    const enter = CH.map((c) => (idx >= 0 ? (r!.fit.accepted[c][idx] ? L('sí', 'yes') : `<em>${L('no', 'no')}</em>`) : '–')).join(' / ');
    CH.forEach((c, k) => { if (sg.nSat[c] > 0) msgs.push({ code: 'saturated', vars: { channel: k }, files: [`${tLabel(s.tC)}, ${num(t)} µs`] }); });
    if (t < 100) msgs.push({ code: 'short_exposure', vars: { a: t } });
    rows.push([tLabel(s.tC), num(t), CH.map((c) => pct(sg.S[c] / S.oven.ceil12, 0)).join(' / '), CH.map((c) => pct(sg.p999[c] / S.oven.ceil12, 0)).join(' / '), CH.map((c) => num(sg.nSat[c])).join(' / '), enter]);
  });
  el.innerHTML = `<p>${L('Marcas. Píxeles saturados (0,995 del techo o más en cuentas crudas) en rojo, píxeles bajo el ruido (S − D menor que 5 σ del oscuro) en azul, píxeles calientes en amarillo y contados aparte. Un píxel es caliente si pasa en más de 5 σ la mediana de sus vecinos de 5 × 5 en los tres tiempos de exposición revisados, con σ medido en la misma zona. Se reemplaza por esa mediana antes de la meseta, así que no la sesga. En el RGB de pylon las marcas de saturados se dilatan 3 × 3 en los tres canales. Un canal de una imagen se usa en el ajuste si su percentil 99,9 en la meseta está bajo 90 % de la escala, no tiene saturados y su señal pasa 10 veces el ruido. Los otros canales de esa imagen se usan igual.', 'Marks. Saturated pixels (0.995 of the ceiling or more in raw counts) in red, pixels below the noise (S − D under 5 σ of the dark) in blue, hot pixels in yellow and counted apart. A pixel is hot when it exceeds the median of its 5 × 5 neighbours by more than 5 σ in the three exposure times checked, with σ measured in the same region. It is replaced by that median before the plateau, so it does not bias it. In pylon RGB the saturation marks are dilated 3 × 3 in the three channels. A channel of an image is used in the fit when its 99.9th percentile in the plateau is below 90 % of full scale, it has no saturated pixels and its signal is over 10 times the noise. The other channels of that image are still used.')}</p>
  <details class="wr-filetable"${rows.length > 14 ? '' : ' open'}><summary>${L(`Imágenes del horno (${rows.length})`, `Oven images (${rows.length})`)}</summary>
  ${tableHtml([L('Serie', 'Series'), 't<sub>exp</sub> (µs)', L('Mediana en la meseta', 'Plateau median') + ' R / G / B', L('Percentil 99,9', '99.9th percentile') + ' R / G / B', L('Saturados en la meseta', 'Saturated in the plateau') + ' R / G / B', L('Entra al ajuste', 'Enters the fit') + ' R / G / B'], rows)}</details>
  <div class="wr-msgs">${msgHtml(msgs)}</div>`;
}
/** How the tail points were read, the reader of calibration.py (parity with the talk after G5). */
function readLine(r: OvenResult): string {
  const rd = r.tailRead;
  if (!rd) return '';
  if (rd.how !== 'calibration.py') return `<p class="meta" id="wr-tail-read">${L('Puntos leídos con la mediana de la meseta de 1a.3, porque la segunda lectura de las imágenes no se pudo hacer. Las cifras pueden diferir de las de la charla.', 'Points read with the plateau median of 1a.3, because the second reading of the images could not be done. The figures may differ from those of the talk.')}</p>`;
  const g = rd.geo.map((x) => `${tLabel(x.tC)} ${num(x.plateauR * rd.scale)}`).join(', ');
  return `<p class="meta" id="wr-tail-read">${L(`Lectura igual a la de la charla (tools/calibration.py). Cada imagen se lee de nuevo en la grilla del sensor, sin agrupar 2 × 2: media de todos los píxeles de un disco de ${num(rd.R)} px de radio, centrado en la abertura de cada temperatura, y un canal cuenta como saturado solo si un píxel suyo del disco llega al techo. El radio es el menor de las ${rd.geo.length} mesetas a ±3 % (${g} px), no el disco de 1a.3.`, `Same reading as the talk (tools/calibration.py). Each image is read again on the sensor grid, without 2 × 2 grouping: mean of every pixel of a disc of radius ${num(rd.R)} px, centred on the aperture of each temperature, and a channel counts as saturated only if one of its own pixels in the disc reaches the ceiling. The radius is the smallest of the ${rd.geo.length} plateaus within ±3 % (${g} px), not the disc of 1a.3.`)}</p>`;
}
/**
 * IR tail calibration of ψ_R with the oven, under the ratio test of 1a.7 (professor after G5). On by default
 * with three or more temperatures, labelled, with the ratio test before and after and its own notice.
 */
function tailBlock(r: OvenResult): string {
  const t: TailCal | null = r.tail;
  const head = `<h4 id="wr-tail-h">${L('Calibración de la cola infrarroja con el horno', 'Calibration of the infrared tail with the oven')}</h4>`;
  if (!t) return '';
  if (!t.ok) return `<div id="wr-tail">${head}<div class="wr-msgs">${msgHtml([{ code: 'tail_few', level: 'info' }])}</div></div>`;
  const on = S.tailOn;
  const msgs: Msg[] = [];
  const range = `${tLabel(t.temps[0])} ${L('y', 'and')} ${tLabel(t.temps[t.temps.length - 1])}`;
  if (!on) msgs.push({ code: 'tail_off', level: 'info' });
  else if (t.criterion.pass) {
    msgs.push({ code: 'tail_ok', level: 'info', vars: { a: num(t.KRKG, 4), b: num(t.u.flameK), c: S.o.tailBias ? (getLang() === 'es' ? S.o.tailBias.es : S.o.tailBias.en) : '–' } });
    msgs.push({ code: 'tail_blue', level: 'info', vars: { a: sgn(t.driftNominalByChannelPct.B, 0) + ' %' } });
  } else msgs.push({ code: 'tail_fail', vars: { a: num(t.criterion.worstPct, 1) } });
  if (t.temps.length === 3) msgs.push({ code: 'tail_three', level: 'info' });
  const rows = t.temps.map((tC) => {
    const l = t.loo.find((x) => x.tC === tC)!, q = t.ratio.find((x) => x.tC === tC)!;
    const crit = t.criterion.from.includes(tC);
    const b = (x: string) => (crit ? `<strong>${x}</strong>` : x);
    return [tLabel(tC), `${sgn(l.KRKGrelNominalPct, 1)}&nbsp;%`, b(`${sgn(l.KRKGrelPct, 2)}&nbsp;%`), sgn(q.dTflameNominalK), b(sgn(l.dToutK, 1))];
  });
  return `<div id="wr-tail">${head}
  ${chk('wr-tail', on, L('Calibrar la cola de ψ<sub>R</sub> sobre 700 nm con el horno', 'Calibrate the tail of ψ<sub>R</sub> above 700 nm with the oven'), L('Por defecto con tres o más temperaturas del horno. Apagada, K<sub>j</sub> sale de las curvas nominales.', 'On by default with three or more oven temperatures. Off, K<sub>j</sub> comes from the nominal curves.'))}
  <p>${L('La curva de la cámara sobre 700 nm es poco segura, y el horno, más frío que una llama, recibe mucha señal de esa parte. La herramienta busca un factor s<sub>R</sub> que multiplica ψ<sub>R</sub> sobre 700 nm (1 es la curva nominal, ψ<sub>G</sub> y ψ<sub>B</sub> quedan nominales) hasta que un solo K<sub>R</sub>/K<sub>G</sub> explica la razón R/G de todas las temperaturas. Usa las imágenes con rojo y verde entre 5 y 90 % de la escala, y cada temperatura pesa lo mismo. Después repite la prueba de la razón dejando fuera una temperatura cada vez.',
    'The camera curve above 700 nm is uncertain, and the oven, colder than a flame, gets much of its signal from that part. The tool looks for a factor s<sub>R</sub> that multiplies ψ<sub>R</sub> above 700 nm (1 is the nominal curve, ψ<sub>G</sub> and ψ<sub>B</sub> stay nominal) until a single K<sub>R</sub>/K<sub>G</sub> explains the R/G ratio at every temperature. It uses the images with red and green between 5 and 90 % of the scale, and every temperature weighs the same. Then it repeats the ratio test leaving out one temperature at a time.')}</p>
  <p id="wr-tail-fit">${L(`s<sub>R</sub> = ${num(t.s, 3)} ± ${num(t.u.s, 3)} con ψ<sub>G</sub> nominal, y K<sub>R</sub>/K<sub>G</sub> = ${num(t.KRKG, 4)}, de ${t.nPairs} imágenes en ${t.temps.length} temperaturas. La incertidumbre junta la del ajuste, ±${num(S.sigmaT)} K del termopar en cada temperatura y +${num(S.sigmaT)} K común. La deriva de K<sub>R</sub>/K<sub>G</sub> entre ${range} pasa de ${sgn(t.driftNominalPct, 1)} % a ${sgn(t.driftPct, 2)} %. Es el residuo del mismo ajuste y no cuenta como prueba.`,
    `s<sub>R</sub> = ${num(t.s, 3)} ± ${num(t.u.s, 3)} with nominal ψ<sub>G</sub>, and K<sub>R</sub>/K<sub>G</sub> = ${num(t.KRKG, 4)}, from ${t.nPairs} images at ${t.temps.length} temperatures. The uncertainty combines the fit, ±${num(S.sigmaT)} K of the thermocouple at each temperature and +${num(S.sigmaT)} K common to all. The drift of K<sub>R</sub>/K<sub>G</sub> between ${range} goes from ${sgn(t.driftNominalPct, 1)} % to ${sgn(t.driftPct, 2)} %. It is the residual of the same fit and does not count as a test.`)}</p>
  ${readLine(r)}
  <h4 id="wr-tail-test-h">${L('Prueba de la razón antes y después, con una temperatura fuera', 'Ratio test before and after, with one temperature left out')}</h4>
  ${tableHtml([L('Temperatura fuera', 'Temperature left out'), L('K<sub>R</sub>/K<sub>G</sub> de esa temperatura sobre el ajuste de las otras, antes', 'K<sub>R</sub>/K<sub>G</sub> of that temperature over the fit of the others, before'), L('después', 'after'), L('ΔT en una llama a 1800 K, antes (K)', 'ΔT in a flame at 1800 K, before (K)'), L('después (K)', 'after (K)')], rows, 'wr-tail-table')}
  <p class="meta">${L('Antes, con las curvas nominales. En ΔT antes usa K<sub>R</sub>/K<sub>G</sub> de la temperatura más alta, como la tabla de arriba. Después, s<sub>R</sub> y K<sub>R</sub>/K<sub>G</sub> salen de las otras temperaturas, y K<sub>R</sub>/K<sub>G</sub> medido en la que quedó fuera se lleva a la llama.', 'Before, with the nominal curves. In ΔT before it uses K<sub>R</sub>/K<sub>G</sub> of the highest temperature, as in the table above. After, s<sub>R</sub> and K<sub>R</sub>/K<sub>G</sub> come from the other temperatures, and K<sub>R</sub>/K<sub>G</sub> measured at the one left out is carried to the flame.')}</p>
  <p id="wr-tail-crit">${L(`Criterio de aceptación, K<sub>R</sub>/K<sub>G</sub> de la temperatura dejada fuera a menos de ${num(t.criterion.limitPct)} % del ajuste de las otras${t.temps.length === 3 ? `, dejando fuera ${tLabel(t.criterion.strongT)}` : ''}. Da ${num(t.criterion.worstPct, 2)} % (antes ${num(t.criterion.nominalPct, 1)} %), <strong>${t.criterion.pass ? 'pasa' : 'no pasa'}</strong>. La prueba más exigente es dejar fuera ${tLabel(t.criterion.strongT)}, la más fría, que da ${sgn(t.criterion.strongPct, 2)} %. En la llama, el peor ΔT pasa de ${num(t.ratioTest.beforeK)} K a ${num(t.ratioTest.afterK)} K.`,
    `Acceptance criterion, K<sub>R</sub>/K<sub>G</sub> of the temperature left out within ${num(t.criterion.limitPct)} % of the fit of the others${t.temps.length === 3 ? `, leaving out ${tLabel(t.criterion.strongT)}` : ''}. It gives ${num(t.criterion.worstPct, 2)} % (before ${num(t.criterion.nominalPct, 1)} %), <strong>${t.criterion.pass ? 'passes' : 'fails'}</strong>. The hardest test is leaving out ${tLabel(t.criterion.strongT)}, the coldest, which gives ${sgn(t.criterion.strongPct, 2)} %. In the flame, the worst ΔT goes from ${num(t.ratioTest.beforeK)} K to ${num(t.ratioTest.afterK)} K.`)}</p>
  <div class="wr-msgs">${msgHtml(msgs)}</div></div>`;
}
function renderFit() {
  const r = S.oven.result;
  const fe = document.getElementById('wr-fit'), le = document.getElementById('wr-lin'), re = document.getElementById('wr-ratio');
  if (!r || !fe || !le || !re) return;
  const f = r.fit;
  const msgs: Msg[] = [];
  const unread = S.items.filter((x) => x.role === 'oven' && (x.tC === undefined || x.texpUs === undefined));
  if (unread.length) msgs.push({ code: 'oven_fit_missing', vars: { n: unread.length }, files: unread.map((x) => x.path), level: 'warn' });
  // said in full once, in stage 2 where the decision is made (GB3r3-adv-07)
  const calOk = S.cal?.mode === 'oven' ? S.cal.consistent && S.cal.tested !== false : f.ratio.RG.tested && f.consistent;
  if (!calOk) msgs.push({ code: 'see_stage2', level: 'info' });
  if (f.calibrated.length === 1) msgs.push({ code: 'cal_one_channel', vars: { channel: CH.indexOf(f.calibrated[0]) } });
  // one notice for every channel and temperature (GB3r1-estudiante-11, fisico-20)
  // grouped per channel, and in one phrase when it is every channel at every temperature (GB3r3-adv-07, estudiante-02)
  const hitsBy = CH.map((c) => r.tails.filter((t) => t[c] > 0.1).map((t) => Math.round(t.tC + T0C)));
  const nHits = hitsBy.reduce((a, h) => a + h.length, 0);
  const same = hitsBy.every((h) => h.join() === hitsBy[0].join());
  const tailText = nHits === 3 * r.tails.length && r.tails.length > 1 ? L('los tres canales a todas las temperaturas', 'the three channels at every temperature')
    : same ? L(`los tres canales a ${hitsBy[0].map((x) => num(x)).join(', ')} K`, `the three channels at ${hitsBy[0].map((x) => num(x)).join(', ')} K`)
    : hitsBy.map((h, k) => (h.length ? `${chName(k)} ${L('a', 'at')} ${h.map((x) => num(x)).join(L(', ', ', '))} K` : '')).filter(Boolean).join(L(' y ', ' and '));
  if (nHits) msgs.push({ code: r.ratioWorst > 50 || !f.ratio.RG.tested ? 'ir_tail' : 'ir_tail_ok', vars: { a: tailText }, level: 'info' });
  const tempsAll = [...new Set(f.channels.R.perT.map((p) => p.tC))];
  const rows = tempsAll.map((tC) => [tLabel(tC), ...CH.map((c) => { const p = f.channels[c].perT.find((x) => x.tC === tC); return p ? `${sig(p.K)} ± ${sig(p.sK, 2)}` : '–'; }), CH.map((c) => pct(r.tails.find((t) => t.tC === tC)?.[c] ?? NaN, 0)).join(' / ')]);
  const unitK = L('DN s⁻¹ por W m⁻² sr⁻¹', 'DN s⁻¹ per W m⁻² sr⁻¹');
  const Tcal = f.channels.R.Tcal;
  fe.innerHTML = `
  <p>${L(`Por canal, la pendiente de S − D contra t<sub>exp</sub> a cada temperatura (mediana en la meseta, en DN s⁻¹) sobre ∫ ψ<sub>j</sub> ε I<sub>bλ</sub>(T) dλ, así que K<sub>j</sub> = (S<sub>j</sub> − D<sub>j</sub>) / (t<sub>exp</sub> ∫ ψ<sub>j</sub> ε I<sub>bλ</sub> dλ). K<sub>j</sub> sale de la temperatura más alta con datos sin saturar${fin(Tcal) ? `, ${tLabel(Tcal - T0C)}` : ''}, la más cercana al espectro de la llama, y las otras temperaturas son la prueba. El error del termopar entra como un desplazamiento común de ±${num(S.sigmaT)} K.${S.cal?.tail?.applied ? ' Con la cola infrarroja calibrada (1a.7), K<sub>G</sub> queda igual y K<sub>R</sub> = K<sub>G</sub> · K<sub>R</sub>/K<sub>G</sub> del ajuste de la cola.' : ''}`, `Per channel, the slope of S − D against t<sub>exp</sub> at each temperature (plateau median, in DN s⁻¹) over ∫ ψ<sub>j</sub> ε I<sub>bλ</sub>(T) dλ, so K<sub>j</sub> = (S<sub>j</sub> − D<sub>j</sub>) / (t<sub>exp</sub> ∫ ψ<sub>j</sub> ε I<sub>bλ</sub> dλ). K<sub>j</sub> comes from the highest temperature with unsaturated data${fin(Tcal) ? `, ${tLabel(Tcal - T0C)}` : ''}, the closest to the flame spectrum, and the other temperatures are the test. The thermocouple error enters as a common shift of ±${num(S.sigmaT)} K.${S.cal?.tail?.applied ? ' With the infrared tail calibrated (1a.7), K<sub>G</sub> stays the same and K<sub>R</sub> = K<sub>G</sub> · K<sub>R</sub>/K<sub>G</sub> of the tail fit.' : ''}`)}</p>
  ${tableHtml([L('Temperatura', 'Temperature'), `K<sub>R</sub>`, `K<sub>G</sub>`, `K<sub>B</sub>`, L('Fracción sobre 700 nm', 'Fraction above 700 nm') + ' R / G / B'], rows)}
  <p class="meta">${L('K de cada temperatura por separado, con su σ. La última columna es la parte de la señal de ese canal que viene de longitudes de onda sobre 700 nm, según la curva de la cámara.', 'K of each temperature on its own, with its σ. The last column is the share of the signal of that channel that comes from wavelengths above 700 nm, according to the camera curve.')}${f.channels.R.perT.length > 1 ? L(` La deriva de K<sub>R</sub>/K<sub>G</sub> entre ${tLabel(tempsAll[0])} y ${tLabel(tempsAll[tempsAll.length - 1])} es ${pct(krkgDrift(f), 1)}.`, ` The drift of K<sub>R</sub>/K<sub>G</sub> between ${tLabel(tempsAll[0])} and ${tLabel(tempsAll[tempsAll.length - 1])} is ${pct(krkgDrift(f), 1)}.`) : ''}</p>
  ${tableHtml(['', 'K', L('σ estadística', 'statistical σ'), L('σ por el termopar', 'σ from the thermocouple'), L('K con la respuesta sobre 700 nm × 0 y × 4', 'K with the response above 700 nm × 0 and × 4')], CH.map((c) => { const ch = f.channels[c]; return [`K<sub>${c}</sub> (${unitK})`, sig(ch.K, 4), pct(ch.sKstat / ch.K, 2), pct(ch.sKT / ch.K, 2), `${pct(ch.tailLo, 1)} / ${pct(ch.tailHi, 1)}`]; }))}
  <p class="meta">${L('La última columna dice cuánto cambiaría K si la respuesta de la cámara sobre 700 nm fuera 0 veces o 4 veces la nominal.', 'The last column says how much K would change if the camera response above 700 nm were 0 times or 4 times the nominal one.')}</p>
  ${f.ratio.RG.tested ? '' : '<!-- one temperature, no K(T) plot -->'}<figure${f.ratio.RG.tested ? '' : ' hidden'}><div id="wr-kt" class="wr-plot"></div><figcaption>${L('K<sub>j</sub>(T) de cada temperatura sobre el K<sub>j</sub> usado, con su barra. Si un solo K por canal explicara el horno, los puntos quedarían en 1.', 'K<sub>j</sub>(T) at each temperature over the K<sub>j</sub> used, with its bar. If one K per channel explained the oven, the points would sit at 1.')}</figcaption></figure>
  <p>${L(`Criterio de consistencia. La dispersión de ln(K<sub>R</sub>/K<sub>G</sub>) entre temperaturas debe ser menor que la que dan ±σ<sub>T</sub> y la estadística. ${f.ratio.RG.tested ? '' : 'Con una sola temperatura no hay dispersión que medir. '}Su desviación estándar da ${f.ratio.RG.tested ? num(f.ratio.RG.dispersion * 100, 2) : '–'} % contra ${num(f.ratio.RG.allowed * 100, 2)} % permitido para R/G y ${num(f.ratio.BG.dispersion * 100, 2)} % contra ${num(f.ratio.BG.allowed * 100, 2)} % para B/G. Es otra medida del mismo defecto que la deriva de punta a punta de arriba.`, `Consistency criterion. The scatter of ln(K<sub>R</sub>/K<sub>G</sub>) across temperatures must be below what ±σ<sub>T</sub> and the statistics allow. ${f.ratio.RG.tested ? '' : 'With a single temperature there is no scatter to measure. '}Its standard deviation is ${f.ratio.RG.tested ? num(f.ratio.RG.dispersion * 100, 2) : '–'} % against ${num(f.ratio.RG.allowed * 100, 2)} % allowed for R/G and ${num(f.ratio.BG.dispersion * 100, 2)} % against ${num(f.ratio.BG.allowed * 100, 2)} % for B/G. It measures the same defect as the end-to-end drift above in another way.`)} <em>${!f.ratio.RG.tested ? L('Con una sola temperatura no se puede probar.', 'It cannot be tested with a single temperature.') : f.consistent ? L(`Consistente, canales ${f.calibrated.join(', ')}.`, `Consistent, channels ${f.calibrated.join(', ')}.`) : L('No consistente.', 'Inconsistent.')}</em></p>
  <div class="wr-msgs">${msgHtml(msgs)}</div>`;
  const kt: Series[] = [];
  CH.forEach((c, k) => {
    const ch = f.channels[c];
    kt.push({ kind: 'points', x: ch.perT.map((p) => p.tC), y: ch.perT.map((p) => p.K / ch.K), cls: chCls[k], label: c });
    ch.perT.forEach((p) => kt.push({ kind: 'line', x: [p.tC, p.tC], y: [(p.K - (fin(p.sK) ? p.sK : 0)) / ch.K, (p.K + (fin(p.sK) ? p.sK : 0)) / ch.K], cls: chCls[k] }));
  });
  if (f.ratio.RG.tested) plot('wr-kt', { xlabel: L('T del horno (°C)', 'Oven T (°C)'), ylabel: 'Kⱼ(T) / Kⱼ', height: 230 }, kt);
  // linearity of the current series
  const s = S.oven.series[S.oven.cur];
  const lmsgs: Msg[] = [];
  const lin: Series[] = [];
  if (s) {
    const sigs = f.signals.filter((x) => x.tC === s.tC).sort((a, b) => a.texpUs - b.texpUs);
    CH.forEach((c, k) => {
      const pt = f.channels[c].perT.find((p) => p.tC === s.tC);
      lin.push({ kind: 'points', x: sigs.map((x) => x.texpUs / 1000), y: sigs.map((x) => x.S[c]), cls: chCls[k], label: c });
      if (pt) { const tm = sigs[sigs.length - 1].texpUs * 1e-6; lin.push({ kind: 'line', x: [0, tm * 1000], y: [fin(pt.a) ? pt.a : 0, (fin(pt.a) ? pt.a : 0) + pt.b * tm], cls: chCls[k] }); if (pt.linMaxPct > 1) lmsgs.push({ code: 'nonlinear', vars: { channel: k, a: Math.round(pt.linMaxPct * 10) / 10 } }); }
    });
  }
  le.innerHTML = `${seriesTabs()}<figure><div id="wr-linp" class="wr-plot"></div><figcaption>${L('S<sub>j</sub> en la meseta contra t<sub>exp</sub> de la serie elegida, con la recta del ajuste. Los puntos bajo 100 µs y sobre el 90 % de la escala no entran.', 'S<sub>j</sub> in the plateau against t<sub>exp</sub> for the chosen series, with the fitted line. Points under 100 µs and above 90 % of full scale do not enter.')}</figcaption></figure>
  <p class="meta">${s ? CH.map((c) => { const p = f.channels[c].perT.find((x) => x.tC === s.tC); return `${c} ${p && fin(p.linMaxPct) ? pct(p.linMaxPct / 100, 2) : '–'}`; }).join(', ') : ''} ${L('de desviación máxima de la recta.', 'maximum departure from the line.')}</p><div class="wr-msgs">${msgHtml(lmsgs)}</div>`;
  plot('wr-linp', { xlabel: 't_exp (ms)', ylabel: 'S (DN)', height: 230 }, [...lin, { kind: 'line', x: [0, (s?.texpUs[s.texpUs.length - 1] ?? 1) / 1000], y: [0.9 * S.oven.ceil12, 0.9 * S.oven.ceil12], cls: 's-muted', dash: true }]);
  // ratio test, always visible with its warning
  const rmsgs: Msg[] = [];
  // the pointer to stage 2 is already under the fit table of this stage (GB3r3-adv-07)
  if (!f.order.ok) rmsgs.push({ code: 'bayer_order' });
  // the warning sits under its own test, so the test and its verdict read together (G5 director decision)
  if (r.ratioWorst > 50) rmsgs.push({ code: 'ratio_test_fail', vars: { a: Math.round(r.ratioWorst), b: ovenBias() } });
  const rowsTop = f.ratioTestTop.map((x) => { const e = r.fitEnergy?.ratioTestTop.find((y) => y.tC === x.tC); return [tLabel(x.tC), num(x.TK), num(x.Tpred), `<strong>${sgn(x.dT)}</strong>`, e ? sgn(e.dT) : '–']; });
  // the calibration temperature itself, ΔT = 0 by construction (GB3r3-estudiante-04)
  const tcal = f.channels.R.Tcal;
  if (fin(tcal) && !f.ratioTestTop.some((x) => Math.abs(x.tC + T0C - tcal) < 0.5)) rowsTop.push([tLabel(tcal - T0C), num(tcal), num(tcal), L('0, referencia', '0, reference'), L('0, referencia', '0, reference')]);
  const rowsLoo = f.ratioTest.map((x) => { const e = r.fitEnergy?.ratioTest.find((y) => y.tC === x.tC); return [tLabel(x.tC), num(x.TK), num(x.Tpred), sgn(x.dT), e ? sgn(e.dT) : '–']; });
  const other = S.basis === 'photon' ? L('leyendo la curva por energía', 'reading the curve per energy') : L('leyendo la curva por fotón', 'reading the curve per photon');
  const tK = tempsAll.map((x) => Math.round(x + T0C)).sort((a, b) => a - b);
  // with one temperature there is no test, one line instead of empty tables (GB3r3-adv-11)
  if (tK.length < 2 || !f.ratio.RG.tested) { re.innerHTML = `<p>${L('Con una sola temperatura no hay prueba de la razón. Carguen al menos dos.', 'With a single temperature there is no ratio test. Load at least two.')}</p><div class="wr-msgs">${msgHtml(rmsgs.filter((m) => m.code !== 'see_stage2'))}</div>${tailBlock(r)}`; return; }
  const tRange = `entre ${num(tK[0])} y ${num(tK[tK.length - 1])} K`;
  const tRangeEn = `between ${num(tK[0])} and ${num(tK[tK.length - 1])} K`;
  re.innerHTML = `<p>${L(`Para cada temperatura, la temperatura de razón R/G del cuerpo negro gris (tabla de la razón G<sub>R</sub>/G<sub>G</sub> con E(m)/λ constante) contra la del termopar. Prueba la forma de ψ<sub>j</sub> con T ${tRange}, no a temperatura de llama. La columna de la derecha repite la prueba ${S.basis === 'photon' ? 'leyendo la curva de la cámara por energía en vez de por fotón' : 'leyendo la curva de la cámara por fotón en vez de por energía'} (etapa 1.0).`, `For each temperature, the R/G ratio temperature of the grey blackbody (table of the ratio G<sub>R</sub>/G<sub>G</sub> with E(m)/λ constant) against the thermocouple. It tests the shape of ψ<sub>j</sub> with T ${tRangeEn}, not at flame temperature. The right column repeats the test ${S.basis === 'photon' ? 'reading the camera curve per energy instead of per photon' : 'reading the camera curve per photon instead of per energy'} (stage 1.0).`)}</p>
  <h4>${L('Con K<sub>R</sub>/K<sub>G</sub> de la temperatura más alta', 'With K<sub>R</sub>/K<sub>G</sub> from the highest temperature')}</h4>
  ${tableHtml([L('Horno', 'Oven'), L('T termopar (K)', 'Thermocouple T (K)'), L('T de la razón (K)', 'Ratio T (K)'), 'ΔT (K)', `ΔT ${other} (K)`], rowsTop)}
  <p class="meta">${L('ΔT es la T de la razón menos la del termopar. Un ΔT positivo quiere decir que la razón da más que el termopar.', 'ΔT is the ratio T minus the thermocouple T. A positive ΔT means the ratio reads higher than the thermocouple.')}</p>
  <h4>${L('Dejando fuera cada temperatura', 'Leaving each temperature out')}</h4>
  <p class="meta">${L('Para cada temperatura, K se calcula sin esa temperatura y se prueba con ella. Cada fila usa otro K. El error cambia de signo al cruzar la temperatura de calibración, positivo en las temperaturas bajo ella y negativo sobre ella. Por eso una llama, más caliente que el horno, sale más fría que la real.', 'For each temperature, K is computed without that temperature and tested on it. Each row uses another K. The error changes sign across the calibration temperature, positive at temperatures below it and negative above it. That is why a flame, hotter than the oven, comes out colder than it is.')}</p>
  ${tableHtml([L('Horno', 'Oven'), L('T termopar (K)', 'Thermocouple T (K)'), L('T de la razón (K)', 'Ratio T (K)'), 'ΔT (K)', `ΔT ${other} (K)`], rowsLoo)}
  <p class="meta">${f.order.ok ? L('Orden de canales en el horno, rojo mayor que verde mayor que azul en todas las temperaturas.', 'Channel order at the oven, red above green above blue at every temperature.') : L(`Orden propuesto ${f.order.proposal.join(' > ')}.`, `Proposed order ${f.order.proposal.join(' > ')}.`)}</p>
  <div class="wr-msgs">${msgHtml(rmsgs)}</div>${tailBlock(r)}`;
}
function renderFlat() {
  const el = document.getElementById('wr-flat');
  if (!el) return;
  const flats = S.items.filter((it) => it.role === 'flat');
  const v = S.flat.vignette;
  el.innerHTML = `<p>${L('Imagen de esfera o difusor que llena el cuadro, con su oscuro, varios cuadros promediados. El mapa de viñeteo se normaliza al centro de la meseta del horno, se agrupa en bloques de 8 × 8 y fuera de la zona iluminada sigue el ajuste radial a + bρ² + cρ⁴. K<sub>j</sub>(x, y) es el K<sub>j</sub> de la meseta por ese mapa.', 'Image of a sphere or diffuser filling the frame, with its dark, several frames averaged. The vignette map is normalised to the centre of the oven plateau, grouped in 8 × 8 blocks and outside the lit region it follows the radial fit a + bρ² + cρ⁴. K<sub>j</sub>(x, y) is the plateau K<sub>j</sub> times that map.')}</p>
  ${flats.length ? `<p>${btn('wr-flat-build', L(`Construir el mapa con ${flats.length} ${flats.length === 1 ? 'imagen' : 'imágenes'}`, `Build the map from ${flats.length} ${flats.length === 1 ? 'image' : 'images'}`))}</p>` : msgHtml([{ code: 'flat_missing', level: 'info' }])}
  ${v ? `<div class="wr-grid2"><figure><canvas id="wr-vig" class="wr-preview"></canvas><figcaption>${L('Viñeteo del verde, K(x, y)/K centro de 0,8 (negro) a 1,05 (blanco).', 'Green vignetting, K(x, y)/K centre from 0.8 (black) to 1.05 (white).')}</figcaption></figure><figure><div id="wr-vigp" class="wr-plot"></div><figcaption>${L('Perfil horizontal por el centro, por canal.', 'Horizontal profile through the centre, per channel.')}</figcaption></figure></div>` : ''}`;
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
  const units: [string, string][] = [['radiance_nm', 'W m⁻² sr⁻¹ nm⁻¹'], ['radiance_um', 'W m⁻² sr⁻¹ µm⁻¹'], ['irradiance_nm', L('W m⁻² nm⁻¹ (irradiancia, ÷ π)', 'W m⁻² nm⁻¹ (irradiance, ÷ π)')], ['irradiance_um', L('W m⁻² µm⁻¹ (irradiancia, ÷ π)', 'W m⁻² µm⁻¹ (irradiance, ÷ π)')], ['uW_cm2_nm', 'µW cm⁻² nm⁻¹'], ['counts', L('cuentas relativas', 'relative counts')]];
  let res = '';
  if (r && r.K) {
    const K = r.K as Rgb<number>, rsd = r.rsd as Rgb<number>, near = r.near as Rgb<number>, mk = r.mask as { nDisc: number; nKept: number; lostSaturated: Rgb<number>; lostDark: Rgb<number>; medianDn: Rgb<number> };
    const msgs: Msg[] = [];
    CH.forEach((c, k) => { if (near[c] > 0.05) msgs.push({ code: 'sphere_near_ceiling', vars: { channel: k, a: Math.round(near[c] * 1000) / 10 } }); });
    for (const c of (r.short as Ch[])) msgs.push({ code: 'sphere_short', vars: { channel: CH.indexOf(c) } });
    res = `${tableHtml([L('Canal', 'Channel'), L('Mediana en el disco (DN)', 'Disc median (DN)'), L('Perdidos por saturados', 'Lost as saturated'), L('Perdidos bajo 12,5 DN', 'Lost under 12.5 DN'), L('Disco sobre 90 % de la escala', 'Disc above 90 % of full scale'), `K<sub>j</sub> (DN s⁻¹ ${L('por', 'per')} W m⁻² sr⁻¹)`, L('Dispersión en el disco', 'Scatter in the disc'), 'τ̄ ND'], CH.map((c) => [near[c] > 0.05 ? `${c} <em>${L('dudoso', 'doubtful')}</em>` : c, num(mk.medianDn[c], 1), num(mk.lostSaturated[c]), num(mk.lostDark[c]), pct(near[c], 1), sig(K[c], 4), pct(rsd[c], 1), num((r.tauBar as Rgb<number>)[c], 4)]))}
    <p class="meta">${L(`Disco de ${num((r.disc as { rFit: number }).rFit, 0)} píxeles de trabajo de radio, ${num(mk.nKept)} de ${num(mk.nDisc)} píxeles usados. Temperatura de color de la lámpara ${num(r.Tlamp as number)} K. Fuera del disco K sigue el relleno ${S.sphere.fill === 'median' ? 'por la mediana' : 'radial'}, que es un modelo.`, `Disc of ${num((r.disc as { rFit: number }).rFit, 0)} working pixels radius, ${num(mk.nKept)} of ${num(mk.nDisc)} pixels used. Lamp colour temperature ${num(r.Tlamp as number)} K. Outside the disc K follows the ${S.sphere.fill === 'median' ? 'median' : 'radial'} fill, which is a model.`)} ${L('Los perdidos son píxeles del disco que no entran por saturados o por tener menos de 12,5 DN sobre el oscuro. τ̄ ND es la transmisión media del filtro neutro ponderada con la respuesta de cada canal.', 'The lost pixels are disc pixels left out as saturated or for having less than 12.5 DN above the dark. τ̄ ND is the mean transmission of the neutral density filter weighted by the response of each channel.')}</p><div class="wr-msgs">${msgHtml(msgs)}</div>`;
  } else if (r && r.msgs) res = `<div class="wr-msgs">${msgHtml(r.msgs as Msg[])}</div>`;
  el.innerHTML = `
  <h3>1b · ${L('Esfera integradora', 'Integrating sphere')}</h3>
  <p>${L('Es el mismo cálculo de la aplicación de calibración del grupo. K<sub>j</sub> = (S − D) / (t<sub>exp</sub> τ̄<sub>j</sub> ∫ ψ<sub>j</sub> L<sub>esfera</sub> dλ) en cada píxel del disco. Carguen las imágenes en la tabla de 1a.1 con el rol esfera y su oscuro dentro de la carpeta esfera, y el espectro CSV.', 'It is the same calculation as the group’s calibration app. K<sub>j</sub> = (S − D) / (t<sub>exp</sub> τ̄<sub>j</sub> ∫ ψ<sub>j</sub> L<sub>sphere</sub> dλ) at every pixel of the disc. Load the images in the 1a.1 table with the sphere role and their dark inside the sphere folder, and the spectrum CSV.')}</p>
  <div class="wr-drop" id="wr-drop"><p class="cluster">${fileBtn('wr-o-dir', L('Elegir una carpeta', 'Choose a folder'), '', true)} ${fileBtn('wr-o-files', L('Elegir archivos o un .zip', 'Choose files or a .zip'), '.tif,.tiff,.zip,.csv,.pfs,.json')}</p></div>
  ${fileTable((it) => it.role === 'sphere' || (it.role === 'dark' && /(esfera|sphere)/i.test(it.path)), 'wr-ft1b')}
  <p class="meta">${L(`${imgs.length} ${imgs.length === 1 ? 'imagen' : 'imágenes'} de esfera y ${darks.length} ${darks.length === 1 ? 'oscuro' : 'oscuros'}.`, `${imgs.length} sphere ${imgs.length === 1 ? 'image' : 'images'} and ${darks.length} ${darks.length === 1 ? 'dark' : 'darks'}.`)} ${S.sphere.spectrumName ? L(`Espectro ${esc(S.sphere.spectrumName)}.`, `Spectrum ${esc(S.sphere.spectrumName)}.`) : ''}</p>
  <div class="wr-grid3">
    ${numIn('wr-tsph', fin(S.sphere.tSphUs) ? S.sphere.tSphUs : imgs[0]?.texpUs ?? NaN, 'min="1" step="any"', L('t<sub>exp</sub> de la esfera (µs)', 'Sphere t<sub>exp</sub> (µs)'))}
    <div>${fileBtn('wr-spec', L('Elegir el espectro CSV', 'Choose the spectrum CSV'), '.csv,.txt,text/csv')}</div>
    ${sel('wr-sunit', units, S.sphere.valUnit, L('Unidad del espectro', 'Spectrum unit'))}
    ${sel('wr-slam', [['nm', 'nm'], ['um', 'µm'], ['A', 'Å']], S.sphere.lamUnit, L('Unidad de λ', 'Unit of λ'))}
    ${sel('wr-nd', [['', L('elijan una opción', 'choose an option')], ['none', L('sin filtro', 'no filter')], ['od', L('densidad óptica', 'optical density')], ['curve', L('curva CSV', 'CSV curve')]], S.sphere.nd, L('Filtro neutro (obligatorio)', 'Neutral density filter (required)'))}
    ${S.sphere.nd === 'od' ? numIn('wr-od', S.sphere.od, 'min="0" max="6" step="0.01"', L('Densidad óptica', 'Optical density')) : S.sphere.nd === 'curve' ? `<div>${fileBtn('wr-ndc', L('Curva de transmisión CSV', 'Transmission CSV'), '.csv,.txt')} <span class="meta">${esc(S.sphere.ndCurveName)}</span></div>` : ''}
    ${sel('wr-fill', [['median', L('mediana', 'median')], ['radial', 'a + bρ² + cρ⁴']], S.sphere.fill, L('Relleno fuera del disco', 'Fill outside the disc'))}
  </div>
  <p>${btn('wr-sph-run', L('Calcular la esfera', 'Compute the sphere'), !imgs.length)}</p>
  ${res}`;
}
function renderCalFile() {
  const el = document.getElementById('wr-calpath');
  if (!el) return;
  const c = S.cal?.mode === 'file' ? S.cal : null;
  el.innerHTML = `<h3>1c · ${L('Archivo de calibración', 'Calibration file')}</h3>
  <p>${L('Una calibración .json guardada antes con esta herramienta, o una de ANNAbel convertida con convertir_calibracion_annabel.py, que viene en el juego de la llama de septiembre. El archivo dice en qué unidad está K. ANNAbel usa DN de 16 bits y la herramienta DN de 12 bits, así que K de ANNAbel se divide por 16. La herramienta recalcula G<sub>j</sub>(T) desde la ψ<sub>j</sub> del archivo y avisa si no coincide.', 'A .json calibration saved earlier with this tool, or an ANNAbel one converted with convertir_calibracion_annabel.py, which comes with the September flame set. The file says which unit K is in. ANNAbel uses 16-bit DN and the tool 12-bit DN, so the ANNAbel K is divided by 16. The tool recomputes G<sub>j</sub>(T) from the ψ<sub>j</sub> of the file and warns if it does not match.')}</p>
  <p>${fileBtn('wr-calin', L('Elegir la calibración .json', 'Choose the calibration .json'), '.json,application/json')}</p>
  ${c ? `<p class="meta">K<sub>R</sub>, K<sub>G</sub>, K<sub>B</sub> = ${sig(c.K.R, 4)}, ${sig(c.K.G, 4)}, ${sig(c.K.B, 4)} · ${esc(c.camera)} · ${esc(c.psi.name)}</p>` : ''}
  <div class="wr-msgs">${msgHtml(S.calFileMsgs)}</div>`;
}

// ============================================================================ stage 2
function renderStage2() {
  const el = document.getElementById('wr-s2');
  if (!el) return;
  const d = calDecision();
  const yes = L('sí', 'yes'), no = L('no', 'no');
  const dT = d.state === 'inconsistent' || d.state === 'untested' ? NaN : dTfromOven();
  const worstT = S.cal?.mode === 'oven' ? S.cal.ratioWorst ?? NaN : NaN;
  const tl = S.cal?.mode === 'oven' && S.cal.tail?.applied ? S.cal.tail : null;
  const tb = S.o.tailBias ? (getLang() === 'es' ? S.o.tailBias.es : S.o.tailBias.en) : '';
  const tailPara = !tl ? '' : tl.pass ? L(` La cola infrarroja de ψ<sub>R</sub> se calibró con el horno (1a.7, s<sub>R</sub> = ${num(tl.s, 3)} ± ${num(tl.u, 3)}) y la prueba de la razón con una temperatura fuera pasa. Con el horno del grupo queda un sesgo residual esperado en la llama ${tb}, que no se resta a T. Sin la corrección T saldría ${ovenBias()} más baja.`, ` The infrared tail of ψ<sub>R</sub> was calibrated with the oven (1a.7, s<sub>R</sub> = ${num(tl.s, 3)} ± ${num(tl.u, 3)}) and the ratio test with one temperature left out passes. With the group’s oven an expected residual bias in the flame of ${tb} remains, and it is not subtracted from T. Without the correction T would come out ${ovenBias()} too low.`)
    : L(` La cola infrarroja de ψ<sub>R</sub> se calibró con el horno y K<sub>R</sub>/K<sub>G</sub> de la temperatura dejada fuera queda a ${num(tl.criterionPct, 1)} % del ajuste, sobre el criterio de 1 % (1a.7).`, ` The infrared tail of ψ<sub>R</sub> was calibrated with the oven and K<sub>R</sub>/K<sub>G</sub> of the temperature left out sits ${num(tl.criterionPct, 1)} % from the fit, above the 1 % criterion (1a.7).`);
  const para = d.state === 'absolute' ? L('La calibración es absoluta. Con ella se calculan la temperatura y la fracción de volumen de hollín en ppm.', 'The calibration is absolute. It gives the temperature and the soot volume fraction in ppm.')
    : d.state === 'relative' ? L('La calibración es relativa. La temperatura sale igual que con una absoluta y la fracción de volumen de hollín queda en unidades relativas, porque falta la emisividad de la cavidad o la transmisión de una ventana.', 'The calibration is relative. The temperature comes out as with an absolute one and the soot volume fraction stays in relative units, because the cavity emissivity or a window transmission is missing.')
    : d.state === 'inconsistent' ? L('La calibración no es consistente. La temperatura se calcula con una advertencia y la fracción de volumen de hollín no se calcula.', 'The calibration is inconsistent. The temperature is computed with a warning and the soot volume fraction is not computed.')
    : d.state === 'untested' ? L('La calibración no se pudo probar porque el ajuste tiene una sola temperatura del horno en rojo y verde. Sin una segunda temperatura no se sabe si un solo K por canal explica el horno, y la prueba de la razón no se hace. La temperatura se calcula con una advertencia y la fracción de volumen de hollín queda provisional.', 'The calibration could not be tested because the fit has a single oven temperature in red and green. Without a second temperature there is no way to know whether one K per channel explains the oven, and the ratio test is not run. The temperature is computed with a warning and the soot volume fraction stays provisional.')
    : d.state === 'one_channel' ? L('Solo un canal quedó calibrado. Sin dos canales no hay temperatura.', 'Only one channel is calibrated. Without two channels there is no temperature.')
    : L('No hay calibración cargada. La razón usa las curvas nominales de la cámara, la temperatura es solo orientativa y no se calcula la fracción de volumen de hollín.', 'No calibration is loaded. The ratio uses the nominal camera curves, the temperature is only indicative and the soot volume fraction is not computed.');
  const prov = d.provisional && d.fs !== 'no' ? L(' f<sub>s</sub> queda provisional hasta confirmar el tiempo de exposición y la escala, y la temperatura no cambia.', ' f<sub>s</sub> stays provisional until the exposure time and the scale are confirmed, and the temperature does not change.') : '';
  const msgs: Msg[] = d.codes.filter((c) => c !== 'texp_unconfirmed' && c !== 'scale_missing').map((c) => ({ code: c, vars: c === 'cal_one_channel' ? { channel: CH.indexOf(d.channels[0] ?? 'R') } : undefined }));
  // ratio_test_fail is shown once, under its own test in stage 1a.7, and the status line above keeps its two numbers
  if (S.flame.sensorMismatch && S.cal) msgs.push({ code: 'sensor_mismatch', vars: { a: `${S.cal.sensor.width} × ${S.cal.sensor.height}`, b: (() => { const f = flameItems()[0]; const s = sensorOf(f); return `${s.width} × ${s.height}`; })() } });
  el.innerHTML = `<h2 class="row__label nonum">${L('Etapa 2 · Estado de la calibración', 'Stage 2 · Calibration status')}</h2>
  ${statusLine('wr-st2', `${stateLabel(d)}${fin(dT) ? (tl ? L(`, cola infrarroja calibrada con el horno, ±${num(dT)} K en T a 1800 K por el parámetro y el termopar`, `, infrared tail calibrated with the oven, ±${num(dT)} K in T at 1800 K from the parameter and the thermocouple`) : L(`, ±${num(dT)} K en T a 1800 K por el termopar del horno`, `, ±${num(dT)} K in T at 1800 K from the oven thermocouple`)) : d.state === 'inconsistent' && tl ? L(`. Con la cola calibrada K<sub>R</sub>/K<sub>G</sub> queda a ${num(tl.criterionPct, 1)} % con una temperatura fuera, sobre el criterio de 1 %`, `. With the tail calibrated K<sub>R</sub>/K<sub>G</sub> sits ${num(tl.criterionPct, 1)} % off with one temperature left out, above the 1 % criterion`) : d.state === 'inconsistent' && fin(worstT) && worstT > 50 ? L(`. En el horno la razón da hasta ${sgn(worstT)} K sobre el termopar, y en una llama T saldría ${ovenBias()} más baja`, `. At the oven the ratio reads up to ${sgn(worstT)} K above the thermocouple, and in a flame T would come out ${ovenBias()} too low`) : ''}`)}
  <p>${para}${prov}${tailPara}</p>
  ${tableHtml([L('Qué se calcula', 'What is computed'), L('Con esta calibración', 'With this calibration')], [[L('Temperatura', 'Temperature'), d.T === 'yes' ? yes : d.T === 'warn' ? L('sí, con advertencia', 'yes, with a warning') : no], [L('Fracción de volumen de hollín f<sub>s</sub>', 'Soot volume fraction f<sub>s</sub>'), d.fs === 'ppm' ? (d.provisional ? L('sí, en ppm, provisional hasta confirmar t<sub>exp</sub> y la escala', 'yes, in ppm, provisional until t<sub>exp</sub> and the scale are confirmed') : L('sí, en ppm', 'yes, in ppm')) : d.fs === 'relative' ? L('relativa', 'relative') : no], [L('Canales calibrados', 'Calibrated channels'), d.state === 'uncalibrated' ? L('ninguno, se usan las curvas nominales', 'none, the nominal curves are used') : d.state === 'inconsistent' ? L('ninguno pasa la prueba. T usa igual el K<sub>R</sub>/K<sub>G</sub> del horno', 'none passes the test. T still uses the oven K<sub>R</sub>/K<sub>G</sub>') : d.channels.join(', ') || '–']])}
  <div class="wr-msgs">${msgHtml(msgs)}</div>
  <details class="wr-note"><summary>${L('Diferencia con el modo relativo de la aplicación ANNAbel del grupo', 'Difference from the relative mode of the group’s ANNAbel app')}</summary><p>${L('El modo relativo de la aplicación ANNAbel usa la respuesta η sin el factor λ, sin filtro infrarrojo y con una ganancia de 1,5 en el verde. Es justo el error frecuente que muestra la presentación. Aquí todos los modos usan ψ<sub>j</sub> = (λ/λ<sub>ref</sub>) η<sub>j</sub> τ. Tampoco se escala el oscuro de la esfera con el tiempo de la llama, porque el nivel negro no crece con t<sub>exp</sub>.', 'The relative mode of the ANNAbel app uses the response η without the λ factor, without the infrared filter and with a gain of 1.5 on green. It is exactly the common error the presentation shows. Here every mode uses ψ<sub>j</sub> = (λ/λ<sub>ref</sub>) η<sub>j</sub> τ. The sphere dark is not scaled with the flame exposure either, because the black level does not grow with t<sub>exp</sub>.')}</p></details>`;
}

// ============================================================================ stage 3
function stage3Status(): string {
  if (staleNote(3)) return staleNote(3);
  if (!flameItems().length) return L('Faltan las imágenes de la llama', 'The flame images are missing');
  if (!fin(S.flame.texpUs)) return L('Falta el tiempo de exposición de la llama', 'The flame exposure time is missing');
  if (S.flame.msgs.some((m) => m.level === 'error')) return L('Detenida, revisen el error de abajo', 'Stopped, see the error below');
  if (!S.flame.prepared && S.flame.prepErr) return L('No se pudieron preparar las proyecciones, revisen el error de abajo', 'The projections could not be prepared, see the error below');
  if (!S.flame.prepared) return L('Lista para preparar las proyecciones', 'Ready to prepare the projections');
  return L(`Lista, ${S.flame.prepared.nz} alturas y ${S.flame.prepared.nr} radios`, `Ready, ${S.flame.prepared.nz} heights and ${S.flame.prepared.nr} radii`);
}
/** With the flame t_exp typed above, the missing name is a note that says which value is used (GB3r3-estudiante-05). */
function stage3Msgs(): Msg[] {
  if (!fin(S.flame.texpUs)) return S.flame.msgs;
  const had = S.flame.msgs.some((m) => m.code === 'name_unread_flame');
  const rest = S.flame.msgs.filter((m) => m.code !== 'name_unread_flame');
  return had ? [...rest, { code: 'texp_typed', vars: { a: num(S.flame.texpUs, Number.isInteger(S.flame.texpUs) ? 0 : 1) }, level: 'info' }] : rest;
}
/** Shift test of R and B on the ruler, for run.json. */
function shiftRecord() {
  const t = S.flame.shift;
  if (!t) return null;
  return { verdict: t.verdict, why: t.why, pattern: t.pattern, k: t.k, kUnc: t.kUnc, pred: t.pred, meas: t.meas, unc: t.unc, axes: t.axes, nEdge: t.nEdge, tiles: t.tiles };
}
/** Readout of the shift test under the ruler (decision on closing G4). Shifts in sensor pixels, positive right and down. */
function shiftHtml(): string {
  const t = S.flame.shift;
  const head = `<h4>${L('Prueba del corrimiento de R y B', 'Shift test of R and B')}</h4>`;
  if (!t) return `${head}<p class="lab__readout" id="wr-shift">${L('No se pudo leer la regla para la prueba.', 'The ruler could not be read for the test.')}</p>`;
  if (t.verdict === 'not_applicable') return `${head}<p class="lab__readout" id="wr-shift" data-verdict="${t.verdict}">${L('La regla no trae R y B replicados en bloques de 2 × 2, así que no es RGB de pylon y la prueba no aplica. El centrado no hace nada en esta imagen.', 'The ruler does not carry R and B replicated in 2 × 2 blocks, so it is not pylon RGB and the test does not apply. The centring does nothing on this image.')}</p>`;
  const px = (v: number) => (Number.isFinite(v) ? num(2 * v, 2) : '—');
  const ax = (q: [number, number] | null) => (q ? L(`${px(q[1])} píxeles del sensor en horizontal y ${px(q[0])} en vertical`, `${px(q[1])} sensor pixels horizontally and ${px(q[0])} vertically`) : '—');
  const one = !t.axes[0] || !t.axes[1];
  const meas = t.why === 'few_edges' && !Number.isFinite(t.k) ? '' : L(`Medido sobre ${num(t.nEdge)} píxeles de borde en ${t.tiles} zonas, el rojo se corre respecto del verde ${ax(t.meas.R)}, y el azul ${ax(t.meas.B)} (positivo hacia la derecha y hacia abajo). El centrado supone ${ax(t.pred.R)} para el rojo y ${ax(t.pred.B)} para el azul, con el patrón ${t.pattern}. Se mide ${num(t.k, 2)} ± ${num(2 * t.kUnc, 2)} veces el corrimiento supuesto (1 si es el supuesto, 0 si no hay, negativo si va al otro lado).${one ? ` La regla tiene bordes en una sola dirección, ${t.axes[1] ? 'se mide solo la horizontal' : 'se mide solo la vertical'}.` : ''} `, `Measured on ${num(t.nEdge)} edge pixels in ${t.tiles} zones, red is shifted from green by ${ax(t.meas.R)}, and blue by ${ax(t.meas.B)} (positive to the right and down). The centring assumes ${ax(t.pred.R)} for red and ${ax(t.pred.B)} for blue, with the ${t.pattern} pattern. The measurement is ${num(t.k, 2)} ± ${num(2 * t.kUnc, 2)} times the assumed shift (1 if it is the assumed one, 0 if there is none, negative if it goes the other way).${one ? ` The ruler has edges in one direction only, ${t.axes[1] ? 'only the horizontal is measured' : 'only the vertical is measured'}.` : ''} `);
  const verdict = t.verdict === 'activate'
    ? L('El corrimiento va en el sentido que supone el centrado. Conviene activarlo con la casilla «Centrar R y B».', 'The shift goes the way the centring assumes. Tick «Centre R and B».')
    : t.verdict === 'keep_off'
      ? L('El corrimiento no va en el sentido que supone el centrado o es menos de la mitad. Dejen el centrado apagado, porque lo empeoraría.', 'The shift does not go the way the centring assumes, or it is less than half of it. Leave the centring off, because it would make it worse.')
      : L('La prueba no decide. Dejen el centrado apagado y fotografíen la regla enfocada, sin saturar y con bordes en las dos direcciones.', 'The test does not decide. Leave the centring off and photograph the ruler in focus, unsaturated and with edges in both directions.');
  const mismatch = (t.verdict === 'activate') !== S.centre && t.verdict !== 'inconclusive' ? ` <strong>${S.centre ? L('Ahora está activado.', 'It is on now.') : L('Ahora está apagado.', 'It is off now.')}</strong>` : '';
  return `${head}<p class="lab__readout" id="wr-shift" data-verdict="${t.verdict}">${meas}${verdict}${mismatch}</p>`;
}
function renderStage3() {
  const el = document.getElementById('wr-s3');
  if (!el) return;
  const fl = flameItems();
  const p = S.flame.prepared;
  const timed = fl.some((it) => it.name.timeS !== undefined);
  const scaleW = S.flame.rulerPts.length === 2 ? Math.hypot(S.flame.rulerPts[1][0] - S.flame.rulerPts[0][0], S.flame.rulerPts[1][1] - S.flame.rulerPts[0][1]) * S.flame.rulerScale : NaN;
  const frames = fl.map((it) => `<li class="wr-frame${it.include ? '' : ' wr-off'}"><label><input type="checkbox" data-incl="${it.id}"${it.include ? ' checked' : ''}> <canvas data-thumb="${it.id}" width="1" height="1"></canvas><span class="wr-path">${esc(it.path.split('/').pop()!)}</span><span class="meta">${L('asimetría', 'asymmetry')} ${num(it.asym ?? NaN, 2)}${it.fi && it.fi.nSatWork ? `, ${num(it.fi.nSatWork)} ${L('saturados', 'saturated')}` : ''}${it.why === 'tilted' ? L(', inclinado', ', tilted') : it.why === 'axis' ? L(', eje alejado de la mediana', ', axis far from the median') : it.why === 'average' ? L(', ya está en el promedio', ', already in the average') : ''}</span></label></li>`).join('');
  el.innerHTML = `<h2 class="row__label nonum">${L('Etapa 3 · Llama', 'Stage 3 · Flame')}</h2>
  ${statusLine('wr-st3', stage3Status())}
  <p>${L('Carguen los cuadros de la llama, el fondo o el oscuro con el mismo t<sub>exp</sub> y la imagen de la regla, en la tabla de 1a.1 o aquí.', 'Load the flame frames, the background or dark with the same t<sub>exp</sub> and the ruler image, in the 1a.1 table or here.')}</p>
  <p class="cluster">${fileBtn('wr-f-dir', L('Elegir una carpeta', 'Choose a folder'), '', true)} ${fileBtn('wr-f-files', L('Elegir archivos o un .zip', 'Choose files or a .zip'), '.tif,.tiff,.zip,.json')}</p>
  ${fileTable((it) => it.role === 'flame' || it.role === 'background' || it.role === 'ruler', 'wr-ft3')}
  <div class="wr-grid3">
    ${numIn('wr-texp', S.flame.texpUs, 'min="1" step="any"', L('t<sub>exp</sub> de la llama (µs)', 'Flame t<sub>exp</sub> (µs)'), L('Vale para todos los cuadros de la llama y manda sobre la tabla.', 'It holds for every flame frame and takes precedence over the table.'))}
    ${chk('wr-texpok', S.flame.texpConfirmed, L('Confirmo este valor con el registro de adquisición', 'I confirm this value with the acquisition log'))}
    ${sel('wr-bin', [['2x2', L('superpíxel 2 × 2 (por defecto)', '2 × 2 superpixel (default)')], ['1x1', L('sin agrupar', 'no grouping')]], S.group ? '2x2' : '1x1', L('Agrupamiento', 'Grouping'), L('El RGB de pylon replica R y B en bloques de 2 × 2, así que cada uno viene de un solo sitio del mosaico. La herramienta agrupa en la fase del rojo.', 'pylon RGB replicates R and B in 2 × 2 blocks, so each comes from a single mosaic site. The tool groups on the red phase.'))}
    ${chk('wr-centre', S.centre, L('Centrar R y B del RGB de pylon (apagado por defecto)', 'Centre R and B of the pylon RGB (off by default)'), L('Cada valor de R y B del RGB agrupado viene de un sitio a un cuarto de superpíxel del centro del verde. Con la casilla marcada la herramienta los corre a ese centro con el patrón Bayer RG, en el horno y en la llama. El sentido del corrimiento no está confirmado para esta cámara, por eso va apagado. La prueba con la imagen de la regla dice si conviene marcarla.', 'Each R and B value of the grouped RGB comes from a site a quarter of a superpixel off the centre of the green. With the box ticked the tool moves them to that centre with the Bayer RG pattern, in the oven and in the flame. The direction of the shift is not confirmed for this camera, so it is off. The test with the ruler image says whether to tick it.'))}
  </div>
  ${fl.length ? `<h3>${L('Cuadros', 'Frames')}</h3>
  ${timed ? `<p>${sel('wr-times', [['repeat', L('repeticiones de la misma llama, se promedian', 'repeats of the same flame, averaged')], ['instants', L('instantes distintos, se usa uno', 'different instants, one is used')]], S.flame.times, L('Los nombres traen tiempos de adquisición', 'The names carry acquisition times'))}${S.flame.times === 'instants' ? sel('wr-instant', fl.map((it) => [it.id, it.path.split('/').pop()!] as [string, string]), S.flame.instant, L('Instante', 'Instant')) : ''}</p>` : ''}
  <ul class="wr-frames">${frames}</ul>` : ''}
  <div class="wr-msgs">${msgHtml(stage3Msgs())}</div>
  <h3>${L('Geometría y escala', 'Geometry and scale')}</h3>
  <div class="wr-grid3">
    ${sel('wr-sensq', [['0', L('sin girar', 'no rotation')], ['1', L('90° a la derecha', '90° clockwise')], ['3', L('90° a la izquierda', '90° counter-clockwise')], ['2', '180°']], String(S.flame.sensorQ), L('Posición del sensor', 'Sensor orientation'), L('Solo si otro programa giró los archivos. No cambia la imagen que se ve. Solo alinea K<sub>j</sub>(x, y) de la calibración con la llama.', 'Only if another program rotated the files. It does not change the image you see. It only aligns K<sub>j</sub>(x, y) of the calibration with the flame.'))}
    ${sel('wr-quarter', [['0', L('como se cargó', 'as loaded')], ['1', L('90° a la derecha', '90° clockwise')], ['3', L('90° a la izquierda', '90° counter-clockwise')], ['2', '180°']], String(S.flame.quarter), L('Girar para dejar la llama vertical', 'Rotate so the flame stands upright'), L('El quemador abajo y la punta arriba en la imagen orientada.', 'Burner at the bottom and tip at the top in the oriented image.'))}
    ${chk('wr-burnerok', S.flame.burnerOk, L('Confirmo que en la vista previa el quemador queda abajo', 'I confirm that the burner is at the bottom of the preview'), L('La herramienta supone que la última fila es la base de la llama. La base es la parte ancha y tenue, y la punta, angosta, va arriba. Una llama cabeza abajo da mapas sin sentido. Si arriba queda más ancha que abajo, la herramienta avisa, y la casilla decide.', 'The tool assumes that the last row is the base of the flame. The base is the wide, dim part and the narrow tip goes at the top. An upside-down flame gives meaningless maps. If it is wider at the top than at the bottom the tool warns, and the box decides.'))}
    ${chk('wr-mirror', S.flame.mirror, L('Espejo izquierda derecha', 'Left-right mirror'))}
    ${numIn('wr-fine', S.flame.fineDeg, 'min="-10" max="10" step="0.1"', L('Rotación fina (°)', 'Fine rotation (°)'))}
    ${chk('wr-axisauto', S.flame.axisAuto, L('Eje automático (como la app ANNAbel del grupo, eje entre dos columnas)', 'Automatic axis (as in the group’s ANNAbel app, axis between two columns)'))}
    ${!S.flame.axisAuto ? numIn('wr-axis', S.flame.axisManual, 'step="0.05"', L('Eje (columna, píxeles de trabajo)', 'Axis (column, working pixels)')) : ''}
    ${numIn('wr-hw', S.flame.halfWidth, 'min="4" step="1"', L('Semiancho de búsqueda del eje (píxeles)', 'Axis search half-width (pixels)'), L('Si queda vacío, 30 % del ancho.', 'If left empty, 30 % of the width.'))}
    ${numIn('wr-nr', S.flame.nr, 'min="4" step="1"', L('Radios por lado (semiancho del recorte)', 'Radii per side (crop half-width)'), L('Si queda vacío, hasta el borde más cercano.', 'If left empty, up to the nearest edge.'))}
    ${numIn('wr-rtop', S.flame.rowTop, 'min="0" step="1"', L('Primera fila', 'First row'))}
    ${numIn('wr-rbot', S.flame.rowBottom, 'min="0" step="1"', L('Última fila', 'Last row'))}
    ${numIn('wr-rph', S.flame.rowsPerHeight, 'min="1" max="32" step="1"', L('Filas por altura', 'Rows per height'), L('Promedio de 4 filas por defecto.', 'Mean of 4 rows by default.'))}
    ${numIn('wr-burner', S.flame.burnerRow, 'min="0" step="1"', L('Fila del quemador', 'Burner row'), L('Si queda vacío, la última fila.', 'If left empty, the last row.'))}
    ${numIn('wr-zb', S.flame.zBurner, 'step="any"', L('Altura de esa fila (mm)', 'Height of that row (mm)'))}
    ${numIn('wr-mmpx', S.flame.mmPerPx, 'min="0" step="any"', L('Escala, mm por píxel del sensor', 'Scale, mm per sensor pixel'), fin(scaleW) ? L(`Regla, ${num(scaleW, 1)} píxeles del sensor entre las marcas.`, `Ruler, ${num(scaleW, 1)} sensor pixels between the marks.`) : '')}
    ${numIn('wr-sunc', S.flame.scaleUnc, 'min="0" max="20" step="0.1"', L('Incertidumbre de la escala (%)', 'Scale uncertainty (%)'), L('Entra en la banda de f<sub>s</sub> de la etapa 6, no en T.', 'It enters the f<sub>s</sub> band of stage 6, not T.'))}
    ${chk('wr-scaleok', S.flame.scaleConfirmed, L('Confirmo la escala', 'I confirm the scale'))}
  </div>
  ${S.flame.rulerPreview ? `<details class="wr-note" open><summary>${L('Regla', 'Ruler')}</summary><p>${L('Hagan clic en dos marcas separadas al menos 10 mm y escriban la distancia entre ellas. La escala se calcula en píxeles del sensor, antes de agrupar.', 'Click two marks at least 10 mm apart and type the distance between them. The scale is computed in sensor pixels, before grouping.')}</p>
    <div class="wr-row">${numIn('wr-rmm', S.flame.rulerMm, 'min="0" step="any"', L('Distancia entre las marcas (mm)', 'Distance between the marks (mm)'))} ${btn('wr-rclear', L('Borrar las marcas', 'Clear the marks'))}</div>
    <canvas id="wr-rcv" class="wr-preview wr-ruler" tabindex="0" aria-label="${L('Imagen de la regla. Hagan clic en dos marcas, o escriban los milímetros por píxel en el campo Escala.', 'Ruler image. Click two marks, or type the millimetres per pixel in the Scale field.')}"></canvas>
    ${shiftHtml()}</details>` : (S.group ? `<p class="meta" id="wr-shift">${L('Sin imagen de la regla no hay prueba del corrimiento de R y B, y el centrado queda apagado.', 'Without a ruler image there is no shift test of R and B, and the centring stays off.')}</p>` : '')}
  <p>${btn('wr-prep', L('Preparar las proyecciones', 'Prepare the projections'), !fl.length)} ${p ? btn('wr-npz', L('Descargar proyecciones .npz para la app Python', 'Download projections .npz for the Python app')) : ''}</p>
  ${p ? `<div class="wr-grid2"><figure><canvas id="wr-fcv" class="wr-preview"></canvas><figcaption>${L('Imagen orientada, suma de los tres canales con gamma 2,2 solo para mirar. Línea continua, eje. Punteadas, recorte y alturas usadas.', 'Oriented image, sum of the three channels with gamma 2.2 for viewing only. Solid line, axis. Dotted, crop and heights used.')}</figcaption></figure>
  <div><p class="lab__readout">${L(`Eje en ${num(p.axis, 2)} píxeles de trabajo, ${p.nr} radios por lado de ${num(p.drMm, 4)} mm, ${p.nz} alturas de ${num(p.zMm[0], 1)} a ${num(p.zMm[p.nz - 1], 1)} mm. En el borde del recorte la señal es ${pct(p.edgeRatio, 2)} del pico. Diferencia entre mitades ${pct(median(p.halvesRms), 1)} del pico de cada altura, en la mediana. ${median(p.halvesRms) > 0.1 ? 'Sobre 10 % los mapas cerca del eje son poco confiables. Prueben «solo la derecha» y «solo la izquierda» en la etapa 4 y comparen.' : 'Bajo 10 % las dos mitades se pueden promediar.'}${p.nHot ? ` ${num(p.nHot)} píxeles calientes reemplazados.` : ''}`, `Axis at ${num(p.axis, 2)} working pixels, ${p.nr} radii per side of ${num(p.drMm, 4)} mm, ${p.nz} heights from ${num(p.zMm[0], 1)} to ${num(p.zMm[p.nz - 1], 1)} mm. At the crop edge the signal is ${pct(p.edgeRatio, 2)} of the peak. Difference between halves ${pct(median(p.halvesRms), 1)} of the peak of each height, median. ${median(p.halvesRms) > 0.1 ? 'Above 10 % the maps near the axis are not reliable. Try “right only” and “left only” in stage 4 and compare.' : 'Below 10 % the two halves can be averaged.'}${p.nHot ? ` ${num(p.nHot)} hot pixels replaced.` : ''}`)}</p>
  ${heightSlider('wr-h3')}
  <figure><div id="wr-halves" class="wr-plot"></div><figcaption>${L(`Proyecciones P<sub>j</sub> a la altura elegida, mitad derecha continua y mitad izquierda reflejada punteada, en ${p.unitAbs ? 'W m⁻² sr⁻¹' : 'unidades relativas'}.`, `Projections P<sub>j</sub> at the chosen height, right half solid and mirrored left half dotted, in ${p.unitAbs ? 'W m⁻² sr⁻¹' : 'relative units'}.`)}</figcaption></figure></div></div>` : ''}`;
  for (const it of fl) {
    const c = el.querySelector(`canvas[data-thumb="${it.id}"]`) as HTMLCanvasElement | null;
    if (c && it.thumb) drawThumb(c, it.thumb, S.flame.quarter, S.flame.mirror);
  }
  drawRuler();
  if (p) drawFlame();
}
/** Frame thumbnail with the rotation chosen under Geometry and scale, like the preview (GB3r3-estudiante-18). */
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
  return `<label class="control"><span class="control__head">${L('Altura', 'Height')} <output>${num(p.zMm[S.flame.height], 2)} mm</output></span><input type="range" id="${id}" class="wr-height" min="0" max="${p.nz - 1}" step="1" value="${S.flame.height}"></label>`;
}
function drawRuler() {
  const cv = document.getElementById('wr-rcv') as HTMLCanvasElement | null;
  const pv = S.flame.rulerPreview;
  if (!cv || !pv) return;
  const g = drawPreview(cv, pv, 3, false, 1.2);
  for (const [x, y] of S.flame.rulerPts) circle(g, x / pv.step, y / pv.step, 4, '#00d0ff', false, 2);
  if (S.flame.rulerPts.length === 2) line(g, S.flame.rulerPts[0][0] / pv.step, S.flame.rulerPts[0][1] / pv.step, S.flame.rulerPts[1][0] / pv.step, S.flame.rulerPts[1][1] / pv.step, '#00d0ff');
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
  plot('wr-halves', { xlabel: 'y (mm)', ylabel: 'Pⱼ', height: 230 }, ser);
}

// ============================================================================ stage 4
function stage4Status(): string {
  if (staleNote(4)) return staleNote(4);
  if (!S.flame.prepared) return L('Faltan las proyecciones de la etapa 3', 'The projections of stage 3 are missing');
  if (S.abel.annabel.running) return L('ANNAbel entrenando', 'ANNAbel training');
  if (!S.abel.out) return L('Lista para invertir', 'Ready to invert');
  return L(`Lista, ${methodName(S.abel.outMethod)}`, `Ready, ${methodName(S.abel.outMethod)}`);
}
const methodName = (m: string) => ({ onion: L('pelado de cebolla', 'onion peeling'), tikhonov: 'Tikhonov', sat: 'SAT', annabel: 'ANNAbel' } as Record<string, string>)[m] ?? m;
function renderStage4() {
  const el = document.getElementById('wr-s4');
  if (!el) return;
  const p = S.flame.prepared, o = S.abel.out, A = S.abel.annabel;
  const lr = Math.log10(S.abel.lambdaRel);
  const msgs: Msg[] = [...A.msgs];
  if (o && o.overReg) {
    const atCorner = S.abel.outMethod === 'tikhonov' && !!S.abel.lcurve && Math.abs(Math.log10(o.lambdaRel / S.abel.lcurve.lambda)) < 0.03;
    const rows = o.overRows ?? [];
    msgs.push(atCorner && p && rows.length ? { code: 'over_regularised_corner', vars: { n: o.overReg, a: num(Math.min(...rows.map((k) => p.zMm[k])), 1), b: num(Math.max(...rows.map((k) => p.zMm[k])), 1) }, level: 'info' } : { code: 'over_regularised', vars: { n: o.overReg } });
  }
  const lc = S.abel.lcurve;
  if (lc && S.abel.method === 'tikhonov' && Math.abs(Math.log10(S.abel.lambdaRel / lc.lambda)) < 0.03 && (lc.index <= 1 || lc.index >= lc.lambdas.length - 2)) msgs.push({ code: 'lcurve_edge', vars: { a: sig(lc.lambda, 2) } });
  // stage 3 already says it with the same count (GB3r2-adv-13)
  if (o && o.excluded.length && o.excluded.length !== p?.satRows.length) msgs.push({ code: 'rows_saturated', vars: { n: o.excluded.length }, level: 'info' });
  el.innerHTML = `<h2 class="row__label nonum">${L('Etapa 4 · Inversión de Abel por canal', 'Stage 4 · Abel inversion per channel')}</h2>
  ${statusLine('wr-st4', stage4Status())}
  <p>${L('Los tres canales usan el mismo método, el mismo λ relativo y la misma malla, porque resoluciones distintas por canal inventan estructura en la razón. La curva L se calcula con el verde y se aplica a todos. Rayos en (i + ½)·dr con el eje entre dos columnas, la convención de la app ANNAbel del grupo.', 'The three channels use the same method, the same relative λ and the same grid, because different resolutions per channel invent structure in the ratio. The L-curve is computed on green and applied to all. Rays at (i + ½)·dr with the axis between two columns, the convention of the group’s ANNAbel app.')}</p>
  <div class="lab"><div class="lab__grid"><div class="lab__controls">
    <p class="seg" role="group" aria-label="${L('Método', 'Method')}">${(['onion', 'tikhonov', 'sat', 'annabel'] as const).map((m) => `<button type="button" data-method="${m}" aria-pressed="${S.abel.method === m}">${methodName(m)}</button>`).join('')}</p>
    ${sel('wr-sym', S.abel.method === 'sat' || S.abel.method === 'annabel' ? [['both', L('las dos mitades en el ajuste', 'both halves in the fit')], ['average', L('promedio de las mitades', 'mean of the halves')], ['right', L('solo la derecha', 'right only')], ['left', L('solo la izquierda', 'left only')]] : [['average', L('promedio de las mitades', 'mean of the halves')], ['right', L('solo la derecha', 'right only')], ['left', L('solo la izquierda', 'left only')]], S.abel.method === 'annabel' ? 'both' : S.abel.sym === 'both' && S.abel.method !== 'sat' ? 'average' : S.abel.sym, L('Simetrización', 'Symmetrisation'))}
    ${S.abel.method === 'tikhonov' || S.abel.method === 'sat' ? '' : ''}
    ${S.abel.method === 'tikhonov' ? `<label class="control"><span class="control__head">λ<sub>rel</sub> <output>${sig(S.abel.lambdaRel, 2)}</output></span><input type="range" id="wr-lam" min="${Math.log10(LAMBDA_REL_MIN)}" max="${Math.log10(LAMBDA_REL_MAX)}" step="0.05" value="${lr}"><span class="meta">${L('Escala logarítmica de 10⁻⁶ a 10³. Tikhonov penaliza la curvatura del perfil (segunda diferencia).', 'Log scale from 10⁻⁶ to 10³. Tikhonov penalises the curvature of the profile (second difference).')}</span></label>
      <p>${btn('wr-lc', L('Calcular la curva L', 'Compute the L-curve'), !p)} ${S.abel.lcurve ? btn('wr-lc-use', L(`Volver a la esquina, ${sig(S.abel.lcurve.lambda, 2)}`, `Back to the corner, ${sig(S.abel.lcurve.lambda, 2)}`)) : ''}</p>${S.abel.lcurve ? `<p class="meta">${L('Al calcular la curva L, λ<sub>rel</sub> pasa a su esquina. Pueden cambiarlo a mano.', 'Computing the L-curve sets λ<sub>rel</sub> to its corner. You can change it by hand.')}</p>` : ''}` : ''}
    ${S.abel.method === 'sat' ? `${numIn('wr-seg', S.abel.segments, 'min="3" max="60" step="1"', L('Segmentos de la spline', 'Spline segments'))}${numIn('wr-satl', S.abel.satLambda, 'min="0" step="any"', 'λ SAT', L('κ = 0, sin LOSA.', 'κ = 0, no LOSA.'))}` : ''}
    ${S.abel.method === 'annabel' ? annabelPanel() : ''}
    <p>${btn('wr-abel', S.abel.method === 'annabel' ? L('Entrenar ANNAbel', 'Train ANNAbel') : L('Calcular', 'Compute'), !p || A.running)} ${A.running ? btn('wr-ann-stop', L('Detener', 'Stop')) : ''}</p>
    ${p ? heightSlider('wr-h4') : ''}
    <p class="lab__readout" id="wr-abel-read" aria-live="polite"></p>
  </div><div class="lab__plots">
    ${S.abel.method === 'tikhonov' && S.abel.lcurve ? `<figure><div id="wr-lcp" class="wr-plot"></div><figcaption>${L('Curva L en el verde, 25 valores de λ<sub>rel</sub> sobre el tercio central de las alturas. Punto naranjo, la esquina (máxima curvatura).', 'L-curve on green, 25 values of λ<sub>rel</sub> over the central third of the heights. Orange point, the corner (maximum curvature).')}</figcaption></figure>` : ''}
    ${o ? `<figure><div id="wr-reproj" class="wr-plot"></div><figcaption>${L('Proyección medida, las dos mitades en línea delgada, y reproyección L f en línea gruesa, por canal, a la altura elegida.', 'Measured projection, both halves as thin lines, and reprojection L f as a thick line, per channel, at the chosen height.')}</figcaption></figure>
    <figure><div id="wr-resid" class="wr-plot"></div><figcaption>${L('Residuo P − L f del verde y la diferencia entre mitades, que da la escala del ruido.', 'Residual P − L f on green and the difference between halves, which sets the noise scale.')}</figcaption></figure>
    <figure><div id="wr-jr" class="wr-plot"></div><figcaption>${L('Emisión j<sub>j</sub>(r) de los tres canales en W m⁻³ sr⁻¹, a la altura elegida.', 'Emission j<sub>j</sub>(r) of the three channels in W m⁻³ sr⁻¹, at the chosen height.')}</figcaption></figure>
    <figure class="wr-maps3">${[0, 1, 2].map((c) => `<div><canvas id="wr-jmap-${c}" class="wr-map"></canvas><span class="meta">j<sub>${CH_LETTER[c]}</sub>(r, z)</span></div>`).join('')}<figcaption>${L('Mapas j<sub>j</sub>(r, z) por canal, del fondo al color del canal, de 0 al máximo del canal, proporción real.', 'Maps j<sub>j</sub>(r, z) per channel, from the background to the channel colour, from 0 to the channel maximum, true aspect ratio.')}</figcaption></figure>` : ''}
  </div></div></div>
  <div class="wr-msgs">${msgHtml(msgs)}</div>
  <details class="wr-note" id="wr-compare"${S.abel.compare ? ' open' : ''}><summary>${L('Comparar métodos', 'Compare methods')}</summary>
    <p>${L('Pelado de cebolla, Tikhonov con el λ elegido y SAT a la misma altura, en el verde.', 'Onion peeling, Tikhonov with the chosen λ and SAT at the same height, on green.')} ${btn('wr-cmp', L('Comparar a esta altura', 'Compare at this height'), !p)}</p>
    ${S.abel.compare ? `<figure><div id="wr-cmpp" class="wr-plot"></div><figcaption>${L('j<sub>G</sub>(r) por método. Gris, pelado. Continua, Tikhonov. Punteada, SAT.', 'j<sub>G</sub>(r) per method. Grey, onion peeling. Solid, Tikhonov. Dotted, SAT.')}</figcaption></figure>` : ''}
  </details>`;
  drawAbel();
}
/** A duration in words, "menos de 1 min" or "cerca de N min" (GB3r2-adv-18). */
function minText(sec: number): string {
  const m = sec / 60;
  if (!(m >= 1)) return L('menos de 1 min', 'under 1 min');
  // rounded up, a promise that does not fall short (GB3r3-adv-13)
  return L(`menos de ${num(Math.ceil(m))} min`, `under ${num(Math.ceil(m))} min`);
}
function annabelPanel(): string {
  const A = S.abel.annabel;
  const est = A.estimate ? L(`Camino ${A.estimate.path === 'webgpu' ? L('tarjeta gráfica', 'graphics card') : `${helpers.length} ${L('hilos del procesador', 'processor threads')}`}, ${num(A.estimate.sPerIter * 1000, 1)} ms por iteración, ${minText(A.estimate.sPerIter * A.iterations)} para ${num(A.iterations)} iteraciones.`, `Path ${A.estimate.path === 'webgpu' ? 'graphics card' : `${helpers.length} processor threads`}, ${num(A.estimate.sPerIter * 1000, 1)} ms per iteration, ${minText(A.estimate.sPerIter * A.iterations)} for ${num(A.iterations)} iterations.`) : '';
  return `<p class="meta">${L('ANNAbel (Escudero et al., Proc. Combust. Inst. 40 (2024) 105493). La herramienta usa la red de la app del grupo, que ajusta cuatro capas de 64 neuronas a las dos mitades de todas las alturas a la vez. Con 30 000 iteraciones el error esperado es de 3 a 5 %. ', 'ANNAbel (Escudero et al., Proc. Combust. Inst. 40 (2024) 105493). The tool uses the network of the group’s app, which fits four layers of 64 neurons to both halves of every height at once. With 30 000 iterations the expected error is 3 to 5 %. ')}${S.gpu ? L('Este navegador puede usar la tarjeta gráfica.', 'This browser can use the graphics card.') : L('Sin tarjeta gráfica disponible corre en el procesador, más lento, como opción avanzada. Si tarda mucho, usen la app Python con el archivo .npz de la etapa 3.', 'Without the graphics card it runs on the processor, more slowly, as an advanced option. If it takes too long, use the Python app with the .npz file of stage 3.')}</p>
  ${numIn('wr-ann-it', A.iterations, 'min="100" max="100000" step="100"', L('Iteraciones', 'Iterations'))}
  ${numIn('wr-seed', S.res.seed, 'min="0" step="1"', L('Semilla', 'Seed'))}
  <p>${btn('wr-ann-est', L('Estimar el tiempo', 'Estimate the time'), !S.flame.prepared || A.running)}</p>
  <p class="meta" id="wr-ann-est-t">${est}</p>
  <p class="meta" id="wr-ann-prog" aria-live="polite"></p>`;
}
function renderAnnabelProgress() {
  const e = document.getElementById('wr-ann-prog');
  const t = S.abel.annabel.tick;
  if (e && t) e.textContent = L(`Iteración ${num(t.iter)} de ${num(t.total)}, pérdida ${sig(t.loss, 3)}, mejor ${sig(t.best, 3)} en la ${num(t.bestIter)}${t.etaS !== null ? `, faltan unos ${num(t.etaS / 60, 1)} min` : ''}.`, `Iteration ${num(t.iter)} of ${num(t.total)}, loss ${sig(t.loss, 3)}, best ${sig(t.best, 3)} at ${num(t.bestIter)}${t.etaS !== null ? `, about ${num(t.etaS / 60, 1)} min left` : ''}.`);
}
function drawAbel() {
  const p = S.flame.prepared, o = S.abel.out;
  const lc = S.abel.lcurve;
  if (lc && document.getElementById('wr-lcp')) {
    plot('wr-lcp', { xlabel: 'ln ‖L f − P‖', ylabel: 'ln ‖Γ f‖', height: 220 }, [
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
  plot('wr-reproj', { xlabel: 'y (mm)', ylabel: 'Pⱼ', height: 220 }, ser);
  const diff = Array.from(p.right[1][k], (v, i) => v - p.left[1][k][i]);
  plot('wr-resid', { xlabel: 'y (mm)', ylabel: 'P − L f', height: 200 }, [
    { kind: 'line', x: p.rMm, y: o.resid[1][k], cls: 's-chG wr-thick' }, { kind: 'line', x: p.rMm, y: diff, cls: 's-muted', dash: true },
    { kind: 'line', x: [0, p.rMm[p.nr - 1]], y: [0, 0], cls: 's-muted' }]);
  plot('wr-jr', { xlabel: 'r (mm)', ylabel: 'jⱼ (W m⁻³ sr⁻¹)', height: 220 }, [0, 1, 2].map((c) => ({ kind: 'line', x: p.rMm, y: o.J[c][k], cls: chCls[c] })));
  for (let c = 0; c < 3; c++) {
    const cv = document.getElementById(`wr-jmap-${c}`) as HTMLCanvasElement | null;
    if (!cv) continue;
    const f = new Float64Array(p.nz * p.nr);
    let mx = 0;
    for (let kk = 0; kk < p.nz; kk++) for (let i = 0; i < p.nr; i++) { const v = o.J[c][kk][i]; f[kk * p.nr + i] = v; if (fin(v) && v > mx) mx = v; }
    drawMap(cv, f, p.nz, p.nr, (['ch_R', 'ch_G', 'ch_B'] as const)[c], 0, mx || 1);
  }
  // reading in words, residual near the axis against the edge
  const r = o.resid[1][k];
  const n3 = Math.max(1, Math.floor(p.nr / 5));
  const rms = (a: ArrayLike<number>, i0: number, i1: number) => { let s = 0, n = 0; for (let i = i0; i < i1; i++) if (fin(a[i])) { s += a[i] ** 2; n++; } return n ? Math.sqrt(s / n) : NaN; };
  const ax = rms(r, 0, n3), ed = rms(r, p.nr - n3, p.nr);
  const rd = document.getElementById('wr-abel-read');
  if (rd) rd.textContent = fin(ax) && fin(ed) && ed > 0 ? L(`A ${num(p.zMm[k], 1)} mm el residuo cerca del eje es ${num(ax / ed, 1)} veces el del borde. ${ax / ed > 2 ? 'El ruido se amplifica hacia el eje, como se espera en la inversión de Abel.' : 'El residuo es parejo en el radio.'}`, `At ${num(p.zMm[k], 1)} mm the residual near the axis is ${num(ax / ed, 1)} times the one at the edge. ${ax / ed > 2 ? 'Noise grows towards the axis, as expected in an Abel inversion.' : 'The residual is even along the radius.'}`) : '';
  const cmp = S.abel.compare;
  if (cmp && document.getElementById('wr-cmpp')) plot('wr-cmpp', { xlabel: 'r (mm)', ylabel: 'j<sub>G</sub>', height: 220 }, [
    { kind: 'line', x: p.rMm, y: cmp.onion, cls: 's-muted' }, { kind: 'line', x: p.rMm, y: cmp.tikhonov, cls: 's-chG' }, { kind: 'line', x: p.rMm, y: cmp.sat, cls: 's-chG', dash: true }]);
}

// ============================================================================ stage 5
function renderStage5() {
  const el = document.getElementById('wr-s5');
  if (!el) return;
  const t = S.tfs.out, p = S.flame.prepared;
  const d = calDecision();
  const msgs: Msg[] = [];
  if (t) {
    if (t.ambiguous) msgs.push({ code: 'ratio_ambiguous', vars: { n: t.ambiguous } });
    if (t.outside) msgs.push({ code: 'ratio_outside', vars: { n: t.outside }, level: 'info' });
    let neg = 0, edge = 0;
    for (const f of t.flags) { if (f & 1) neg++; if (f & 2) edge++; }
    if (neg) msgs.push({ code: 'fs_negative', vars: { n: neg }, level: 'info' });
    if (edge) msgs.push({ code: 'T_at_grid_edge', vars: { n: edge }, level: 'info' });
    if (t.chem.some((x) => x)) msgs.push({ code: 'chemilum', level: 'info' });
    if (d.state === 'uncalibrated') msgs.push({ code: 'chem_off_uncal', level: 'info' });
    // a plausibility check on the zone used, values a sooting flame does not reach (GB3r2-estudiante-11)
    const Ts = usedSorted(t.T), fsv = fsShown();
    const T95 = pctOf(Ts, 0.95), F95 = fsv && fsv.unit === 'ppm' ? pctOf(usedSorted(fsv.v), 0.95) : NaN;
    // single pixels far above the field, noise of the ratio near the axis with few frames (GB3r3-adv-05)
    if (fsv && p) {
      const Fu = usedSorted(fsv.v).filter((x) => x > 0), f95 = pctOf(Fu, 0.95);
      let fmax = -Infinity, at: [number, number] = [NaN, NaN], over = 0;
      for (let k = 0; k < p.nz; k++) for (let i = 0; i < p.nr; i++) { const q = k * p.nr + i; if (t.mask[q] !== 1 || !fin(fsv.v[q])) continue; if (fsv.v[q] > 3 * f95) over++; if (fsv.v[q] > fmax) { fmax = fsv.v[q]; at = [p.zMm[k], p.rMm[i]]; } }
      if (fin(f95) && f95 > 0 && (fmax > 5 * f95 || over > 0.01 * Fu.length)) msgs.push({ code: 'fs_outliers', vars: { a: num(fmax / f95, 0), b: num(at[0], 1), c: num(at[1], 2), n: over } });
    }
    if ((fin(T95) && T95 > 2400) || (fin(F95) && F95 > 50)) msgs.push(fin(F95) ? { code: 'implausible', vars: { a: num(T95), b: sig(F95, 2) } } : { code: 'implausible_T', vars: { a: num(T95) } });
  }
  if (S.tfs.path === 'net') {
    msgs.push(...S.tfs.net.msgs);
    const r = S.tfs.net.result;
    if (t && r?.xcheck && !r.xcheck.agrees) msgs.push({ code: 'net_xcheck_bad', vars: { a: xcheckBad(r.xcheck) } });
    if (t && r?.bounds?.warnT) msgs.push({ code: 'net_bounds_T', level: 'info' });
    if (t && r?.bounds?.warnFs && calDecision().fs === 'ppm') msgs.push({ code: 'net_bounds_fs', level: 'info' });
  }
  // uncalibrated is said once, in stage 2 (GB3r2-adv-13)
  el.innerHTML = `<h2 class="row__label nonum">${L('Etapa 5 · Temperatura y f<sub>s</sub>', 'Stage 5 · Temperature and f<sub>s</sub>')}</h2>
  ${statusLine('wr-st5', staleNote(5) || (!S.abel.out ? L('Falta la emisión de la etapa 4', 'The emission of stage 4 is missing') : S.tfs.net.running ? L('Red de RGB-ANNAbel entrenando', 'RGB-ANNAbel network training') : t ? L('Lista', 'Ready') : L('Lista para calcular', 'Ready to compute')))}
  <div class="lab"><div class="lab__grid"><div class="lab__controls">
    ${sel('wr-path', [['ratio', L('razón de dos canales', 'ratio of two channels')], ['fit3', L('avanzado, ajuste de T y f<sub>s</sub> con tres canales', 'advanced, fit of T and f<sub>s</sub> with three channels')], ['net', L('T y f<sub>s</sub> con red (RGB-ANNAbel)', 'T and f<sub>s</sub> with a network (RGB-ANNAbel)')]], S.tfs.path, L('Camino', 'Path'), S.tfs.path === 'fit3' ? L('Opción avanzada. El azul entra con su K<sub>B</sub>, que solo vale en la llama con pasa-cortos de 700 nm o con la esfera. Sin eso el camino recomendado es la razón R/G con f<sub>s</sub> de R y G.', 'Advanced option. Blue enters with its K<sub>B</sub>, which only holds in the flame with a 700 nm short-pass filter or with the sphere. Without that the recommended path is the R/G ratio with f<sub>s</sub> from R and G.') : '')}
    ${S.tfs.path === 'net' ? netPanel() : ''}
    ${S.tfs.path === 'ratio' ? sel('wr-pair', [['RG', 'R/G'], ['BG', 'B/G'], ['BR', 'B/R']], S.tfs.pair, L('Par de canales', 'Channel pair'), (preset().id === 'a2A1920-160ucPRO' ? L('R/G es el par principal. Con el sensor de la cámara del laboratorio (Sony IMX392) una misma razón B/G puede corresponder a dos temperaturas, y R/G es más sensible.', 'R/G is the main pair. With the sensor of the lab camera (Sony IMX392) one B/G ratio can belong to two temperatures, and R/G is more sensitive.') : L(`R/G es el par principal, el mismo que usa la presentación. La curva elegida es la de la ${preset().id}.`, `R/G is the main pair, the one the presentation uses. The chosen curve is that of the ${preset().id}.`))) : ''}
    ${sel('wr-em', [['group', L('ley del grupo (Yon et al. 2011)', 'group law (Yon et al. 2011)')], ['coeffs', 'a + b ln²λ + c lnλ/λ'], ['constant', L('E constante', 'constant E')]], S.tfs.em, 'E(m<sub>λ</sub>)', S.tfs.em === 'group' ? '0,284 + 0,272 ln²λ + 0,057 lnλ/λ, λ en µm'.replace(/,(?=\d)/g, getLang() === 'es' ? ',' : '.').replace(' en ', getLang() === 'es' ? ' en ' : ' in ') : '')}
    ${S.tfs.em === 'coeffs' ? `<div class="wr-row">${numIn('wr-ema', S.tfs.a, 'step="any"', 'a')}${numIn('wr-emb', S.tfs.b, 'step="any"', 'b')}${numIn('wr-emc', S.tfs.c, 'step="any"', 'c')}</div>` : ''}
    ${S.tfs.em === 'constant' ? numIn('wr-eme', S.tfs.E, 'min="0.01" max="1" step="0.01"', 'E') : ''}
    ${numIn('wr-sig', S.tfs.sigFrac * 100, 'min="0" max="50" step="0.5"', L('Máscara de señal baja, Σ<sub>j</sub> j<sub>j</sub> bajo este % del máximo', 'Low-signal mask, Σ<sub>j</sub> j<sub>j</sub> under this % of the maximum'))}
    ${numIn('wr-island', S.tfs.islandFrac * 100, 'min="0" max="50" step="1"', L('Islas sueltas, se quitan las zonas separadas con menos de este % del área de la zona mayor', 'Loose islands, separate regions with less than this % of the area of the largest region are removed'), t ? islandNote(t) : L('0 deja todas las zonas.', '0 keeps every region.'))}
    ${numIn('wr-zcut', S.tfs.zCut, 'step="any"', L('Quimioluminiscencia, alturas bajo z (mm)', 'Chemiluminescence, heights below z (mm)'), t?.zCutSuggest !== null && t?.zCutSuggest !== undefined ? zCutNote(t.zCutSuggest) : '')}
    ${numIn('wr-chem', S.tfs.chemThr * 100, 'min="0" max="200" step="1"', L('y píxeles con B/G sobre el predicho por T de R/G en más de este %', 'and pixels with B/G above the one predicted by T from R/G by more than this %'))}
    ${S.tfs.path === 'ratio' ? sel('wr-fsch', [['RG', L('rojo y verde (por defecto)', 'red and green (default)')], ['R', L('solo rojo', 'red only')], ['G', L('solo verde', 'green only')], ['RGB', L('avanzado, rojo, verde y azul', 'advanced, red, green and blue')], ['B', L('avanzado, solo azul', 'advanced, blue only')]], S.tfs.fsChannel, L('Canales para f<sub>s</sub>, mínimos cuadrados de j<sub>j</sub> = f<sub>s</sub> G<sub>j</sub>(T)', 'Channels for f<sub>s</sub>, least squares of j<sub>j</sub> = f<sub>s</sub> G<sub>j</sub>(T)'), fsChannelNote()) : ''}
    <p>${btn('wr-tfs', S.tfs.path === 'net' ? L('Calcular y entrenar la red', 'Compute and train the network') : L('Calcular', 'Compute'), !S.abel.out || S.tfs.net.running)} ${S.tfs.net.running ? btn('wr-net-stop', L('Detener', 'Stop')) : ''}</p>
    <p class="lab__readout">${t && p ? losReading(t, p) : ''}</p>
    ${S.tfs.path === 'net' && t && S.tfs.net.result ? `<p class="lab__readout" id="wr-net-xc">${netReading(S.tfs.net.result)}</p>` : ''}
  </div><div class="lab__plots">
    <figure><div id="wr-tab" class="wr-plot"></div><figcaption>${L(`Tabla T ↔ ${S.tfs.pair.split('').join('/')} de 1000 a 2600 K cada 1 K, con ψ<sub>j</sub>, Planck y E(m<sub>λ</sub>). Punteada, sensibilidad d ln(razón)/dT en % por 100 K, escalada al rango de la razón y sin eje propio.`, `Table T ↔ ${S.tfs.pair.split('').join('/')} from 1000 to 2600 K every 1 K, with ψ<sub>j</sub>, Planck and E(m<sub>λ</sub>). Dotted, sensitivity d ln(ratio)/dT in % per 100 K, scaled to the range of the ratio, without an axis of its own.`)}</figcaption></figure>
    ${t && S.tfs.path === 'fit3' ? `<figure><canvas id="wr-chi" class="wr-map"></canvas><figcaption>${L('χ² relativo del ajuste de tres canales, de 0 (fondo) a su percentil 99 (tinta). En la zona marcada como quimioluminiscencia el azul tiene peso 0.', 'Relative χ² of the three-channel fit, from 0 (background) to its 99th percentile (ink). In the region marked as chemiluminescence blue has weight 0.')}</figcaption></figure>` : ''}
  </div></div></div>
  ${t && S.tfs.path === 'net' && S.tfs.net.result ? `<figure class="wr-maps3 wr-netmaps"><div><canvas id="wr-net-tcf" class="wr-map"></canvas><span class="meta">T ${L('forma cerrada', 'closed form')}</span></div><div><canvas id="wr-net-dt" class="wr-map"></canvas><span class="meta">|T<sub>${L('red', 'net')}</sub> − T<sub>${L('cerrada', 'closed')}</sub>|</span></div><figcaption>${L(`Comprobación de la red. A la izquierda, T del ajuste directo por píxel de la misma emisión, con la escala de los mapas de la etapa 6. A la derecha, la diferencia absoluta con la red, de 0 (fondo) a ${num(XCHECK_WARN_P90_K)} K (tinta). Sin color quedan los píxeles fuera de la máscara y los que el ajuste directo deja en un borde de la tabla.`, `Check of the network. Left, T of the direct per-pixel fit of the same emission, with the scale of the stage 6 maps. Right, the absolute difference with the network, from 0 (background) to ${num(XCHECK_WARN_P90_K)} K (ink). Pixels outside the mask, and those the direct fit leaves at an edge of the table, have no colour.`)}</figcaption></figure>` : ''}
  <div class="wr-msgs">${msgHtml(msgs)}</div>`;
  drawTable();
  const nr = S.tfs.net.result;
  if (t && p && S.tfs.path === 'net' && nr) {
    const a = document.getElementById('wr-net-tcf') as HTMLCanvasElement | null, b = document.getElementById('wr-net-dt') as HTMLCanvasElement | null;
    const d = Float64Array.from(t.T, (v, q) => (fin(v) && fin(nr.Tcf[q]) ? Math.abs(v - nr.Tcf[q]) : NaN));
    const ext = mapExtent(p);
    if (a) { drawMap(a, nr.Tcf, p.nz, p.nr, 't', S.res.Tmin, S.res.Tmax); a.style.aspectRatio = `${ext.aspect}`; }
    if (b) { drawMap(b, d, p.nz, p.nr, 'fs', 0, XCHECK_WARN_P90_K); b.style.aspectRatio = `${ext.aspect}`; }
  }
  if (t && S.tfs.path === 'fit3') {
    const cv = document.getElementById('wr-chi') as HTMLCanvasElement | null;
    const v = Array.from(t.chi2rel).filter(fin).sort((a, b) => a - b);
    if (cv && p) drawMap(cv, t.chi2rel, p.nz, p.nr, 'fs', 0, v.length ? v[Math.floor(0.99 * (v.length - 1))] : 1);
  }
}
/** Panel of the RGB-ANNAbel option, iterations, estimated time, progress bar. */
function netPanel(): string {
  const N = S.tfs.net, e = N.estimate;
  const where = (path: string) => (path === 'webgpu' ? L('tarjeta gráfica', 'graphics card') : path === 'workers' ? L(`${helpers.length} hilos del procesador`, `${helpers.length} processor threads`) : L('un hilo del procesador', 'one processor thread'));
  const est = e ? L(`Camino ${where(e.path)}, ${num(e.sPerIter * 1000, 1)} ms por iteración, ${minText(e.sPerIter * N.iterations)} para ${num(N.iterations)} iteraciones.`, `Path ${where(e.path)}, ${num(e.sPerIter * 1000, 1)} ms per iteration, ${minText(e.sPerIter * N.iterations)} for ${num(N.iterations)} iterations.`) : '';
  const tk = N.tick;
  return `<p class="meta">${L('El segundo paso de RGB-ANNAbel (Magaña et al., Case Stud. Therm. Eng. 76 (2025) 107396). En el artículo el primer paso es ANNAbel. Aquí la red usa la emisión de la etapa 4, del método de inversión que hayan elegido. La red recibe la posición (r, z) y entrega T y f<sub>s</sub>. Se entrena para que f<sub>s</sub> G<sub>j</sub>(T) reproduzca la emisión j<sub>j</sub> de la etapa 4 en una de cada dos alturas, y para que las razones entre canales coincidan con las de la tabla donde hay señal. Reemplaza la lectura de la tabla píxel a píxel. La tabla sigue siendo el camino recomendado. Aquí se calcula también el ajuste directo por píxel, que sirve para revisar la red. La banda de incertidumbre de la etapa 6 usa ese ajuste directo.', 'The second step of RGB-ANNAbel (Magaña et al., Case Stud. Therm. Eng. 76 (2025) 107396). In the paper the first step is ANNAbel. Here the network uses the emission of stage 4, from whichever inversion method you chose. The network takes the position (r, z) and returns T and f<sub>s</sub>. It is trained so that f<sub>s</sub> G<sub>j</sub>(T) reproduces the emission j<sub>j</sub> of stage 4 on one height in two, and so that the ratios between channels match those of the table where there is signal. It replaces reading the table pixel by pixel. The table is still the recommended path. The direct per-pixel fit is computed here too, to check the network. The uncertainty band of stage 6 uses that direct fit.')} ${S.gpu ? L('Este navegador puede usar la tarjeta gráfica.', 'This browser can use the graphics card.') : L('Sin tarjeta gráfica corre en el procesador, más lento.', 'Without the graphics card it runs on the processor, more slowly.')}</p>
  ${numIn('wr-net-it', N.iterations, 'min="10" max="100000" step="100"', L('Iteraciones de la red', 'Network iterations'), L(`La herramienta usa ${num(NET_ITERATIONS)}. La aplicación usa ${num(STAGE2_DEFAULTS.iterations)}, pero con ellas la red queda lejos del ajuste directo en la llama de ejemplo.`, `The tool uses ${num(NET_ITERATIONS)}. The app uses ${num(STAGE2_DEFAULTS.iterations)}, but with them the network stays far from the direct fit on the example flame.`))}
  <p>${btn('wr-net-est', L('Estimar el tiempo', 'Estimate the time'), !S.abel.out || N.running)}</p>
  <p class="meta" id="wr-net-est-t">${est}</p>
  <progress id="wr-net-bar" class="wr-progress" max="${tk ? tk.total : N.iterations}" value="${tk ? tk.iter : 0}"${N.running || tk ? '' : ' hidden'} aria-label="${L('Avance del entrenamiento de la red', 'Progress of the network training')}"></progress>
  <p class="meta" id="wr-net-prog" aria-live="polite">${tk ? netProgressText() : ''}</p>`;
}
function netProgressText(): string {
  const t = S.tfs.net.tick;
  if (!t) return '';
  return L(`Iteración ${num(t.iter)} de ${num(t.total)}, pérdida ${sig(t.loss, 3)}, mejor ${sig(t.best, 3)} en la ${num(t.bestIter)}${t.etaS !== null && t.iter < t.total ? `, faltan unos ${num(Math.max(t.etaS, 0), 0)} s` : ''}.`, `Iteration ${num(t.iter)} of ${num(t.total)}, loss ${sig(t.loss, 3)}, best ${sig(t.best, 3)} at ${num(t.bestIter)}${t.etaS !== null && t.iter < t.total ? `, about ${num(Math.max(t.etaS, 0), 0)} s left` : ''}.`);
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
  if (x.bad.includes('T_median')) parts.push(L(`T difiere en ${num(x.TMedianAbsDiffK)} K en la mediana (umbral ${num(XCHECK_WARN_K)} K)`, `T differs by ${num(x.TMedianAbsDiffK)} K in the median (threshold ${num(XCHECK_WARN_K)} K)`));
  if (x.bad.includes('T_p90')) parts.push(L(`${num(x.TP90AbsDiffK)} K en el percentil 90 (umbral ${num(XCHECK_WARN_P90_K)} K)`, `${num(x.TP90AbsDiffK)} K at the 90th percentile (threshold ${num(XCHECK_WARN_P90_K)} K)`));
  if (x.bad.includes('fs')) parts.push(L(`f_s difiere en un factor ${num(Math.exp(x.fsMedianAbsLogRatio), 2)} en la mediana (umbral ${num(Math.exp(XCHECK_WARN_FS_LOG), 2)})`, `f_s differs by a factor ${num(Math.exp(x.fsMedianAbsLogRatio), 2)} in the median (threshold ${num(Math.exp(XCHECK_WARN_FS_LOG), 2)})`));
  return parts.join(L(' y ', ' and ')) + '.';
}
/** The _xcheck of the app in words, after training. */
function netReading(r: NetOut): string {
  const tr = L(`La red se entrenó ${num(r.completed)} de ${num(r.planned)} iteraciones en ${num(r.elapsedS, 1)} s (${r.path === 'webgpu' ? 'tarjeta gráfica' : 'procesador'}) con ${num(r.nTrain)} puntos de ${num(r.rows)} alturas, una de cada dos, y ${num(r.nMask)} puntos con señal para las razones. Mejor resultado en la iteración ${num(r.bestIter)}.`,
    `The network trained ${num(r.completed)} of ${num(r.planned)} iterations in ${num(r.elapsedS, 1)} s (${r.path === 'webgpu' ? 'graphics card' : 'processor'}) on ${num(r.nTrain)} points of ${num(r.rows)} heights, one in two, and ${num(r.nMask)} points with signal for the ratios. Best result at iteration ${num(r.bestIter)}.`);
  const x = r.xcheck;
  if (!x) return `${tr} ${L('Ningún píxel sirve para comparar con el ajuste directo.', 'No pixel is suitable for comparing with the direct fit.')}`;
  return `${tr} ${L(`Comparada con el ajuste directo por píxel de la misma emisión en ${num(x.nPixels)} píxeles con señal fuerte, T difiere en ${num(x.TMedianAbsDiffK)} K en la mediana y ${num(x.TP90AbsDiffK)} K en el percentil 90, y la razón f<sub>s</sub> red sobre f<sub>s</sub> directo tiene mediana ${num(x.fsMedianRatio, 2)}. ${x.agrees ? `Está dentro de los umbrales de la aplicación (${num(XCHECK_WARN_K)} K, ${num(XCHECK_WARN_P90_K)} K y un factor ${num(Math.exp(XCHECK_WARN_FS_LOG), 2)}).` : 'Queda fuera de los umbrales de la aplicación. Con más iteraciones la red suele acercarse al ajuste directo.'}`,
    `Compared with the direct per-pixel fit of the same emission on ${num(x.nPixels)} pixels with strong signal, T differs by ${num(x.TMedianAbsDiffK)} K in the median and ${num(x.TP90AbsDiffK)} K at the 90th percentile, and the ratio of network f<sub>s</sub> to direct f<sub>s</sub> has median ${num(x.fsMedianRatio, 2)}. ${x.agrees ? `It is within the thresholds of the app (${num(XCHECK_WARN_K)} K, ${num(XCHECK_WARN_P90_K)} K and a factor ${num(Math.exp(XCHECK_WARN_FS_LOG), 2)}).` : 'It is outside the thresholds of the app. More iterations usually bring the network closer to the direct fit.'}`)}`;
}
/** Hint under the f_s channels. R and G is the default of the presentation, blue is the advanced option (G4 r2, director). */
function fsChannelNote(): string {
  const m = S.tfs.fsChannel;
  if (m === 'RGB' || m === 'B') {
    const noB = !!S.cal && !(S.cal.calibrated ?? []).includes('B');
    return L(`Opción avanzada. El azul usa K<sub>B</sub>, y el K<sub>B</sub> del horno no vale en la llama porque casi toda la señal azul del horno viene de la cola sobre 700 nm. Úsenla solo con pasa-cortos de 700 nm o con la esfera.${m === 'RGB' ? ' En la zona de quimioluminiscencia el azul tiene peso 0.' : ''}${noB ? (m === 'RGB' ? ' Esta calibración no tiene K<sub>B</sub>, así que el resultado es el de rojo y verde.' : ' Esta calibración no tiene K<sub>B</sub>.') : ''}`,
      `Advanced option. Blue uses K<sub>B</sub>, and the oven K<sub>B</sub> does not hold in the flame because almost all the blue signal of the oven comes from the tail above 700 nm. Use it only with a 700 nm short-pass filter or with the sphere.${m === 'RGB' ? ' In the chemiluminescence zone blue has weight 0.' : ''}${noB ? (m === 'RGB' ? ' This calibration has no K<sub>B</sub>, so the result is that of red and green.' : ' This calibration has no K<sub>B</sub>.') : ''}`);
  }
  if (m === 'R' || m === 'G') return L('Con T de R/G, el rojo solo y el verde solo dan el mismo f<sub>s</sub> que rojo y verde juntos. Sirve para revisar.', 'With T from R/G, red alone and green alone give the same f<sub>s</sub> as red and green together. Useful as a check.');
  return L('Rojo y verde con el mismo peso, igual que la presentación y la cadena sintética. El azul queda fuera porque su K<sub>B</sub> del horno no vale en la llama. Rojo, verde y azul es una opción avanzada para cuando hay pasa-cortos o esfera.', 'Red and green with equal weight, as in the presentation and the synthetic chain. Blue is left out because its oven K<sub>B</sub> does not hold in the flame. Red, green and blue is an advanced option for when there is a short-pass filter or the sphere.');
}
/** What the suggested cut height does, and a word when it covers most of the flame (GB3r3-estudiante-12). */
function zCutNote(z: number): string {
  const p = S.flame.prepared;
  const big = !!p && z > p.zMm[0] + 0.5 * (p.zMm[p.nz - 1] - p.zMm[0]);
  return L(`Sugerencia ${num(z, 1)} mm, donde B/G deja de salir de la tabla. Si la escriben, bajo esa altura salen solo el par B/G y el azul del ajuste de tres canales. T y f<sub>s</sub> de R/G no cambian, y el camino recomendado no la aplica.${big ? ' Esa altura cubre más de la mitad de la llama. Con la razón R/G no hace falta escribirla.' : ''}`, `Suggestion ${num(z, 1)} mm, where B/G stops falling outside the table. If you type it, below that height only the B/G pair and blue in the three-channel fit are left out. T and f<sub>s</sub> from R/G do not change, and the recommended path does not apply it.${big ? ' That height covers more than half the flame. With the R/G ratio there is no need to type it.' : ''}`);
}
function losReading(t: TfsOut, p: Prepared): string {
  let a = 0, b = 0, n = 0, ax = 0, na = 0;
  for (let q = 0; q < t.T.length; q++) if (t.mask[q] === 1 && fin(t.T[q]) && fin(t.Tlos[q])) { a += t.T[q]; b += t.Tlos[q]; n++; if (q % p.nr === 0) { ax += t.Tlos[q] - t.T[q]; na++; } }
  const T = usedSorted(t.T);
  if (!n || !T.length) return '';
  const d = (b - a) / n, dAx = na ? ax / na : NaN;
  const more = (x: number) => (x >= 0 ? L(`${num(x)} K más`, `${num(x)} K more`) : L(`${num(-x)} K menos`, `${num(-x)} K less`));
  return L(`En la zona usada, T mediana ${num(pctOf(T, 0.5))} K, entre ${num(pctOf(T, 0.05))} y ${num(pctOf(T, 0.95))} K (percentiles 5 y 95). El máximo del campo no se lee como resultado, porque junta el ruido que la inversión amplifica en el eje. La razón de las proyecciones sin invertir, que promedia la línea de visión, da en promedio ${more(d)} que la T local${fin(dAx) ? ` y ${more(dAx)} en el eje` : ''}.`,
    `In the zone used, median T ${num(pctOf(T, 0.5))} K, between ${num(pctOf(T, 0.05))} and ${num(pctOf(T, 0.95))} K (5th and 95th percentiles). The maximum of the field is not read as a result, because it gathers the noise the inversion amplifies on the axis. The ratio of the projections without inversion, the line-of-sight average, gives on average ${more(d)} than the local T${fin(dAx) ? ` and ${more(dAx)} on the axis` : ''}.`);
}
function drawTable() {
  const ps = psiNow();
  const band = makeBand(ps.lamNm, ps.psi, 'gl', emLaw());
  const Tg = Float64Array.from({ length: 161 }, (_, k) => 1000 + 10 * k);
  const tab = ratioTable(band, Tg);
  const r = S.tfs.pair === 'RG' ? tab.RG : S.tfs.pair === 'BG' ? tab.BG : tab.BR;
  const sens = Array.from(r, (_, i) => { const a = Math.max(0, i - 1), b = Math.min(r.length - 1, i + 1); return (Math.log(r[b]) - Math.log(r[a])) / (Tg[b] - Tg[a]) * 100 * 100; });
  const lo = Math.min(...r), hi = Math.max(...r);
  const smax = Math.max(...sens.map(Math.abs)) || 1;
  plot('wr-tab', { xlabel: 'T (K)', ylabel: S.tfs.pair.split('').join('/'), height: 230 }, [
    { kind: 'line', x: Tg, y: r, cls: 's-data' },
    { kind: 'line', x: Tg, y: sens.map((s) => lo + ((s / smax + 1) / 2) * (hi - lo)), cls: 's-muted', dash: true },
  ]);
}

// ============================================================================ stage 6
function renderStage6() {
  const el = document.getElementById('wr-s6');
  if (!el) return;
  const t = S.tfs.out, p = S.flame.prepared;
  const d = calDecision();
  const fs = fsShown();
  const ok = !!(t && p && S.abel.out);
  // the oven bias only applies to a calibration with the real oven that fails the ratio test (GB3r1-estudiante-20, fisico-12)
  const tailOn = S.cal?.mode === 'oven' && !!S.cal.tail?.applied && !!S.flame.calUsed;
  const ovenFails = S.cal?.mode === 'oven' && (S.cal.ratioWorst ?? 0) > 50 && !!S.flame.calUsed && !tailOn;
  const biases = S.o.biases.filter((b) => (b.key === 'oven_dT_flame_conv' ? ovenFails : b.key === 'oven_tailcal_dT_flame' ? tailOn : true)).map((b) => `<li>${subs(esc(getLang() === 'es' ? b.es : b.en))}</li>`).join('');
  const hz = S.res.heights.filter((k) => fin(k) && p && k >= 0 && k < p.nz);
  // three colours and a legend inside the plot (GB3r3-estudiante-10)
  const legend = p ? L(`${hz.map((k) => `${num(p.zMm[k], 1)}`).join(', ')} mm, cada una con su color y su rótulo dentro del gráfico`, `${hz.map((k) => `${num(p.zMm[k], 1)}`).join(', ')} mm, each with its colour and its label inside the plot`) : '';
  const axes = p ? L(`r de −${num(p.rMm[p.nr - 1] + p.drMm / 2, 1)} a ${num(p.rMm[p.nr - 1] + p.drMm / 2, 1)} mm, z de ${num(p.zMm[0], 1)} mm abajo a ${num(p.zMm[p.nz - 1], 1)} mm arriba`, `r from −${num(p.rMm[p.nr - 1] + p.drMm / 2, 1)} to ${num(p.rMm[p.nr - 1] + p.drMm / 2, 1)} mm, z from ${num(p.zMm[0], 1)} mm at the bottom to ${num(p.zMm[p.nz - 1], 1)} mm at the top`) : '';
  const bandText = S.res.bandHow === 'oven' ? L(`La banda tiene ${S.res.nBand} repeticiones con el ruido de la diferencia entre mitades, el error del termopar como un ΔT común a los tres K<sub>j</sub>, la σ estadística de cada K<sub>j</sub> y la incertidumbre de la escala en f<sub>s</sub>. Semilla ${S.res.seed}.`, `The band has ${S.res.nBand} repetitions with the noise from the difference between halves, the thermocouple error as one ΔT common to the three K<sub>j</sub>, the statistical σ of each K<sub>j</sub> and the scale uncertainty on f<sub>s</sub>. Seed ${S.res.seed}.`)
    : S.res.bandHow === 'noise' ? L(`La banda tiene ${S.res.nBand} repeticiones con el ruido de la diferencia entre mitades y la incertidumbre de la escala en f<sub>s</sub>. No lleva la incertidumbre de la esfera ni del archivo de calibración. Semilla ${S.res.seed}.`, `The band has ${S.res.nBand} repetitions with the noise from the difference between halves and the scale uncertainty on f<sub>s</sub>. It does not carry the uncertainty of the sphere or of the calibration file. Seed ${S.res.seed}.`)
    : L(`La banda tiene ${S.res.nBand} repeticiones y lleva solo el ruido de la diferencia entre mitades. Semilla ${S.res.seed}.`, `The band has ${S.res.nBand} repetitions and carries only the noise from the difference between halves. Seed ${S.res.seed}.`);
  const bandLabel = S.flame.calUsed ? L('Calcular la banda de ruido y calibración', 'Compute the noise and calibration band') : L('Calcular la banda de ruido', 'Compute the noise band');
  el.innerHTML = `<h2 class="row__label nonum">${L('Etapa 6 · Resultados', 'Stage 6 · Results')}</h2>
  ${statusLine('wr-st6', staleNote(6) || (ok ? stateLabel(d) : L('Falta la etapa 5', 'Stage 5 is missing')))}
  ${ok ? `<div class="wr-grid3">
    ${numIn('wr-tmin', S.res.Tmin, 'step="10"', L('T mínima de la escala (K)', 'Scale minimum T (K)'))}
    ${numIn('wr-tmax', S.res.Tmax, 'step="10"', L('T máxima de la escala (K)', 'Scale maximum T (K)'))}
    ${fs ? numIn('wr-fsmax', +S.res.fsMax.toPrecision(3), 'min="0" step="any"', L(`f<sub>s</sub> máxima de la escala (${fs.unit})`, `Scale maximum f<sub>s</sub> (${fs.unit})`), L('Por defecto el percentil 95 de la zona usada.', 'By default the 95th percentile of the zone used.')) : ''}
    <p class="meta">${L('La escala de T sale por defecto de los percentiles 0,5 y 99,5 de la zona usada.', 'The T scale comes by default from the 0.5th and 99.5th percentiles of the zone used.')}${S.res.Tout ? L(` ${num(S.res.Tout)} píxeles de la zona usada quedan fuera de la escala y se pintan con el color del extremo.`, ` ${num(S.res.Tout)} pixels of the zone used fall outside the scale and are painted with the end colour.`) : ''}</p>
  </div>
  <div class="wr-maps">
    <figure><div class="wr-mapax" id="wr-map-T-ax"><canvas id="wr-map-T" class="wr-map"></canvas></div><canvas id="wr-bar-T" class="wr-bar"></canvas><p class="wr-barlab"><span>${num(S.res.Tmin)}</span><span>T (K)</span><span>${num(S.res.Tmax)}</span></p><figcaption>${L(`T(r, z), viridis. ${stateLabel(d)}. Proporción real 2R × Δz, zonas enmascaradas como fondo. ${axes}.`, `T(r, z), viridis. ${stateLabel(d)}. True aspect ratio 2R × Δz, masked regions as background. ${axes}.`).replace(/f_s/g, 'f<sub>s</sub>')}</figcaption></figure>
    ${fs ? `<figure><div class="wr-mapax" id="wr-map-fs-ax"><canvas id="wr-map-fs" class="wr-map"></canvas></div><canvas id="wr-bar-fs" class="wr-bar"></canvas><p class="wr-barlab"><span>0</span><span>f<sub>s</sub> (${fs.unit})</span><span>${sig(S.res.fsMax, 2)}</span></p><figcaption>${L(`f<sub>s</sub>(r, z), más hollín más tinta. ${stateLabel(d)}.${fs.ref ? ` Relativa a ${fs.ref}.` : ''}`, `f<sub>s</sub>(r, z), more soot more ink. ${stateLabel(d)}.${fs.ref ? ` Relative to ${fs.ref}.` : ''}`).replace(/f_s/g, 'f<sub>s</sub>')}</figcaption></figure>` : `<figure><p class="meta">${L('Sin f<sub>s</sub> con esta calibración.', 'No f<sub>s</sub> with this calibration.')}</p></figure>`}
    <figure><div class="wr-mapax" id="wr-map-los-ax"><canvas id="wr-map-los" class="wr-map"></canvas></div><figcaption>${L('T de la razón de las proyecciones sin invertir, rotulada promedio en la línea de visión, misma escala que T. Comparación didáctica.', 'T from the ratio of the projections without inversion, labelled line-of-sight average, same scale as T. Teaching comparison.')}</figcaption></figure>
  </div>
  <h3>${L('Perfiles radiales', 'Radial profiles')}</h3>
  <div class="wr-row">${[0, 1, 2].map((i) => numIn(`wr-hz${i}`, fin(S.res.heights[i]) ? +p!.zMm[S.res.heights[i]].toFixed(2) : NaN, 'step="any"', L(`Altura ${i + 1} (mm)`, `Height ${i + 1} (mm)`))).join('')}</div>
  <p>${numIn('wr-nband', S.res.nBand, 'min="5" max="200" step="1"', L('Repeticiones de la banda', 'Band repetitions'))} ${numIn('wr-seed6', S.res.seed, 'min="0" step="1"', L('Semilla', 'Seed'))} ${btn('wr-band', bandLabel)}</p>
  <div class="wr-grid2"><figure><div id="wr-profT" class="wr-plot"></div><figcaption>${L(`T(r) a las alturas elegidas, ${legend}. Banda gris, percentiles 16 y 84 de la banda. Un tramo sin línea es zona enmascarada.`, `T(r) at the chosen heights, ${legend}. Grey band, 16th and 84th percentiles of the band. A stretch without a line is a masked region.`)}</figcaption></figure>
  ${fs ? `<figure><div id="wr-profF" class="wr-plot"></div><figcaption>${L(`f<sub>s</sub>(r) en ${fs.unit} a las mismas alturas y con las mismas líneas, con su banda. El eje llega al doble de la escala del mapa y lo que pasa de ahí queda recortado, casi siempre ruido cerca del eje.`, `f<sub>s</sub>(r) in ${fs.unit} at the same heights and with the same lines, with its band. The axis reaches twice the map scale and anything above it is clipped, nearly always noise near the axis.`)}</figcaption></figure>` : ''}</div>
  <p class="meta">${S.res.band ? bandText : L('La banda aparece al calcularla.', 'The band appears once computed.')}</p>
  <h4>${L('Sesgos que no entran en la banda', 'Biases not in the band')}</h4>
  <ul>${biases}</ul>` : ''}
  <h3>${L('Descargas', 'Downloads')}</h3>
  <p class="cluster">${btn('wr-dl-csv', 'CSV', !ok)} ${getLang() === 'es' ? `${btn('wr-dl-csvpt', 'CSV con punto decimal', !ok)} ` : ''}${btn('wr-dl-png-T', L('PNG de T', 'PNG of T'), !ok)} ${btn('wr-dl-png-fs', L('PNG de f<sub>s</sub>', 'PNG of f<sub>s</sub>'), !ok || !fs)} ${btn('wr-dl-run', L('Registro run.json', 'Log run.json'), !S.items.length)} ${btn('wr-dl-zip', L('Descargar todo .zip', 'Download everything .zip'), !S.items.length)}</p>
  ${ok ? `<p class="cluster wr-npy">NPY ${Object.keys(npyFields()).map((k) => `<button class="btn" type="button" data-npy="${k}">${k}</button>`).join(' ')}</p>` : ''}
  <p class="meta">${L('El CSV en español usa punto y coma entre columnas y coma decimal. El CSV con punto decimal usa coma entre columnas, como verdad_T_fs.csv de la llama sintética. La columna mask_bits suma 1 si el píxel se usó, 2 si la señal es baja, 4 si hay quimioluminiscencia, 8 si la altura está saturada y 16 si la razón queda fuera de la tabla.', 'The CSV uses commas between columns and a decimal point, like verdad_T_fs.csv of the synthetic flame. The mask_bits column adds 1 if the pixel was used, 2 for low signal, 4 for chemiluminescence, 8 for a saturated height and 16 for a ratio outside the table.')}</p>
  <p class="meta">${L('El registro guarda la versión de la herramienta y su identificador, el SHA-256 de cada archivo de entrada, todos los ajustes, la semilla, la ley de E(m) y la curva ψ<sub>j</sub>. Cargado en la etapa 0 rearma el cálculo. No lleva las imágenes. Los PNG salen con el tema de la página y lo dicen en el nombre.', 'The log keeps the tool version and its identifier, the SHA-256 of each input file, every setting, the seed, the E(m) law and the ψ<sub>j</sub> curve. Loaded in stage 0 it rebuilds the calculation. It does not contain the images. The PNG files use the page theme and say so in the name.')}</p>`;
  if (ok) drawResults();
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
    'wr-tail': () => { S.tailOn = (t as HTMLInputElement).checked; if (S.oven.result) setCalFromOven(); renderAll(); },
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
    'wr-burnerok': () => { S.flame.burnerOk = (t as HTMLInputElement).checked; },
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
    'wr-lam': () => { S.abel.lambdaRel = 10 ** val; const o = t.parentElement?.querySelector('output'); if (o) o.textContent = sig(S.abel.lambdaRel, 2); invalidate(4, 'abel'); },
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
    const o = t.parentElement?.querySelector('output'); if (o && S.flame.prepared) o.textContent = `${num(S.flame.prepared.zMm[S.flame.height], 2)} mm`;
    drawFlame(); drawAbel();
    return;
  }
  const f = map[id];
  if (!f) return;
  f();
  const live = new Set(['wr-eps', 'wr-lam', 'wr-pcx', 'wr-pcy', 'wr-pr']);
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
    const pv = S.flame.rulerPreview, r = t.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * pv.w * pv.step, y = ((e.clientY - r.top) / r.height) * pv.h * pv.step;
    if (S.flame.rulerPts.length >= 2) S.flame.rulerPts = [];
    S.flame.rulerPts.push([x, y]);
    rulerScale();
    keepFocus(renderStage3);
    return;
  }
  const actions: Record<string, () => unknown> = {
    'wr-reco': recommended,
    'wr-q-use': async () => { const q = S.quick.map((it) => ({ path: it.path, blob: it.blob })); S.quick = []; await loadFiles(q, L('revisión rápida', 'quick check')); },
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
    'wr-abel': runAbelNow,
    'wr-ann-stop': () => S.abel.annabel.ctrl?.abort(),
    'wr-net-stop': () => S.tfs.net.ctrl?.abort(),
    'wr-net-est': estimateNet,
    'wr-ann-est': estimateAnnabel,
    'wr-lc': runLcurve,
    'wr-lc-use': () => { if (S.abel.lcurve) { S.abel.lambdaRel = S.abel.lcurve.lambda; invalidate(4, 'abel'); renderAll(); } },
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
  if (id === 'wr-o-dir' || id === 'wr-o-files' || id === 'wr-f-dir' || id === 'wr-f-files') { const f = filesFromInput(t); await loadFiles(f, f.length === 1 ? f[0].path : L(`${f.length} archivos`, `${f.length} files`)); return; }
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
  root.innerHTML = `<p class="wr-busy" id="wr-busy" role="status" aria-live="polite" hidden></p>
  <p class="sr-only" id="wr-announce" role="status" aria-live="polite"></p><p class="sr-only" id="wr-alert" role="alert"></p>
  ${[0, 1, 2, 3, 4, 5, 6].map((k) => `<section class="section row wr-stage" id="wr-s${k}"></section>`).join('')}`;
  root.addEventListener('input', onInput);
  root.addEventListener('change', (e) => { const t = e.target as HTMLInputElement; if (t.type === 'file') onFiles(e); else onInput(e); });
  root.addEventListener('click', (e) => { onClick(e as MouseEvent); });
  root.addEventListener('keydown', onKey);
  root.addEventListener('pointerdown', () => { pointerDown = true; });
  window.addEventListener('pointerup', () => { pointerDown = false; if (pendingRender) setTimeout(() => { if (pendingRender && !pointerDown) { pendingRender = false; renderAll(); } }, 0); });
  root.addEventListener('dragover', (e) => { const d = (e.target as HTMLElement).closest('.wr-drop'); if (d) { e.preventDefault(); d.classList.add('wr-over'); } });
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
