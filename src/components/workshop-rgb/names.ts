/**
 * Reading of oven temperature, exposure time, acquisition time and role from a relative path
 * (WEB_SPEC §5.2), registro.csv, pylon .pfs files and recorte.json. From the stage 1a maquette
 * (manim/qa/web/maquette/src/names.ts), with the exposure also read after a prefix
 * (`fondo_208exp_00.tiff`, `promedio_128cuadros_208exp.tiff`) and parse_time_from_name of
 * annabel_app/core/io.py. No prose here, the interface translates roles and units.
 */

export type Role = 'oven' | 'dark' | 'flat' | 'sphere' | 'flame' | 'background' | 'ruler' | 'ignore';
export const ROLES: Role[] = ['oven', 'dark', 'flat', 'sphere', 'flame', 'background', 'ruler', 'ignore'];

export interface NameInfo {
  path: string;
  base: string;
  role: Role;
  /** Oven set point in °C, from a folder such as 1000C/ or 1273K/. */
  tC?: number;
  tUnit?: 'C' | 'K';
  texpUs?: number;
  /** How the exposure was written, for example "0,5 ms". */
  texpAs?: string;
  frame?: number;
  model?: string;
  serial?: string;
  /** Acquisition time in s (ANNAbel's `12s.tiff`), not an exposure time. */
  timeS?: number;
  /** Folder under llama/ (the flame condition). */
  condition?: string;
}

const NUM = String.raw`(\d+(?:[.,]\d+)?)`;
const RE_T = new RegExp(String.raw`^${NUM}\s*°?\s*(C|K)$`, 'i');
const RE_TEXP_FILE = new RegExp(String.raw`(?:^|[_\s-])${NUM}\s*(exp|us|µs|μs|ms)(?:_(\d+))?\.tiff?$`, 'i');
const RE_TEXP_DIR = new RegExp(String.raw`^${NUM}\s*(exp|us|µs|μs|ms)$`, 'i');
const RE_PYLON = /^Basler_(.+?)__([^_]+)__.*?_(\d+)\.tiff?$/i;
/** exp before the number, `exp20000.tiff` or `texp_500us.tiff` (GB3r2-adv-07). */
const RE_TEXP_PRE = new RegExp(String.raw`(?:^|[_\s-])t?exp[_\s-]?${NUM}\s*(us|µs|μs|ms)?(?:_(\d+))?\.tiff?$`, 'i');

/** Folder names as typed on a Spanish keyboard, 1050ºC (ordinal º), 1050 grados, 1050 °C (GB3r2-adv-07). */
export function normDir(d: string): string {
  return d.trim().replace(/[º˚]/g, '°').replace(/\s*grados?(\s*(c|celsius))?$/i, ' °C');
}
/** File name without the copy marks that Finder and Windows add, ` (1)`, ` copia`, ` - copia`, ` copy 2` (GB3r2-adv-07). */
export function stripCopy(base: string): string {
  return base.replace(/(?:\s*-\s*(?:copia|copy)|\s+(?:copia|copy)(?:\s*\d+)?|\s*\(\d+\))+(?=\.tiff?$)/i, '');
}

const toNum = (s: string) => Number(s.replace(',', '.'));
const toUs = (v: number, unit: string) => (unit.toLowerCase() === 'ms' ? v * 1000 : v);

/** Files the tool skips without a word (macOS and Windows litter). */
export function ignoredSilently(path: string): boolean {
  const parts = path.split('/');
  const base = parts[parts.length - 1];
  return parts.includes('__MACOSX') || base.startsWith('._') || base.startsWith('.') ||
    /^thumbs\.db$/i.test(base) || /^desktop\.ini$/i.test(base);
}

export const isTiff = (p: string) => /\.tiff?$/i.test(p);

const TIME_PATTERNS: [RegExp, number][] = [
  [/(?<![\d.])(\d+(?:\.\d+)?)\s*(?:ms)(?![a-z])/gi, 1e-3],
  [/(?<![\d.])(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)(?![a-z])/gi, 1],
  [/(?:^|[_\-\s])t\s*=?\s*(\d+(?:\.\d+)?)(?![\d.])/gi, 1],
];

/** parse_time_from_name of annabel_app/core/io.py, s, undefined when absent or ambiguous. */
export function parseTimeFromName(name: string): number | undefined {
  const stem = name.replace(/\.[^.]+$/, '').split('/').pop() ?? '';
  if (RE_TEXP_FILE.test(name) || /exp/i.test(stem)) return undefined;
  const vals: number[] = [];
  for (const [re, sc] of TIME_PATTERNS) for (const m of stem.matchAll(re)) vals.push(Number(m[1]) * sc);
  const uniq = [...new Set(vals.map((v) => Math.round(v * 1e12) / 1e12))];
  return uniq.length === 1 ? vals[0] : undefined;
}

export function parseName(path: string): NameInfo {
  const parts = path.split('/').filter(Boolean);
  const base = parts[parts.length - 1] ?? path;
  const dirs = parts.slice(0, -1).map(normDir);
  const low = dirs.map((d) => d.toLowerCase());
  const info: NameInfo = { path, base, role: 'oven' };

  for (const d of dirs) {
    const m = d.match(RE_T);
    if (m) { const v = toNum(m[1]); info.tUnit = m[2].toUpperCase() as 'C' | 'K'; info.tC = info.tUnit === 'K' ? v - 273.15 : v; }
    const e = d.match(RE_TEXP_DIR);
    if (e) { info.texpUs = toUs(toNum(e[1]), e[2]); info.texpAs = `${e[1]} ${e[2]}`; }
  }
  const clean = stripCopy(base);
  const f = clean.match(RE_TEXP_FILE);
  const fp = f ? null : clean.match(RE_TEXP_PRE);
  if (f) {
    info.texpUs = toUs(toNum(f[1]), f[2]);
    info.texpAs = `${f[1]} ${f[2]}`;
    if (f[3]) info.frame = Number(f[3]);
  } else if (fp) {
    info.texpUs = toUs(toNum(fp[1]), fp[2] ?? 'us');
    info.texpAs = `exp ${fp[1]}${fp[2] ? ` ${fp[2]}` : ''}`;
    if (fp[3]) info.frame = Number(fp[3]);
  }
  const p = base.match(RE_PYLON);
  if (p) { info.model = p[1]; info.serial = p[2]; info.frame = Number(p[3]); }
  if (!p && !f && !fp) info.timeS = parseTimeFromName(clean);

  const has = (...names: string[]) => low.some((d) => names.includes(d));
  if (/^(regla|ruler)/i.test(base)) info.role = 'ruler';
  else if (has('campo_plano', 'flat')) info.role = low.includes('oscuro') || low.includes('dark') ? 'dark' : 'flat';
  else if (has('oscuro', 'dark') || /_oscuro\.tiff?$/i.test(base)) info.role = 'dark';
  else if (has('esfera', 'sphere')) info.role = low.includes('oscuro') ? 'dark' : 'sphere';
  else if (has('fondo', 'background') || /^fondo/i.test(base)) info.role = 'background';
  else if (has('llama', 'flame')) info.role = 'flame';
  const li = low.findIndex((d) => d === 'llama' || d === 'flame');
  if (li >= 0 && dirs[li + 1] && !RE_TEXP_DIR.test(dirs[li + 1]) && !/^fondo$/i.test(dirs[li + 1])) info.condition = dirs[li + 1];
  if (info.role === 'oven' && info.tC === undefined && info.timeS !== undefined) info.role = 'flame';
  return info;
}

/** registro.csv of the lab guide, carpeta, T_termopar_C, texp_us, ganancia_dB, numero_f, hora, nota. */
export interface LogRow { carpeta: string; tC?: number; texpUs?: number }
export function parseLog(text: string): LogRow[] {
  const lines = text.replace(/^﻿/, '').split(/\r?\n/).filter((l) => l.trim());
  if (!lines.length) return [];
  const sep = lines[0].includes(';') ? ';' : lines[0].includes('\t') ? '\t' : ',';
  const head = lines[0].split(sep).map((h) => h.trim().toLowerCase());
  const ic = head.indexOf('carpeta'), it = head.indexOf('t_termopar_c'), ie = head.indexOf('texp_us');
  if (ic < 0) return [];
  const cell = (row: string[], i: number) => (i >= 0 && row[i] !== undefined && row[i].trim() !== '' ? toNum(row[i].trim()) : undefined);
  return lines.slice(1).map((l) => {
    const r = l.split(sep);
    return { carpeta: (r[ic] ?? '').trim().replace(/\/+$/, ''), tC: cell(r, it), texpUs: cell(r, ie) };
  });
}

/** Row of registro.csv that describes a file, the longest `carpeta` that is a folder of its path. */
export function logFor(path: string, rows: LogRow[]): LogRow | undefined {
  const dir = path.split('/').slice(0, -1).join('/');
  let best: LogRow | undefined;
  for (const r of rows) {
    if (!r.carpeta) continue;
    const c = r.carpeta.replace(/^\.\//, '');
    const hit = dir === c || dir.endsWith('/' + c) || dir.startsWith(c + '/') || dir.includes('/' + c + '/');
    if (hit && (!best || c.length > best.carpeta.length)) best = r;
  }
  return best;
}

/** pylon feature file (.pfs), lines "Name<TAB>value". Only the keys stage 1a shows. */
export const PFS_KEYS = ['Gain', 'BlackLevel', 'BalanceRatio', 'PixelFormat', 'ExposureTime', 'ReverseX', 'ReverseY', 'OffsetX', 'OffsetY', 'Gamma'];
export function parsePfs(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split(/\r?\n/)) {
    if (!line || line.startsWith('#')) continue;
    const m = line.match(/^([A-Za-z]+)\s+(.+)$/);
    if (m && PFS_KEYS.includes(m[1]) && !(m[1] in out)) out[m[1]] = m[2].trim();
  }
  return out;
}

/** recorte.json, the origin of a crop on the sensor, { x0, y0 } or per folder { "1000C": { x0, y0 } }. */
export type CropMap = Record<string, { x0: number; y0: number; sensor?: [number, number] }>;
export function parseCrop(text: string): CropMap {
  const j = JSON.parse(text) as Record<string, unknown>;
  const pick = (o: Record<string, unknown>) => ({ x0: Number(o.x0 ?? o.x ?? 0), y0: Number(o.y0 ?? o.y ?? 0), sensor: Array.isArray(o.sensor) ? (o.sensor as [number, number]) : undefined });
  if (typeof j.x0 === 'number' || typeof j.y0 === 'number') return { '*': pick(j) };
  const out: CropMap = {};
  for (const [k, v] of Object.entries(j)) if (v && typeof v === 'object') out[k] = pick(v as Record<string, unknown>);
  return out;
}
/** Folder key for recorte.json, the same reading as the temperature, so 1050ºC, 1050 C and 1050C match (GB3r3-adv-06). */
function cropKey(d: string): string {
  const n = normDir(d);
  const m = n.match(RE_T);
  return m ? `${toNum(m[1])}${m[2].toUpperCase()}` : n.toLowerCase();
}
export function cropFor(path: string, crops: CropMap): { x0: number; y0: number } | undefined {
  const byKey = new Map(Object.entries(crops).map(([k, v]) => [k === '*' ? '*' : cropKey(k), v]));
  const parts = path.split('/');
  for (let k = parts.length - 2; k >= 0; k--) { const c = byKey.get(cropKey(parts[k])); if (c) return c; }
  return byKey.get('*');
}
