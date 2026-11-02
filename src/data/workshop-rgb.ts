/**
 * Shared data of the RGB pyrometry workshop pages (Metrology in Fire Science, Universidad Adolfo Ibáñez).
 *
 * Numbers printed on the pages come from `workshop-rgb-checks.json`, an export of the presentation's
 * `data/checks.json` written by `manim/tools/make_web_checks.py`. The example data manifest comes from
 * `manim/tools/make_web_samples.py`. Neither file is edited by hand.
 *
 * The tool page, the offline copy and the presentation are built elsewhere. The pages check at build
 * time whether they exist and only link them when they do, so no page points to a missing file.
 */
import { existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import checks from './workshop-rgb-checks.json';
import samples from './workshop-rgb-samples.json';
import type { Lang } from '../i18n/ui';

export const WS = '/teaching/workshop-rgb-pyrometry/';
export const FILES = `${WS}files/`;
export const GUIDE_PDF = { es: 'guia_laboratorio_pirometria_rgb.pdf', en: 'lab_guide_rgb_pyrometry.pdf' } as const;
export const LOG_TEMPLATE = 'registro_plantilla.csv';

type Check = { es: string; en: string; unidad?: string | null; esquema: boolean; cond_es?: string; cond_en?: string };
const C = checks as unknown as Record<string, Check>;

/** Screen string of a checks.json key, decimal comma in Spanish. */
export function ck(key: string, lang: Lang): string {
  const v = C[key];
  if (!v) throw new Error(`workshop-rgb: key ${key} missing from workshop-rgb-checks.json`);
  return lang === 'es' ? v.es : v.en;
}

/** Condition that checks.json puts next to the figure on screen (pantalla_cond), as a clause (a semicolon becomes a comma). */
export function ckCond(key: string, lang: Lang): string {
  const v = C[key];
  if (!v) throw new Error(`workshop-rgb: key ${key} missing from workshop-rgb-checks.json`);
  return (lang === 'es' ? v.cond_es : v.cond_en)?.replace(/;\s*/g, ', ') ?? '';
}

/** True when checks.json marks the value as illustrative (*Esquema*). */
export function isSketch(key: string): boolean {
  return C[key]?.esquema === true;
}

/** Values the example button fills in stage 3 (from the LEEME of the set). */
export interface SamplePreset { texpUs: number; mmPerPx: number; zBurnerMm: number }
export interface SampleSet { file: string; bytes: number; sha256: string; images: number; camera: string; preset?: SamplePreset }
export const sampleSets = (samples as { sets: SampleSet[] }).sets;
export const sampleTotal = (samples as { total_bytes: number }).total_bytes;
export const sample = (file: string): SampleSet => {
  const s = sampleSets.find((x) => x.file === file);
  if (!s) throw new Error(`workshop-rgb: ${file} missing from workshop-rgb-samples.json`);
  return s;
};

/** Size in MB with one decimal and the language's decimal mark. */
export function mb(bytes: number, lang: Lang): string {
  const s = (bytes / 1e6).toFixed(1);
  return `${lang === 'es' ? s.replace('.', ',') : s} MB`;
}

// ---- what exists at build time (paths relative to the project root, where astro build runs)
const root = process.cwd();
const pub = (p: string) => join(root, 'public', p.replace(/^\//, ''));
const toolRoutes = import.meta.glob('/src/pages/teaching/workshop-rgb-pyrometry/tool/index.astro');

export const has = {
  tool: Object.keys(toolRoutes).length > 0,
  presentation: existsSync(pub(`${WS}presentacion/index.html`)),
  presentationEn: existsSync(pub(`${WS}presentacion/en/index.html`)),
  guidePdf: { es: existsSync(pub(`${FILES}${GUIDE_PDF.es}`)), en: existsSync(pub(`${FILES}${GUIDE_PDF.en}`)) },
};

/** File name prefix of the single-file offline copy in each language (one file per language, GB3 r1). */
export const OFFLINE_PREFIX = { es: 'herramienta_pirometria_rgb_', en: 'rgb_pyrometry_tool_' } as const;

/** Single-file offline copy of the tool in the page's language, if one was published in files/. */
export function offlineCopy(lang: Lang): string | undefined {
  const dir = pub(FILES);
  if (!existsSync(dir)) return undefined;
  const pre = OFFLINE_PREFIX[lang];
  return readdirSync(dir).filter((f) => f.startsWith(pre) && f.endsWith('.html')).sort().pop();
}

/** True when a presentation folder holds a playable deck (index.html). Under the size budget only the dark decks may be published. */
export function deckHtml(folder: string): boolean {
  return existsSync(pub(`${WS}${folder}index.html`));
}

/** First existing PDF of a presentation folder, named like the deck (workshop_rgb_<lang>[_sala].pdf). */
export function deckPdf(folder: string): string | undefined {
  const dir = pub(`${WS}${folder}`);
  if (!existsSync(dir)) return undefined;
  return readdirSync(dir).find((f) => f.endsWith('.pdf'));
}

/**
 * Plain text to HTML for the workshop prose. Escapes the text, then typesets symbols with a
 * subscript (t_exp, K_R/K_G, f_s, j_λ, λ_ref…) as an italic letter with its subscript, so the pages
 * read like the presentation. File names such as T_termopar_C or numero_f are left alone.
 */
export function tx(s: string): string {
  const esc = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return esc.replace(/(?<![A-Za-z0-9_.])([A-Za-zλψηκε])_(exp|ref|abs|[A-Za-zλ])(?![A-Za-z0-9_])/g,
    (_m, a: string, b: string) => `<i>${a}</i><sub>${b}</sub>`);
}

/** Share of the signal above 700 nm per channel at the oven and at a flame, as a small table (GB3r1-estudiante-32). */
export function tailTable(lang: Lang): { head: string[]; rows: string[][] } {
  const es = lang === 'es';
  return {
    head: [es ? 'Cuerpo negro a' : 'Blackbody at', 'R', 'G', 'B'],
    rows: [
      [es ? '1273 K, horno' : '1273 K, oven', ck('tail700_R_1273', lang), ck('tail700_G_1273', lang), ck('tail700_B_1273', lang)],
      [es ? '1800 K, llama' : '1800 K, flame', ck('tail700_R_1800', lang), ck('tail700_G_1800', lang), ck('tail700_B_1800', lang)],
    ],
  };
}

/** Label for an illustrative value (*Esquema* in checks.json). */
export const sketchLabel = (lang: Lang) => (lang === 'es' ? 'valor ilustrativo' : 'illustrative value');
