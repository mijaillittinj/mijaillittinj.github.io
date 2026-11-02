/**
 * Canvas drawing of the tool. Previews in grey with gamma 2,2 (for viewing only, rotulado), flag marks,
 * circles and lines over the previews, and the result maps with the colour maps of utils/maps.py
 * (src/lib/workshop-rgb/colormaps.ts), true aspect ratio, masked pixels as the page background and never
 * as zero, with a colour bar and its units. PNG export draws the same into an off-screen canvas.
 */
import { colorize, table, type MapName, type Theme } from '../../lib/workshop-rgb/colormaps.ts';

export function theme(): Theme {
  const t = document.documentElement.dataset.theme;
  if (t === 'dark' || t === 'light') return t;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#ffffff';
}
/** The page background as #rrggbb (masked pixels). */
export function paperHex(): string {
  const v = cssVar('--paper');
  if (/^#[0-9a-f]{6}$/i.test(v)) return v;
  if (/^#[0-9a-f]{3}$/i.test(v)) return '#' + v.slice(1).split('').map((c) => c + c).join('');
  const m = v.match(/\d+/g);
  return m && m.length >= 3 ? '#' + m.slice(0, 3).map((x) => (+x).toString(16).padStart(2, '0')).join('') : theme() === 'dark' ? '#16171a' : '#fefefe';
}

export interface Planes { w: number; h: number; step: number; v: Float32Array; flag: Uint8Array }

/** Grey preview of channel c (or the sum when c = 3) with gamma 2,2 and the flag colours. */
export function drawPreview(cv: HTMLCanvasElement, p: Planes, c: number, marks = true, gain = 1): CanvasRenderingContext2D {
  cv.width = p.w; cv.height = p.h;
  const g = cv.getContext('2d')!;
  const im = g.createImageData(p.w, p.h);
  for (let i = 0; i < p.w * p.h; i++) {
    const val = c < 3 ? p.v[3 * i + c] : (p.v[3 * i] + p.v[3 * i + 1] + p.v[3 * i + 2]) / 3;
    const y = 255 * Math.pow(Math.min(1, Math.max(0, val * gain)), 1 / 2.2);
    let r = y, gg = y, b = y;
    if (marks) {
      const f = p.flag[i];
      if (f & 1) { r = 230; gg = 30; b = 30; } else if (f & 4) { r = 250; gg = 210; b = 0; } else if (f & 2) { r = 40; gg = 90; b = 240; }
    }
    im.data[4 * i] = r; im.data[4 * i + 1] = gg; im.data[4 * i + 2] = b; im.data[4 * i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return g;
}

/** Grey image of a single plane already in [0, 1]. */
export function drawGrey(cv: HTMLCanvasElement, v: Float32Array, w: number, h: number, gamma = true): CanvasRenderingContext2D {
  cv.width = w; cv.height = h;
  const g = cv.getContext('2d')!;
  const im = g.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const y = 255 * (gamma ? Math.pow(Math.min(1, Math.max(0, v[i])), 1 / 2.2) : Math.min(1, Math.max(0, v[i])));
    im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = y; im.data[4 * i + 3] = 255;
  }
  g.putImageData(im, 0, 0);
  return g;
}

export function circle(g: CanvasRenderingContext2D, cx: number, cy: number, r: number, color: string, dashed = false, width = 1.5) {
  g.save(); g.strokeStyle = color; g.lineWidth = width; if (dashed) g.setLineDash([4, 3]);
  g.beginPath(); g.arc(cx, cy, Math.max(0.5, r), 0, 2 * Math.PI); g.stroke(); g.restore();
}
export function line(g: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, color: string, dashed = false, width = 1.2) {
  g.save(); g.strokeStyle = color; g.lineWidth = width; if (dashed) g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke(); g.restore();
}

/**
 * Map of a field (nz × nr, row 0 the lowest height) mirrored about the axis, so the canvas is 2 nr wide
 * and nz tall with the highest z on top, the true aspect ratio 2R × Δz.
 */
export function drawMap(cv: HTMLCanvasElement, f: ArrayLike<number>, nz: number, nr: number, name: MapName, vmin: number, vmax: number, mask?: ArrayLike<number> | null): void {
  const W = 2 * nr, H = nz;
  const vals = new Float64Array(W * H), m = new Uint8Array(W * H);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) {
    const y = nz - 1 - k, v = f[k * nr + i], ok = mask ? (mask[k * nr + i] ? 1 : 0) : 1;
    vals[y * W + nr + i] = v; vals[y * W + nr - 1 - i] = v;
    m[y * W + nr + i] = ok; m[y * W + nr - 1 - i] = ok;
  }
  const rgba = colorize(vals, name, vmin, vmax, theme(), paperHex(), m);
  cv.width = W; cv.height = H;
  const g = cv.getContext('2d')!;
  const im = g.createImageData(W, H);
  im.data.set(rgba);
  g.putImageData(im, 0, 0);
}

/** Round tick values between a and b, about n of them (steps of 1, 2 or 5 × 10^k). */
export function niceTicks(a: number, b: number, n = 5): number[] {
  const span = b - a;
  if (!(span > 0)) return [];
  const raw = span / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * p).find((x) => x >= raw) ?? 10 * p;
  const out: number[] = [];
  for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9 * step; v += step) out.push(Math.abs(v) < 1e-9 * step ? 0 : v);
  return out;
}
export interface MapExtent { rMax: number; zBot: number; zTop: number }
/**
 * Axes in mm around a result map (director after GB3 r1). `box` holds the canvas and gets tick labels
 * placed in % of its size, r below (−R to R) and z on the left (zBot at the bottom, zTop at the top).
 */
export function mapAxes(box: HTMLElement, e: MapExtent, fmt: (v: number) => string): void {
  box.querySelectorAll('.wr-tk, .wr-axt').forEach((x) => x.remove());
  const add = (cls: string, text: string, style: string) => {
    const s = document.createElement('span'); s.className = cls; s.textContent = text; s.setAttribute('style', style); s.setAttribute('aria-hidden', 'true'); box.appendChild(s);
  };
  for (const r of niceTicks(-e.rMax, e.rMax, 4)) add('wr-tk wr-tk--x', fmt(r), `left:${(100 * (r + e.rMax)) / (2 * e.rMax)}%`);
  for (const z of niceTicks(e.zBot, e.zTop, 6)) add('wr-tk wr-tk--y', fmt(z), `top:${(100 * (e.zTop - z)) / (e.zTop - e.zBot)}%`);
  add('wr-axt wr-axt--x', 'r (mm)', '');
  add('wr-axt wr-axt--y', 'z (mm)', '');
}

/** Horizontal colour bar. */
export function drawBar(cv: HTMLCanvasElement, name: MapName, w = 256, h = 12): void {
  cv.width = w; cv.height = h;
  const g = cv.getContext('2d')!;
  const t = table(name, theme());
  for (let x = 0; x < w; x++) { g.fillStyle = t[Math.round((x / (w - 1)) * 255)]; g.fillRect(x, 0, 1, h); }
}

/**
 * PNG of a map with its colour bar, units and labels, in the current theme (the file name says it).
 * Each data pixel becomes `scale` screen pixels.
 */
export async function mapPng(src: HTMLCanvasElement, name: MapName, labels: { title: string; unit: string; vmin: string; vmax: string; axes: string }, scale = 4, ext?: MapExtent & { aspect: number; fmt: (v: number) => string }): Promise<Blob> {
  // true aspect ratio when the extent is known (the canvas holds one pixel per cell, cells are not square)
  const W = src.width * scale, H = ext ? Math.round(W / ext.aspect) : src.height * scale;
  const pad = 16, barH = 14, textH = 18, left = ext ? 56 : 0, tickH = ext ? 34 : 0;
  const out = document.createElement('canvas');
  out.width = Math.max(W + left, 320) + 2 * pad; out.height = H + barH + 3 * textH + 3 * pad + tickH;
  const g = out.getContext('2d')!;
  g.fillStyle = paperHex(); g.fillRect(0, 0, out.width, out.height);
  g.imageSmoothingEnabled = false;
  const x0 = pad + left, y0 = pad + textH;
  g.drawImage(src, x0, y0, W, H);
  const ink = cssVar('--ink') || '#111111';
  g.fillStyle = ink; g.strokeStyle = ink; g.font = '14px "CMU Serif", serif';
  g.fillText(labels.title, pad, pad + 12);
  if (ext) {
    g.font = '12px "CMU Serif", serif'; g.lineWidth = 1;
    g.strokeRect(x0 - 0.5, y0 - 0.5, W + 1, H + 1);
    for (const r of niceTicks(-ext.rMax, ext.rMax, 4)) {
      const x = x0 + (W * (r + ext.rMax)) / (2 * ext.rMax), t = ext.fmt(r);
      g.beginPath(); g.moveTo(x, y0 + H); g.lineTo(x, y0 + H + 4); g.stroke();
      g.fillText(t, x - g.measureText(t).width / 2, y0 + H + 16);
    }
    for (const z of niceTicks(ext.zBot, ext.zTop, 6)) {
      const y = y0 + (H * (ext.zTop - z)) / (ext.zTop - ext.zBot), t = ext.fmt(z);
      g.beginPath(); g.moveTo(x0 - 4, y); g.lineTo(x0, y); g.stroke();
      g.fillText(t, x0 - 6 - g.measureText(t).width, y + 4);
    }
    g.fillText('r (mm)', x0 + W / 2 - 18, y0 + H + 30);
    g.save(); g.translate(pad + 8, y0 + H / 2 + 18); g.rotate(-Math.PI / 2); g.fillText('z (mm)', 0, 0); g.restore();
    g.font = '14px "CMU Serif", serif';
  }
  g.fillText(labels.axes, pad, pad + textH + H + 14 + tickH);
  const bar = document.createElement('canvas');
  drawBar(bar, name, 256, barH);
  const by = pad * 2 + textH * 2 + H + tickH;
  g.drawImage(bar, pad, by, Math.min(256, out.width - 2 * pad), barH);
  g.fillText(labels.vmin, pad, by + barH + 14);
  const r = `${labels.vmax} ${labels.unit}`;
  g.fillText(r, pad + Math.min(256, out.width - 2 * pad) - g.measureText(r).width, by + barH + 14);
  return await new Promise<Blob>((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error('png'))), 'image/png'));
}
