/**
 * Lightweight plotting for scientific islands (no chart library).
 *  - colormaps: perceptually uniform viridis / cividis approximations, diverging RdBu
 *  - drawField: render a 2D array to a canvas (optionally mirrored about the axis)
 *  - LinePlot: responsive SVG axes with line, point and band series
 * Colours of axes and text come from CSS custom properties, so plots follow dark mode.
 */

export type RGB = [number, number, number];
export type Cmap = (t: number) => RGB;

const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);

// Polynomial fits (Matt Zucker, CC0) of matplotlib's viridis
export const viridis: Cmap = (t) => {
  t = clamp01(t);
  const c0 = [0.2777273272234177, 0.005407344544966578, 0.3340998053353061];
  const c1 = [0.1050930431085774, 1.404613529898575, 1.384590162594685];
  const c2 = [-0.3308618287255563, 0.214847559468213, 0.09509516302823659];
  const c3 = [-4.634230498983486, -5.799100973351585, -19.33244095627987];
  const c4 = [6.228269936347081, 14.17993336680509, 56.69055260068105];
  const c5 = [4.776384997670288, -13.74514537774601, -65.35303263337234];
  const c6 = [-5.435455855934631, 4.645852612178535, 26.3124352495832];
  const out = [0, 0, 0].map((_, i) => c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i]))))));
  return out.map((v) => Math.round(255 * clamp01(v))) as RGB;
};

/** Diverging blue-white-red for signed errors. */
export const rdbu: Cmap = (t) => {
  t = clamp01(t);
  const a: RGB = [33, 102, 172], m: RGB = [247, 247, 247], b: RGB = [178, 24, 43];
  const lerp = (p: RGB, q: RGB, s: number) => p.map((v, i) => Math.round(v + (q[i] - v) * s)) as RGB;
  return t < 0.5 ? lerp(a, m, t * 2) : lerp(m, b, (t - 0.5) * 2);
};

/** Sequential for loss landscapes: light paper to accent ink. */
export const landscape: Cmap = (t) => {
  t = clamp01(t);
  const stops: RGB[] = [[150, 178, 214], [190, 206, 228], [220, 229, 239], [239, 242, 244], [251, 250, 247]];
  const x = t * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const s = x - i;
  return stops[i].map((v, k) => Math.round(v + (stops[i + 1][k] - v) * s)) as RGB;
};

export function cssVar(name: string, el: Element = document.documentElement): string {
  return getComputedStyle(el).getPropertyValue(name).trim();
}

export interface FieldOptions {
  nx: number;
  ny: number;
  vmin?: number;
  vmax?: number;
  cmap?: Cmap;
  /** mirror columns about x = 0 (axisymmetric fields stored for r >= 0) */
  mirror?: boolean;
  /** flip vertically so row 0 is at the bottom */
  originLower?: boolean;
  nanColor?: RGB | null;
}

/** Render a row-major field (ny rows × nx cols) into a canvas at native resolution. */
export function drawField(canvas: HTMLCanvasElement, data: ArrayLike<number>, o: FieldOptions) {
  const { nx, ny } = o;
  const W = o.mirror ? 2 * nx : nx;
  canvas.width = W;
  canvas.height = ny;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(W, ny);
  let vmin = o.vmin, vmax = o.vmax;
  if (vmin === undefined || vmax === undefined) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < data.length; i++) { const v = data[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    vmin ??= lo; vmax ??= hi;
  }
  const cmap = o.cmap ?? viridis;
  const span = vmax - vmin || 1;
  for (let j = 0; j < ny; j++) {
    const row = o.originLower === false ? j : ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const v = data[j * nx + i];
      let rgb: RGB | null;
      if (!Number.isFinite(v)) rgb = o.nanColor ?? null;
      else rgb = cmap((v - vmin) / span);
      const cols = o.mirror ? [nx + i, nx - 1 - i] : [i];
      for (const c of cols) {
        const p = (row * W + c) * 4;
        if (rgb) { img.data[p] = rgb[0]; img.data[p + 1] = rgb[1]; img.data[p + 2] = rgb[2]; img.data[p + 3] = 255; }
        else img.data[p + 3] = 0;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Load a uint16-quantised field (little-endian; 65535 = masked), exported by scripts/export_*.py. */
export async function loadU16Field(url: string, min: number, max: number): Promise<Float32Array> {
  const buf = new Uint16Array(await (await fetch(url)).arrayBuffer());
  const out = new Float32Array(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = buf[i] === 65535 ? NaN : min + ((max - min) * buf[i]) / 65534;
  return out;
}

/** Load a uint8-quantised field exported by scripts/export_*.py. */
export async function loadU8Field(url: string, min: number, max: number): Promise<Float32Array> {
  const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const out = new Float32Array(buf.length);
  for (let i = 0; i < buf.length; i++) out[i] = buf[i] === 255 ? NaN : min + ((max - min) * buf[i]) / 254;
  return out;
}

/* ------------------------------------------------------------------ LinePlot */

const SVGNS = 'http://www.w3.org/2000/svg';
function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  parent?.appendChild(e);
  return e;
}

export function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export function fmt(v: number): string {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(0).replace('e+', 'e');
  return String(+v.toPrecision(3));
}

export interface Series {
  kind: 'line' | 'points' | 'band' | 'step';
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  y2?: ArrayLike<number>; // for band: upper curve
  cls: string; // CSS class: s-data, s-physics, s-learned, s-truth, s-warn, s-muted
  label?: string;
  dash?: boolean;
}

export interface LinePlotOptions {
  width?: number;
  height?: number;
  xlabel?: string;
  ylabel?: string;
  xlim?: [number, number];
  ylim?: [number, number];
  logy?: boolean;
  title?: string;
  legend?: boolean;
}

export class LinePlot {
  svg: SVGSVGElement;
  private g: SVGGElement;
  private o: Required<Pick<LinePlotOptions, 'width' | 'height'>> & LinePlotOptions;
  private m = { l: 52, r: 14, t: 16, b: 40 };
  private aspect: number;
  private last: [Series[], Partial<Pick<LinePlotOptions, 'xlim' | 'ylim'>>] | null = null;
  private container: HTMLElement;

  constructor(container: HTMLElement, o: LinePlotOptions = {}) {
    this.o = { width: 560, height: 300, ...o };
    this.aspect = this.o.height / this.o.width;
    this.container = container;
    this.svg = el('svg', { viewBox: `0 0 ${this.o.width} ${this.o.height}`, class: 'lineplot', role: 'img' });
    if (o.title) el('title', {}, this.svg).textContent = o.title;
    this.g = el('g', {}, this.svg);
    container.appendChild(this.svg);
    // Match the viewBox to the rendered width so text stays readable on small screens.
    let lastW = 0;
    new ResizeObserver(() => {
      const w = Math.round(container.clientWidth);
      if (w > 0 && Math.abs(w - lastW) > 8 && this.last) { lastW = w; this.render(...this.last); }
    }).observe(container);
  }

  /** Change axis labels (applied on the next render). */
  setLabels(xlabel?: string, ylabel?: string) {
    if (xlabel !== undefined) this.o.xlabel = xlabel;
    if (ylabel !== undefined) this.o.ylabel = ylabel;
  }

  render(series: Series[], lim: Partial<Pick<LinePlotOptions, 'xlim' | 'ylim'>> = {}) {
    this.last = [series, lim];
    const cw = this.container.clientWidth;
    if (cw > 0) {
      this.o.width = Math.max(300, Math.min(cw, 900));
      // shorter plots on narrow screens, but not flatter than the requested aspect
      this.o.height = Math.round(Math.max(this.o.width * this.aspect, Math.min(220, this.o.width * 0.75)));
      this.svg.setAttribute('viewBox', `0 0 ${this.o.width} ${this.o.height}`);
    }
    const { width: W, height: H } = this.o;
    const m = this.m;
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    this.g.replaceChildren();
    const logy = !!this.o.logy;
    const ty = (v: number) => (logy ? Math.log10(Math.max(v, 1e-300)) : v);

    let [x0, x1] = lim.xlim ?? this.o.xlim ?? [Infinity, -Infinity];
    let [y0, y1] = lim.ylim ?? this.o.ylim ?? [Infinity, -Infinity];
    const autoX = !(lim.xlim ?? this.o.xlim), autoY = !(lim.ylim ?? this.o.ylim);
    for (const s of series) {
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i];
        const vals = [s.y[i], s.y2?.[i]].filter((v): v is number => v !== undefined && Number.isFinite(v));
        if (autoX && Number.isFinite(xv)) { x0 = Math.min(x0, xv); x1 = Math.max(x1, xv); }
        if (autoY) for (const v of vals) { const t = ty(v); if (Number.isFinite(t)) { y0 = Math.min(y0, t); y1 = Math.max(y1, t); } }
      }
    }
    if (!autoY && logy) { y0 = ty(y0); y1 = ty(y1); }
    if (!(x1 > x0)) { x0 -= 1; x1 += 1; }
    if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
    if (autoY && !logy) { const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad; }

    const X = (v: number) => m.l + ((v - x0) / (x1 - x0)) * pw;
    const Y = (v: number) => m.t + ph - ((ty(v) - y0) / (y1 - y0)) * ph;
    const Yraw = (t: number) => m.t + ph - ((t - y0) / (y1 - y0)) * ph;

    // clip
    const id = `clip${Math.random().toString(36).slice(2, 8)}`;
    const defs = el('defs', {}, this.g);
    const cp = el('clipPath', { id }, defs);
    el('rect', { x: m.l, y: m.t, width: pw, height: ph }, cp);

    // axes & ticks
    const axes = el('g', { class: 'axes' }, this.g);
    for (const v of niceTicks(x0, x1, 5)) {
      el('line', { x1: X(v), x2: X(v), y1: m.t, y2: m.t + ph, class: 'grid' }, axes);
      const t = el('text', { x: X(v), y: m.t + ph + 16, 'text-anchor': 'middle', class: 'tick' }, axes);
      t.textContent = fmt(v);
    }
    const yt = logy
      ? Array.from({ length: Math.floor(y1) - Math.ceil(y0) + 1 }, (_, k) => Math.ceil(y0) + k)
      : niceTicks(y0, y1, 5);
    const ytStep = logy && yt.length > 6 ? Math.ceil(yt.length / 6) : 1;
    yt.forEach((v, k) => {
      if (k % ytStep) return;
      el('line', { x1: m.l, x2: m.l + pw, y1: Yraw(v), y2: Yraw(v), class: 'grid' }, axes);
      const t = el('text', { x: m.l - 6, y: Yraw(v) + 4, 'text-anchor': 'end', class: 'tick' }, axes);
      t.textContent = logy ? `1e${v}` : fmt(v);
    });
    el('rect', { x: m.l, y: m.t, width: pw, height: ph, class: 'plotframe' }, axes);
    if (this.o.xlabel) { const t = el('text', { x: m.l + pw / 2, y: H - 6, 'text-anchor': 'middle', class: 'axlabel' }, axes); t.textContent = this.o.xlabel; }
    if (this.o.ylabel) { const t = el('text', { x: 12, y: m.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${m.t + ph / 2})`, class: 'axlabel' }, axes); t.textContent = this.o.ylabel; }

    const plot = el('g', { 'clip-path': `url(#${id})` }, this.g);
    for (const s of series) {
      const n = s.x.length;
      if (s.kind === 'band' && s.y2) {
        let d = '';
        for (let i = 0; i < n; i++) d += `${i ? 'L' : 'M'}${X(s.x[i]).toFixed(1)},${Y(s.y2[i]).toFixed(1)}`;
        for (let i = n - 1; i >= 0; i--) d += `L${X(s.x[i]).toFixed(1)},${Y(s.y[i]).toFixed(1)}`;
        el('path', { d: d + 'Z', class: `band ${s.cls}` }, plot);
      } else if (s.kind === 'line' || s.kind === 'step') {
        let d = '';
        let pen = false;
        for (let i = 0; i < n; i++) {
          const yv = s.y[i];
          if (!Number.isFinite(yv)) { pen = false; continue; }
          const px = X(s.x[i]).toFixed(1), py = Y(yv).toFixed(1);
          if (s.kind === 'step' && pen && i > 0) d += `H${px}`;
          d += `${pen ? 'L' : 'M'}${px},${py}`;
          pen = true;
        }
        el('path', { d, class: `line ${s.cls}${s.dash ? ' dash' : ''}` }, plot);
      } else {
        for (let i = 0; i < n; i++) {
          if (!Number.isFinite(s.y[i])) continue;
          el('circle', { cx: X(s.x[i]), cy: Y(s.y[i]), r: 2.6, class: `pt ${s.cls}` }, plot);
        }
      }
    }

    if (this.o.legend !== false) {
      const labelled = series.filter((s) => s.label);
      const lg = el('g', { class: 'legend' }, this.g);
      let yy = m.t + 12;
      for (const s of labelled) {
        const x = m.l + pw - Math.min(150, pw * 0.5);
        if (s.kind === 'points') el('circle', { cx: x + 9, cy: yy - 4, r: 3, class: `pt ${s.cls}` }, lg);
        else if (s.kind === 'band') el('rect', { x, y: yy - 9, width: 18, height: 9, class: `band ${s.cls}` }, lg);
        else el('line', { x1: x, x2: x + 18, y1: yy - 4, y2: yy - 4, class: `line ${s.cls}${s.dash ? ' dash' : ''}` }, lg);
        const t = el('text', { x: x + 24, y: yy, class: 'legtext' }, lg);
        t.textContent = s.label!;
        yy += 16;
      }
    }
    return { X, Y };
  }
}
