/**
 * Legend drawn as an HTML key below a plot, so it never covers the data.
 * Swatches reuse the semantic series classes (s-data, s-physics, s-learned, …).
 */
export interface KeyItem { label: string; cls: string; kind?: 'line' | 'dash' | 'point' | 'band' }

export function renderKey(el: HTMLElement, items: KeyItem[]) {
  const NS = 'http://www.w3.org/2000/svg';
  el.replaceChildren();
  el.classList.add('plot-key');
  for (const it of items) {
    const span = document.createElement('span');
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('width', '18'); svg.setAttribute('height', '10'); svg.setAttribute('aria-hidden', 'true');
    const kind = it.kind ?? 'line';
    let shape: SVGElement;
    if (kind === 'point') { shape = document.createElementNS(NS, 'circle'); shape.setAttribute('cx', '9'); shape.setAttribute('cy', '5'); shape.setAttribute('r', '3'); shape.setAttribute('class', `pt ${it.cls}`); }
    else if (kind === 'band') { shape = document.createElementNS(NS, 'rect'); shape.setAttribute('x', '1'); shape.setAttribute('y', '1'); shape.setAttribute('width', '16'); shape.setAttribute('height', '8'); shape.setAttribute('class', `band ${it.cls}`); }
    else { shape = document.createElementNS(NS, 'line'); shape.setAttribute('x1', '1'); shape.setAttribute('x2', '17'); shape.setAttribute('y1', '5'); shape.setAttribute('y2', '5'); shape.setAttribute('class', `line ${it.cls}${kind === 'dash' ? ' dash' : ''}`); }
    svg.appendChild(shape);
    svg.classList.add('lineplot');
    span.append(svg, document.createTextNode(it.label));
    el.appendChild(span);
  }
}

/** Minimal shape of a plot series (compatible with LinePlot's Series). */
interface KeySeries { kind: string; cls: string; label?: string; dash?: boolean }

/** Key items from the labelled series of a plot (duplicates removed). */
export function itemsFromSeries(series: KeySeries[]): KeyItem[] {
  const seen = new Set<string>();
  const out: KeyItem[] = [];
  for (const s of series) {
    if (!s.label || seen.has(s.label)) continue;
    seen.add(s.label);
    out.push({ label: s.label, cls: s.cls, kind: s.kind === 'points' ? 'point' : s.kind === 'band' ? 'band' : s.dash ? 'dash' : 'line' });
  }
  return out;
}

/**
 * Wrap a plot so that every render also refreshes an HTML key placed right below
 * the plot container. The plot itself should be created with `legend: false`.
 */
export function keyed<S extends KeySeries, L>(plot: { render: (s: S[], lim?: L) => unknown }, container: Element) {
  const key = document.createElement('div');
  key.className = 'plot-key';
  container.after(key);
  return (series: S[], lim?: L) => { plot.render(series, lim); renderKey(key, itemsFromSeries(series)); };
}
