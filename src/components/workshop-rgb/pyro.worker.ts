/**
 * Coordinator Worker of the RGB pyrometry tool (WEB_SPEC §7), the first Worker of the site. It reads one
 * file at a time from the Blobs the page posts, keeps the working frames in a bounded cache (re-reading a
 * file when it was evicted), and runs every heavy step of stages 1 to 6. Typed with a local scope
 * interface, because `/// <reference lib="webworker" />` next to lib.dom fails with TS6200 (WEB_SPEC §7.1).
 * Created at page load with new Worker(new URL('./pyro.worker.ts', import.meta.url), { type: 'module' }),
 * so it is fetched before the network can fail (§11.5).
 */
import type { ToWorker, FromWorker, PsiMsg } from './protocol.ts';
import {
  loadFrame, thumbnail, previewPlanes, seriesHot, ovenFrameOf, analyseSeries, plateauAt, runOvenFit, frameSignal, fitDarkLine, darkLevel,
  buildVignette, runSphere, frameAsym, prepareFlame, applyHot, runAbel, runLcurve, runTfs, runBand, bandFor, rulerShift, tailGrid, readTailSignals,
  type Frame, type FrameInfo, type LoadOptions, type Prepared, type AbelOut, type AbelRun, type EncOverride,
} from './engine.ts';
import { sha256Hex } from './sha256.ts';
import { TiffError } from '../../lib/workshop-rgb/tiff.ts';
import { replaceHot } from '../../lib/workshop-rgb/frames.ts';
import type { Aperture } from '../../lib/workshop-rgb/oven.ts';
import type { Rgb } from '../../lib/workshop-rgb/radiometry.ts';

interface WorkerScope { postMessage(m: FromWorker, transfer?: Transferable[]): void; onmessage: ((e: MessageEvent<ToWorker>) => void) | null }
const ctx = self as unknown as WorkerScope;

const CACHE_BYTES = 360e6;
const items = new Map<string, { path: string; blob: Blob; info?: FrameInfo; enc?: EncOverride | null }>();
const cache = new Map<string, Frame>();
let cached = 0;
let opts: LoadOptions = { group: true, bayer: 'RGGB', centre: false };

const bytesOf = (f: Frame) => f.dn.byteLength + f.sat.byteLength;
function put(id: string, f: Frame) {
  if (cache.has(id)) return;
  cache.set(id, f); cached += bytesOf(f);
  for (const [k, v] of cache) { if (cached <= CACHE_BYTES) break; if (k === id) continue; cache.delete(k); cached -= bytesOf(v); }
}
async function frameOf(id: string): Promise<{ frame: Frame; info: FrameInfo }> {
  const it = items.get(id);
  if (!it) throw new Error('missing_item');
  const hit = cache.get(id);
  if (hit && it.info) { cache.delete(id); cache.set(id, hit); return { frame: hit, info: it.info }; }
  const { info, frame } = loadFrame(await it.blob.arrayBuffer(), opts, it.enc);
  it.info = info;
  put(id, frame);
  return { frame, info };
}

function meanOf(frames: Frame[]): Frame {
  const f0 = frames[0];
  const dn = new Float32Array(f0.dn.length), sat = new Uint8Array(f0.sat.length);
  for (const f of frames) {
    if (f.dn.length !== dn.length) throw new TiffError('series_mismatch');
    for (let i = 0; i < dn.length; i++) { dn[i] += f.dn[i]; if (f.sat[i]) sat[i] = 1; }
  }
  for (let i = 0; i < dn.length; i++) dn[i] /= frames.length;
  return { w: f0.w, h: f0.h, dn, sat };
}

// ------------------------------------------------------------------ oven state
interface SeriesState { tC: number; ids: { id: string; texpUs: number }[]; hot: Uint8Array | null; aperture: Aperture; plateauId: string }
let series: SeriesState[] = [];
let darks: { id: string; texpUs: number }[] = [];
let ceil12 = 4095;

// ------------------------------------------------------------------ flame state
let prepared: Prepared | null = null;
let abel: AbelOut | null = null;

const psiOf = (p: PsiMsg) => ({ lamNm: Float64Array.from(p.lamNm), psi: { R: Float64Array.from(p.R), G: Float64Array.from(p.G), B: Float64Array.from(p.B) } as Rgb<Float64Array> });

async function handle(m: ToWorker, progress: (d: number, t: number) => void): Promise<unknown> {
  switch (m.type) {
    case 'opts': {
      opts = { group: m.group, bayer: m.bayer, centre: m.centre === true };
      cache.clear(); cached = 0;
      for (const it of items.values()) it.info = undefined;
      return null;
    }
    case 'add': { for (const it of m.items) items.set(it.id, { path: it.path, blob: it.blob }); return null; }
    case 'enc': {
      for (const id of m.ids) { const it = items.get(id); if (!it) continue; it.enc = m.enc; it.info = undefined; const f = cache.get(id); if (f) { cached -= bytesOf(f); cache.delete(id); } }
      return null;
    }
    case 'forget': { for (const id of m.ids) { items.delete(id); const f = cache.get(id); if (f) { cached -= bytesOf(f); cache.delete(id); } } return null; }
    case 'scan': {
      const it = items.get(m.id);
      if (!it) throw new Error('missing_item');
      const buf = await it.blob.arrayBuffer();
      const sha = await sha256Hex(buf);
      const { info, frame } = loadFrame(buf, opts, it.enc);
      it.info = info;
      if (m.keep) put(m.id, frame);
      return { info, sha: sha.hex, shaHow: sha.how, thumb: m.thumb ? thumbnail(frame, info.ceil12) : null };
    }
    case 'oven.analyse': {
      darks = m.darks;
      const darkFrames: Frame[] = [];
      for (const d of darks) darkFrames.push((await frameOf(d.id)).frame);
      const out = [];
      series = [];
      let k = 0;
      const total = m.series.reduce((a, s) => a + s.frames.length, 0);
      for (const s of m.series) {
        const sorted = [...s.frames].sort((a, b) => a.texpUs - b.texpUs);
        const frames: Frame[] = [];
        const infos: FrameInfo[] = [];
        for (const f of sorted) { const r = await frameOf(f.id); frames.push(r.frame); infos.push(r.info); progress(++k, total); }
        ceil12 = infos[0].ceil12;
        const sameSize = darkFrames.filter((d) => d.w === frames[0].w && d.h === frames[0].h);
        const hot = seriesHot(frames, sameSize, ceil12);
        const hotCount = [0, 0, 0];
        for (let i = 0; i < hot.length; i++) if (hot[i]) hotCount[i % 3]++;
        // plateau frame, the brightest with p99,9 under 90 % in every channel and nothing saturated
        let pick = -1;
        infos.forEach((inf, i) => {
          const ok = inf.satRaw.every((v) => v === 0) && inf.p999.every((v) => v < 0.9 * inf.ceil12);
          if (ok && (pick < 0 || inf.meanG > infos[pick].meanG)) pick = i;
        });
        if (pick < 0) infos.forEach((inf, i) => { if (inf.satRaw.every((v) => v === 0) && (pick < 0 || inf.meanG > infos[pick].meanG)) pick = i; });
        if (pick < 0) { pick = 0; infos.forEach((inf, i) => { if (inf.nSatWork < infos[pick].nSatWork) pick = i; }); }
        const of = ovenFrameOf(s.tC, sorted[pick].texpUs, frames[pick], hot);
        const { aperture, plateau } = analyseSeries(of);
        series.push({ tC: s.tC, ids: sorted, hot, aperture, plateauId: sorted[pick].id });
        // outside the aperture, zeros and stray light with t_exp
        const sigs = frames.map((f, i) => frameSignal(ovenFrameOf(s.tC, sorted[i].texpUs, f, hot), plateau, aperture));
        const n0 = frames[0].w * frames[0].h;
        let zeros = 0, outside = 0;
        const f0 = frames[0];
        for (let p = 0; p < n0; p++) {
          const y = Math.floor(p / f0.w), x = p % f0.w;
          if (Math.hypot(y - aperture.cy, x - aperture.cx) > 2.5 * aperture.r) { outside++; for (let c = 0; c < 3; c++) if (f0.dn[3 * p + c] === 0) zeros++; }
        }
        const narrow = outside < 200;
        const flags = new Uint8Array(f0.w * f0.h);
        const pf = frames[pick];
        for (let p = 0; p < flags.length; p++) {
          if (hot[3 * p] || hot[3 * p + 1] || hot[3 * p + 2]) flags[p] |= 4;
          const inside = Math.hypot(Math.floor(p / pf.w) - aperture.cy, (p % pf.w) - aperture.cx) < aperture.r;
          const sg = sigs[pick];
          if (inside) for (let c = 0; c < 3; c++) { const ch = (['R', 'G', 'B'] as const)[c]; const sd = Number.isFinite(sg.stripSigma[ch]) ? sg.stripSigma[ch] : 1; const D = Number.isFinite(sg.strip[ch]) ? sg.strip[ch] : 0; if (pf.dn[3 * p + c] - D < 5 * sd) flags[p] |= 2; }
        }
        const pv = previewPlanes(pf, ceil12, 480, flags);
        out.push({ tC: s.tC, aperture, plateau, texpUs: sorted.map((f) => f.texpUs), plateauId: sorted[pick].id, previewTexp: sorted[pick].texpUs, hotCount, zeroFrac: outside ? zeros / (3 * outside) : NaN, narrow, signals: sigs, preview: pv, w: f0.w, h: f0.h });
      }
      const darkStats = [];
      for (let i = 0; i < darks.length; i++) { const dl = darkLevel(darkFrames[i]); darkStats.push({ id: darks[i].id, texpUs: darks[i].texpUs, ...dl }); }
      return { series: out, darks: darkStats, ceil12 };
    }
    case 'oven.plateau': {
      const s = series.find((x) => x.tC === m.tC);
      if (!s) throw new Error('missing_series');
      const { frame } = await frameOf(s.plateauId);
      const tex = s.ids.find((x) => x.id === s.plateauId)!.texpUs;
      return plateauAt(ovenFrameOf(s.tC, tex, frame, s.hot), s.aperture, m.cy, m.cx, m.radius);
    }
    case 'oven.fit': {
      const signals = [];
      let k = 0;
      const nFrames = series.reduce((a, s) => a + s.ids.length, 0);
      const total = 2 * nFrames + series.length;
      const max12: Record<string, number[]> = {};
      for (const s of series) {
        const pl = m.plateaus[String(s.tC)];
        for (const f of s.ids) {
          const { frame, info } = await frameOf(f.id);
          max12[f.id] = info.max12;
          signals.push(frameSignal(ovenFrameOf(s.tC, f.texpUs, frame, s.hot), pl, s.aperture));
          progress(++k, total);
        }
      }
      // IR tail, read again at the sensor grid as calibration.py (ovenread.ts), one frame at a time and not cached
      const full = async (id: string) => { const it = items.get(id); if (!it) throw new Error('missing_item'); return loadFrame(await it.blob.arrayBuffer(), { ...opts, group: false }, it.enc); };
      const strips = new Map(signals.map((sg) => [`${sg.tC}|${sg.texpUs}`, sg.strip]));
      let tailIn: Awaited<ReturnType<typeof readTailSignals>> | null = null;
      try {
        tailIn = await readTailSignals(series.map((s) => ({ tC: s.tC, texpUs: s.ids.map((f) => f.texpUs), max12: s.ids.map((f) => max12[f.id]), load: (i: number) => full(s.ids[i].id) })), ceil12, strips, () => progress(++k, total));
      } catch { tailIn = null; }
      let darkLine: Rgb<{ a: number; b: number }> | undefined;
      if (m.darkMode === 'files' && darks.length) {
        const lv = [];
        for (const d of darks) lv.push({ texpUs: d.texpUs, level: darkLevel((await frameOf(d.id)).frame).level });
        darkLine = fitDarkLine(lv);
      }
      const p = psiOf(m.psi);
      const band = bandFor(p.lamNm, p.psi, m.emLaw);
      const pe = m.psiEnergy ? psiOf(m.psiEnergy) : null;
      const res = runOvenFit({ signals, band, bandEnergy: pe ? bandFor(pe.lamNm, pe.psi, m.emLaw) : null, eps: m.eps, sigmaT: m.sigmaT, ceil12, darkMode: m.darkMode, darkLine, perT: m.perT, calibrateAt: m.calibrateAt, tailGrid: tailGrid(p.lamNm, p.psi, m.emLaw), tailSignals: tailIn?.signals ?? null, tailRead: tailIn?.read ?? null });
      // hot pixels of the calibration, as working pixel index·3 + channel of the top series grid
      const top = series.find((s) => s.tC === res.fit.Ttop - 273.15) ?? series[series.length - 1];
      const hot: number[] = [];
      if (top?.hot) for (let i = 0; i < top.hot.length && hot.length < 200000; i++) if (top.hot[i]) hot.push(i);
      return { ...res, darkLine: darkLine ?? null, hot: Int32Array.from(hot) };
    }
    case 'flat.build': {
      const flats: Frame[] = [];
      for (const id of m.flats) flats.push((await frameOf(id)).frame);
      const mean = meanOf(flats);
      if (m.darks.length) {
        const dk: Frame[] = [];
        for (const id of m.darks) dk.push((await frameOf(id)).frame);
        const md = meanOf(dk);
        for (let i = 0; i < mean.dn.length; i++) mean.dn[i] -= md.dn[i];
      }
      const v = buildVignette(mean.dn, mean.w, mean.h, m.cx, m.cy);
      return { vignette: v, w: mean.w, h: mean.h };
    }
    case 'sphere.fit': {
      const fr: Frame[] = [];
      let c12 = 4095;
      for (const id of m.images) { const r = await frameOf(id); fr.push(r.frame); c12 = r.info.ceil12; }
      const mean = meanOf(fr);
      let dark: Frame | null = null;
      if (m.darks.length) { const dk: Frame[] = []; for (const id of m.darks) dk.push((await frameOf(id)).frame); dark = meanOf(dk); }
      const p = psiOf(m.psi);
      const res = runSphere({ mean, dark, ceil12: c12, tSphS: m.tSphS, spectrum: m.spectrum, valUnit: m.valUnit, lamUnit: m.lamUnit, nd: m.nd, fill: m.fill, lamNm: p.lamNm, psi: p.psi });
      return { ...res, w: mean.w, h: mean.h, preview: previewPlanes(mean, c12, 360) };
    }
    case 'flame.load': {
      const out = [];
      let k = 0;
      for (const id of m.ids) {
        const { frame, info } = await frameOf(id);
        const a = frameAsym(frame, m.quarter, m.mirror);
        out.push({ id, axis: a.axis, asym: a.asym, nSat: info.nSatWork, satRaw: info.satRaw, thumb: thumbnail(frame, info.ceil12, 120) });
        progress(++k, m.ids.length);
      }
      return out;
    }
    case 'flame.prepare': {
      let geom = { scale: 1, gx0: 0, gy0: 0 };
      let sum: Float32Array | null = null, sat: Uint8Array | null = null, w = 0, h = 0;
      let nHot = 0, k = 0;
      // pixels saturated also in the background are hot pixels of the sensor, not flame saturation (WEB_SPEC §5.4)
      let bgHot: Uint8Array | null = null;
      if (m.backgrounds.length) {
        const bf = (await frameOf(m.backgrounds[0])).frame;
        bgHot = new Uint8Array(bf.sat.length);
        for (let p = 0; p < bf.w * bf.h; p++) if (bf.sat[3 * p] || bf.sat[3 * p + 1] || bf.sat[3 * p + 2]) bgHot[3 * p] = bgHot[3 * p + 1] = bgHot[3 * p + 2] = 1;
        if (!bgHot.some((v) => v)) bgHot = null;
      }
      const clean = (f: Frame) => {
        if (!bgHot || bgHot.length !== f.sat.length) return 0;
        const n0 = replaceHot({ width: f.w, height: f.h, nc: 3, data: f.dn }, bgHot);
        for (let i = 0; i < bgHot.length; i++) if (bgHot[i]) f.sat[i] = 0;
        return n0;
      };
      for (const id of m.ids) {
        const { frame, info } = await frameOf(id);
        geom = { scale: info.scale, gx0: info.gx0, gy0: info.gy0 };
        const f: Frame = { w: frame.w, h: frame.h, dn: frame.dn.slice(), sat: frame.sat.slice() };
        nHot += applyHot(f, m.cal, { x0: m.geom.cropX0, y0: m.geom.cropY0 }, geom) + clean(f);
        if (!sum) { sum = new Float32Array(f.dn.length); sat = new Uint8Array(f.sat.length); w = f.w; h = f.h; }
        if (f.dn.length !== sum.length) throw new TiffError('series_mismatch');
        for (let i = 0; i < sum.length; i++) { sum[i] += f.dn[i]; if (f.sat[i]) sat![i] = 1; }
        progress(++k, m.ids.length + m.backgrounds.length);
      }
      if (!sum || !sat) throw new Error('no_frames');
      const n = m.ids.length;
      for (let i = 0; i < sum.length; i++) sum[i] /= n;
      let bgHow: 'files' | 'darkline' | 'border' = 'files';
      if (m.backgrounds.length) {
        const bg = new Float32Array(sum.length);
        for (const id of m.backgrounds) {
          const { frame: fr0 } = await frameOf(id);
          const frame: Frame = { w: fr0.w, h: fr0.h, dn: fr0.dn.slice(), sat: fr0.sat.slice() };
          clean(frame);
          if (frame.dn.length !== bg.length) throw new TiffError('series_mismatch');
          for (let i = 0; i < bg.length; i++) bg[i] += frame.dn[i] / m.backgrounds.length;
          progress(++k, m.ids.length + m.backgrounds.length);
        }
        for (let i = 0; i < sum.length; i++) sum[i] -= bg[i];
      } else if (m.darkLine) {
        bgHow = 'darkline';
        for (let i = 0; i < sum.length; i++) sum[i] -= m.darkLine[i % 3];
      } else {
        bgHow = 'border';
        const bw = Math.max(2, Math.floor(0.05 * w));
        for (let c = 0; c < 3; c++) {
          const v: number[] = [];
          for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x < bw || x >= w - bw) v.push(sum[(y * w + x) * 3 + c]);
          v.sort((a, b) => a - b);
          const med = v[Math.floor(v.length / 2)];
          for (let p = 0; p < w * h; p++) sum[3 * p + c] -= med;
        }
      }
      prepared = prepareFlame(sum, sat, w, h, m.texpUs, m.cal, geom, m.geom);
      abel = null;
      return { prepared, bgHow, nHot, frameGeom: geom, nFrames: n };
    }
    case 'ruler': {
      const { frame, info } = await frameOf(m.id);
      const it = items.get(m.id)!;
      let shift = null;
      try { shift = rulerShift(await it.blob.arrayBuffer(), opts.bayer, it.enc); } catch { shift = null; }
      return { preview: previewPlanes(frame, info.ceil12, 720), scale: info.scale, w: frame.w, h: frame.h, shift };
    }
    case 'abel.run': {
      if (!prepared) throw new Error('no_projections');
      abel = runAbel(prepared, m.run);
      return abel;
    }
    case 'abel.lcurve': {
      if (!prepared) throw new Error('no_projections');
      return runLcurve(prepared, m.rows, m.ref, m.sym);
    }
    case 'abel.set': {
      if (!prepared) throw new Error('no_projections');
      const P = runAbel(prepared, { ...m.run, method: 'onion', sym: 'average' }).Pused;
      const resid = m.reproj.map((ch, c) => ch.map((q, k) => q.map((v, i) => P[c][k][i] - v)));
      abel = { J: m.J, reproj: m.reproj, resid, Pused: P, overReg: 0, excluded: prepared.satRows, lambdaRel: NaN };
      return abel;
    }
    case 'tfs.run': {
      if (!prepared || !abel) throw new Error('no_emission');
      const p = psiOf(m.psi);
      const band = bandFor(p.lamNm, p.psi, m.opts.emLaw);
      return runTfs(abel.J, prepared, band, m.opts, abel.Pused);
    }
    case 'band.run': {
      if (!prepared) throw new Error('no_projections');
      const p = psiOf(m.psi);
      const band = bandFor(p.lamNm, p.psi, m.tfs.emLaw);
      return runBand(prepared, band, { n: m.n, seed: m.seed, sigmaT: m.sigmaT, Tcal: m.Tcal, eps: m.eps, run: m.run as AbelRun, tfs: m.tfs, absolute: true, sKrel: m.sKrel, scaleRel: m.scaleRel });
    }
  }
}

ctx.onmessage = (ev) => {
  const m = ev.data;
  const progress = (done: number, total: number) => ctx.postMessage({ req: m.req, kind: 'progress', done, total });
  handle(m, progress).then(
    (data) => ctx.postMessage({ req: m.req, kind: 'ok', data }),
    (e: unknown) => {
      const err = e as Error & { code?: string; detail?: Record<string, string | number> };
      const known = err instanceof TiffError || (typeof err?.code === 'string');
      const msg = String(err?.message ?? e);
      const code = known ? err.code! : /memory|allocation|Array buffer/i.test(msg) ? 'out_of_memory'
        : /^(oven_no_aperture|sphere_no_disc|no_frames|no_projections|no_emission|series_mismatch|lambda_out_of_range|kernel_nonpositive|sat_not_spd|tikhonov_not_spd|odd_width)$/.test(msg) ? msg : 'worker_failed';
      ctx.postMessage({ req: m.req, kind: 'error', code, detail: err?.detail ?? { message: msg } });
    },
  );
};
