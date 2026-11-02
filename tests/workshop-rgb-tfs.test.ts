// solve_T_fs, signal_mask and f_s against radiometric_calibration.py:
// run with `node --experimental-strip-types tests/workshop-rgb-tfs.test.ts`
import { readFileSync } from 'node:fs';
import { solveTfs, refineTfs, tfsFlags, maskTfs, TFS_FS_NEGATIVE, TFS_T_EDGE, signalMask, kernelTableTrapz, fsFromChannel, fsThreeChannels, fsChannels, ratioTemperature, makeBand, ratioTable, buildPsi, CH } from '../src/lib/workshop-rgb/radiometry.ts';
import { DEFAULT_PRESET } from '../src/lib/workshop-rgb/presets.ts';
import { runTfs, fsMode, M_USED } from '../src/components/workshop-rgb/engine.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };
const fx = (p: string) => new URL(`./fixtures/workshop-rgb/${p}`, import.meta.url);
const ref = JSON.parse(readFileSync(fx('tfs_ref.json'), 'utf8'));

const Tg: number[] = ref.T_grid;
const G = { R: ref.G[0], G: ref.G[1], B: ref.G[2] };
const j = Float64Array.from(ref.j.flat());
const n = j.length / 3;

// 1. kernel table on the same grid
{
  const imx = DEFAULT_PRESET.response;
  const K = kernelTableTrapz(imx.lamNm, buildPsi(imx.lamNm, imx.eta), Tg);
  let m = 0;
  for (const c of CH) for (let t = 0; t < Tg.length; t++) m = Math.max(m, Math.abs(K[c][t] / G[c][t] - 1));
  check('kernel_table on 1000 to 2600 K every 2 K', m < 1e-9, `(max rel ${m.toExponential(1)})`);
}
// 2. solve_T_fs, T exact on the grid, f_s 1e-9, χ² relative to Σ w j²
const cmp = (T: Float64Array, fs: Float64Array, chi2: Float64Array, rT: (number | null)[], rf: (number | null)[], rc: (number | null)[], w: number[]) => {
  let okT = true, mf = 0, mc = 0;
  for (let p = 0; p < n; p++) {
    if (rT[p] === null) { if (!Number.isNaN(T[p])) okT = false; continue; }
    if (T[p] !== rT[p]) okT = false;
    mf = Math.max(mf, Math.abs(fs[p] / rf[p]! - 1));
    const A = w[0] * j[3 * p] ** 2 + w[1] * j[3 * p + 1] ** 2 + w[2] * j[3 * p + 2] ** 2;
    mc = Math.max(mc, Math.abs(chi2[p] - rc[p]!) / A);
  }
  return { okT, mf, mc };
};
{
  const r = solveTfs(j, G, Tg);
  const c = cmp(r.T, r.fs, r.chi2, ref.T, ref.fs, ref.chi2, [1, 1, 1]);
  check('solve_T_fs, T exact on the grid', c.okT);
  check('solve_T_fs, f_s and χ²', c.mf < 1e-9 && c.mc < 1e-9, `(f_s ${c.mf.toExponential(1)}, χ² ${c.mc.toExponential(1)})`);
  const mask = signalMask(j);
  check('signal_mask (Σ_j j_j > 0,02 max)', Array.from(mask).every((v, i) => !!v === ref.mask[i]), `(${mask.reduce((a, b) => a + b, 0)} of ${n})`);
  const w: [number, number, number] = [1, 1, 0];
  const r2 = solveTfs(j, G, Tg, w, mask);
  const c2 = cmp(r2.T, r2.fs, r2.chi2, ref.T_w, ref.fs_w, ref.chi2_w, w);
  check('solve_T_fs with the blue weighted 0 and the mask (chemiluminescence zone)', c2.okT && c2.mf < 1e-9 && c2.mc < 1e-9);
}
// 3. f_s = j_c / G_c(T) and the three channel least squares, recovering a synthetic soot field
{
  const imx = DEFAULT_PRESET.response;
  const band = makeBand(imx.lamNm, buildPsi(imx.lamNm, imx.eta));
  const rt = ratioTable(band);
  const Ttrue = [1500, 1650.5, 1800, 1999.25, 2300], fsTrue = [1e-6, 2.5e-6, 4e-6, 3e-6, 0.5e-6];
  const jj = new Float64Array(15);
  const interpG = (c: 'R' | 'G' | 'B', T: number) => { const i = Math.floor(T - 1000), g = T - 1000 - i; return rt.G[c][i] * (1 - g) + rt.G[c][i + 1] * g; };
  Ttrue.forEach((T, p) => { for (let c = 0; c < 3; c++) jj[3 * p + c] = fsTrue[p] * interpG(CH[c], T); });
  const jR = Float64Array.from({ length: 5 }, (_, p) => jj[3 * p]), jG = Float64Array.from({ length: 5 }, (_, p) => jj[3 * p + 1]);
  const T = ratioTemperature(jR, jG, rt.T, rt.RG);
  check('T from R/G of a synthetic soot field within 0,5 K (linear table on 1 K)', T.every((v, p) => Math.abs(v - Ttrue[p]) < 0.5), `(${Array.from(T, (v) => v.toFixed(2)).join(', ')})`);
  const fR = fsFromChannel(jR, Float64Array.from(Ttrue), rt.T, rt.G.R);
  check('f_s = j_R / G_R(T) at the true T', fR.every((v, p) => Math.abs(v / fsTrue[p] - 1) < 1e-12));
  const f3 = fsThreeChannels(jj, Float64Array.from(Ttrue), rt.T, rt.G);
  check('f_s from the three channels at the true T', f3.every((v, p) => Math.abs(v / fsTrue[p] - 1) < 1e-12));
  const nan = fsFromChannel(Float64Array.of(1), Float64Array.of(NaN), rt.T, rt.G.R);
  check('f_s is NaN where T is NaN (masked, never zero)', Number.isNaN(nan[0]));
}
// 4. edge cases of the three channel fit (GB2r1-numerico-10) and the off-grid refinement (GB2r1-numerico-11)
{
  const imx = DEFAULT_PRESET.response;
  const psi = buildPsi(imx.lamNm, imx.eta);
  const g = (T: number) => { const k = kernelTableTrapz(imx.lamNm, psi, [T]); return [k.R[0], k.G[0], k.B[0]]; };
  const T0 = 1801.3, fs0 = 3e-6;
  const j0 = g(T0).map((v) => v * fs0);
  const jj = Float64Array.from([...j0, ...j0.map((v) => -v), 0, 0, 0]);
  const r = solveTfs(jj, G, Tg);
  const fl = tfsFlags(r, Tg);
  check('negative j gives fs_negative, zero j gives T_at_grid_edge and fs_negative, a good pixel no flag',
    fl[0] === 0 && (fl[1] & TFS_FS_NEGATIVE) !== 0 && (fl[2] & TFS_T_EDGE) !== 0, `(flags ${Array.from(fl).join(', ')}, grid T ${Array.from(r.T).join(', ')} K)`);
  const m = maskTfs(r, Tg);
  check('masked pixels are NaN in T, f_s and χ², the good one is kept', Number.isNaN(m.res.T[1]) && Number.isNaN(m.res.fs[2]) && m.res.T[0] === r.T[0]);
  const e0 = Math.abs(r.fs[0] / fs0 - 1);
  const rr = refineTfs(jj, G, Tg, r);
  const eT = Math.abs(rr.T[0] - T0), eF = Math.abs(rr.fs[0] / fs0 - 1);
  check('off-grid refinement brings T within 0,05 K and f_s within 0,05 % (grid of 2 K)', eT < 0.05 && eF < 5e-4,
    `(grid ${r.T[0]} K and ${(100 * e0).toFixed(2)} %, refined ${rr.T[0].toFixed(3)} K and ${(100 * eF).toFixed(3)} %)`);
  const same = refineTfs(j, G, Tg, solveTfs(j, G, Tg));
  check('refinement keeps the fixture pixels within one grid step of solve_T_fs', Array.from(same.T).every((v, p) => Number.isNaN(v) ? ref.T[p] === null : Math.abs(v - ref.T[p]) <= 2));
}

// 5. f_s from R and G by default (director after G4 r2), blue as the advanced option
{
  // R and G at the T of solve_T_fs with blue weighted 0 is the f_s of that fit (fs_3ch with chans=(0, 1))
  const Tw = Float64Array.from(ref.T_w as (number | null)[], (v) => (v === null ? NaN : v));
  const fw = fsChannels(j, Tw, Tg, G, [1, 1, 0]);
  let mw = 0, nw = 0;
  for (let p = 0; p < n; p++) if (ref.T_w[p] !== null) { mw = Math.max(mw, Math.abs(fw[p] / ref.fs_w[p] - 1)); nw++; } else if (!Number.isNaN(fw[p])) mw = Infinity;
  check('f_s from R and G at the T of the fit with blue weighted 0 equals its f_s, NaN where T is NaN', mw < 1e-9, `(${nw} pixels, ${mw.toExponential(1)})`);
  const T1 = Float64Array.of(1800, 1800);
  const j1 = Float64Array.of(1, 2, 3, 1, 2, 3);
  const all = fsChannels(j1, T1, Tg, G, [1, 1, 1]), noB = fsChannels(j1, T1, Tg, G, [1, 1, 1], Uint8Array.of(0, 1));
  check('fsThreeChannels is fsChannels with weights 1, and the noBlue mask drops blue per pixel',
    all[0] === fsThreeChannels(j1, T1, Tg, G)[0] && noB[0] === all[0] && noB[1] === fsChannels(j1, T1, Tg, G, [1, 1, 0])[1] && noB[1] !== all[1]);
  check('fsMode keeps the listed modes and sends anything else, an old run.json without it included, to R and G',
    fsMode('RG') === 'RG' && fsMode('RGB') === 'RGB' && fsMode('R') === 'R' && fsMode('B') === 'B' && fsMode(undefined) === 'RG' && fsMode('x') === 'RG');

  // runTfs on a soot field whose blue is 30 % off (an oven K_B that does not hold in the flame)
  const imx = DEFAULT_PRESET.response;
  const band = makeBand(imx.lamNm, buildPsi(imx.lamNm, imx.eta));
  const rt = ratioTable(band);
  const nz = 3, nr = 4;
  const Tt = (k: number, i: number) => 1600 + 100 * k + 40 * i, Ft = (k: number, i: number) => (1 + k + 0.5 * i) * 1e-6;
  const interpG = (c: 'R' | 'G' | 'B', T: number) => { const a = Math.floor(T - 1000), g = T - 1000 - a; return rt.G[c][a] * (1 - g) + rt.G[c][a + 1] * g; };
  const J = CH.map((c) => Array.from({ length: nz }, (_, k) => Float64Array.from({ length: nr }, (_, i) => Ft(k, i) * interpG(c, Tt(k, i)) * (c === 'B' ? 1.3 : 1))));
  const prep = { nz, nr, satRows: [] as number[], zMm: Float64Array.from({ length: nz }, (_, k) => 10 + k), rMm: Float64Array.from({ length: nr }, (_, i) => 0.1 * i) };
  const opts = (fsChannel: string, chemBG = false) => ({ path: 'ratio' as const, pair: 'RG' as const, emLaw: undefined as never, sigFrac: 0, zCutMm: null, chemThr: 0.1, fsChannel: fsMode(fsChannel), refine: false, weights: [1, 1, 1] as [number, number, number], islandFrac: 0, chemBG });
  const err = (o: ReturnType<typeof runTfs>) => { let m = 0; for (let k = 0; k < nz; k++) for (let i = 0; i < nr; i++) m = Math.max(m, Math.abs(o.fs[k * nr + i] / Ft(k, i) - 1)); return m; };
  const rg = runTfs(J, prep as never, band, opts('RG'), undefined, rt), r3 = runTfs(J, prep as never, band, opts('RGB'), undefined, rt);
  const rR = runTfs(J, prep as never, band, opts('R'), undefined, rt), rB = runTfs(J, prep as never, band, opts('B'), undefined, rt), r3c = runTfs(J, prep as never, band, opts('RGB', true), undefined, rt);
  check('runTfs, f_s from R and G is right with a wrong blue, and every pixel is used', err(rg) < 2e-3 && Array.from(rg.mask).every((m) => (m & M_USED) !== 0), `(${(100 * err(rg)).toFixed(3)} %)`);
  check('runTfs, R and G gives the f_s of red alone with T from R/G', Array.from(rg.fs).every((v, q) => Math.abs(v / rR.fs[q] - 1) < 1e-9));
  check('runTfs, blue alone carries its 30 % error, the advanced three channels a part of it (G_B is small), and drop blue where B/G marks chemiluminescence', Math.abs(err(rB) - 0.3) < 2e-3 && err(r3) > 1e-3 && Array.from(r3c.fs).every((v, q) => r3c.chem[q] ? Math.abs(v / rg.fs[q] - 1) < 1e-12 : true) && r3c.chem.some((x) => x), `(${(100 * err(rB)).toFixed(1)} % blue alone, ${(100 * err(r3)).toFixed(2)} % with three channels)`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
