// Checks for the earthquake-location tab: run with `node tests/eng-civil.test.ts`
import { arrivalTimes, misfitMap, misfitAt, confidenceLevel, regionArea, contourSegments, hiddenQuake, type Pt } from '../src/lib/eng/quake.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

const spread: Pt[] = [{ x: 30, y: 40 }, { x: 170, y: 30 }, { x: 100, y: 170 }, { x: 40, y: 150 }, { x: 160, y: 140 }, { x: 100, y: 80 }];
const line: Pt[] = [20, 50, 80, 110, 140, 170].map((x) => ({ x, y: 100 }));

// 1. Noise-free data: the epicentre and origin time are recovered
{
  const src = { x: 123.4, y: 67.8 }, t0 = 3.2;
  const m = misfitMap(spread, arrivalTimes(spread, src, t0), 100);
  const err = Math.hypot(m.best.x - src.x, m.best.y - src.y);
  check('noise-free epicentre recovered', err < 0.1, `(error ${err.toFixed(3)} km, t0 ${m.t0.toFixed(3)} s)`);
  check('noise-free origin time recovered', Math.abs(m.t0 - t0) < 0.02);
}

// 2. Stations on a line cannot tell a source from its mirror image
{
  const src = { x: 90, y: 140 }, mirror = { x: 90, y: 60 };
  const t = arrivalTimes(line, src, 1);
  check('collinear stations: mirror image fits equally well', misfitAt(line, t, mirror).misfit < 1e-12);
}

// 3. Enclosing stations give a smaller 95 % region than a line of stations (same noise)
{
  const q = hiddenQuake(4);
  const sig = 0.2;
  const obs = (st: Pt[]) => arrivalTimes(st, q.src, q.t0).map((t, i) => t + sig * q.draws[i]);
  const a = misfitMap(spread, obs(spread), 80), b = misfitMap(line, obs(line), 80);
  const Aa = regionArea(a, confidenceLevel(a.min, sig)), Ab = regionArea(b, confidenceLevel(b.min, sig));
  check('spread stations constrain the epicentre better than a line', Aa < Ab, `(${Aa.toFixed(0)} vs ${Ab.toFixed(0)} km²)`);
}

// 4. Marching squares: a circle gives a closed contour of the right radius
{
  const n = 60, data = new Float32Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) data[j * n + i] = Math.hypot(i - 30, j - 30);
  const segs = contourSegments(data, n, 12);
  let maxDev = 0;
  for (const [x1, y1, x2, y2] of segs) maxDev = Math.max(maxDev, Math.abs(Math.hypot(x1 - 30, y1 - 30) - 12), Math.abs(Math.hypot(x2 - 30, y2 - 30) - 12));
  check('contour of a cone is a circle', segs.length > 40 && maxDev < 0.1, `(${segs.length} segments, max deviation ${maxDev.toFixed(3)})`);
}

if (fails) { console.error(`${fails} check(s) failed`); process.exit(1); }
console.log('All checks passed.');
