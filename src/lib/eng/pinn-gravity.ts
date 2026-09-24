/**
 * Physics-constrained network for gravity inversion (mining tab).
 *
 * A coordinate network gives the density contrast on the subsurface section,
 *     Δρ_θ(x, z) = ρ_s · softplus(c + Σ_j a_j tanh(p_j ξ + q_j ζ + b_j)),
 * with ξ = x / 500 m and ζ = 2z / Z − 1, evaluated at the centres of a grid of cells.
 * Each cell acts as a 2D line mass, so the anomaly is linear in the cell densities,
 * g = G Δρ, and the loss is
 *     L = ‖G Δρ_θ − d‖² / ‖d‖² + α · mean (w(z) Δρ_θ / ρ_s)²,
 * with w(z) = 1 (plain minimum-norm) or w(z) = ((z₀ + z_ref) / (z₀ + z))^{β/2}
 * (depth weighting, Li & Oldenburg). Gravity alone cannot fix depth: without depth
 * weighting the smallest model that fits the data sits near the surface.
 */
import { rng } from '../linalg.ts';
import { G as GRAV } from './gravity.ts';

const MGAL = 1e5;

export interface GravGrid { nx: number; nz: number; xh: number; zmax: number; xc: Float64Array; zc: Float64Array; area: number }

export function makeGrid(nx = 40, nz = 20, xh = 500, zmax = 350): GravGrid {
  const dx = (2 * xh) / nx, dz = zmax / nz;
  const xc = Float64Array.from({ length: nx }, (_, i) => -xh + (i + 0.5) * dx);
  const zc = Float64Array.from({ length: nz }, (_, j) => (j + 0.5) * dz);
  return { nx, nz, xh, zmax, xc, zc, area: dx * dz };
}

/** Kernel (stations × cells), mGal per kg/m³: line mass ρ·area at each cell centre. Cell index = j·nx + i. */
export function kernel(grid: GravGrid, stations: ArrayLike<number>): Float64Array {
  const n = stations.length, m = grid.nx * grid.nz, K = new Float64Array(n * m);
  for (let s = 0; s < n; s++) for (let j = 0; j < grid.nz; j++) for (let i = 0; i < grid.nx; i++) {
    const d = stations[s] - grid.xc[i], z = grid.zc[j];
    K[s * m + j * grid.nx + i] = (2 * GRAV * grid.area * z * MGAL) / (d * d + z * z);
  }
  return K;
}

export interface GravNet {
  a: Float64Array; p: Float64Array; q: Float64Array; b: Float64Array; c: Float64Array;
  m: Float64Array; v: Float64Array; t: number;
}

export const RHO_S = 1000; // kg/m³

export function initGravNet(hidden: number, seed: number): GravNet {
  const R = rng(seed), P = 4 * hidden + 1;
  const net: GravNet = {
    a: new Float64Array(hidden), p: new Float64Array(hidden), q: new Float64Array(hidden), b: new Float64Array(hidden),
    c: new Float64Array([-4]), m: new Float64Array(P), v: new Float64Array(P), t: 0,
  };
  for (let j = 0; j < hidden; j++) {
    net.p[j] = R.normal() * 3; net.q[j] = R.normal() * 3; net.b[j] = R.normal(); net.a[j] = R.normal() * 0.1;
  }
  return net;
}

const softplus = (z: number) => (z > 20 ? z : Math.log1p(Math.exp(z)));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Density contrast (kg/m³) in every cell. */
export function evalGravNet(net: GravNet, grid: GravGrid): Float64Array {
  const out = new Float64Array(grid.nx * grid.nz);
  for (let j = 0; j < grid.nz; j++) for (let i = 0; i < grid.nx; i++) {
    const xi = grid.xc[i] / grid.xh, ze = (2 * grid.zc[j]) / grid.zmax - 1;
    let z = net.c[0];
    for (let h = 0; h < net.a.length; h++) z += net.a[h] * Math.tanh(net.p[h] * xi + net.q[h] * ze + net.b[h]);
    out[j * grid.nx + i] = RHO_S * softplus(z);
  }
  return out;
}

export function depthWeights(grid: GravGrid, on: boolean, beta = 1): Float64Array {
  const z0 = grid.zmax / grid.nz / 2, zref = grid.zmax / 2;
  return Float64Array.from({ length: grid.nx * grid.nz }, (_, k) => {
    if (!on) return 1;
    const z = grid.zc[Math.floor(k / grid.nx)];
    return ((z0 + zref) / (z0 + z)) ** (beta / 2);
  });
}

/** Loss and gradient with respect to [a, p, q, b, c]. */
export function gravLossGrad(net: GravNet, grid: GravGrid, K: Float64Array, d: ArrayLike<number>, w: Float64Array, alpha: number) {
  const H = net.a.length, m = grid.nx * grid.nz, n = d.length;
  const S = new Float64Array(m * H), zc = new Float64Array(m), rho = new Float64Array(m);
  const XI = new Float64Array(m), ZE = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const i = k % grid.nx, j = Math.floor(k / grid.nx);
    const xi = grid.xc[i] / grid.xh, ze = (2 * grid.zc[j]) / grid.zmax - 1;
    XI[k] = xi; ZE[k] = ze;
    let z = net.c[0];
    for (let h = 0; h < H; h++) { const s = Math.tanh(net.p[h] * xi + net.q[h] * ze + net.b[h]); S[k * H + h] = s; z += net.a[h] * s; }
    zc[k] = z; rho[k] = RHO_S * softplus(z);
  }
  let dd = 0;
  for (let s = 0; s < n; s++) dd += d[s] * d[s];
  dd = dd || 1;
  const res = new Float64Array(n);
  let Ld = 0;
  for (let s = 0; s < n; s++) {
    let g = -d[s];
    for (let k = 0; k < m; k++) g += K[s * m + k] * rho[k];
    res[s] = g; Ld += g * g;
  }
  Ld /= dd;
  let Lr = 0;
  const grho = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    let t = 0;
    for (let s = 0; s < n; s++) t += K[s * m + k] * res[s];
    const r = (w[k] * rho[k]) / RHO_S;
    Lr += r * r;
    grho[k] = (2 / dd) * t + ((2 * alpha) / m) * r * (w[k] / RHO_S);
  }
  Lr /= m;
  const grad = new Float64Array(4 * H + 1);
  for (let k = 0; k < m; k++) {
    const gz = grho[k] * RHO_S * sigmoid(zc[k]);
    if (gz === 0) continue;
    for (let h = 0; h < H; h++) {
      const s = S[k * H + h], dh = gz * net.a[h] * (1 - s * s);
      grad[h] += gz * s;
      grad[H + h] += dh * XI[k];
      grad[2 * H + h] += dh * ZE[k];
      grad[3 * H + h] += dh;
    }
    grad[4 * H] += gz;
  }
  return { loss: Ld + alpha * Lr, Ld, Lr, grad, rho };
}

export function gravAdam(net: GravNet, grad: Float64Array, lr = 0.02, b1 = 0.9, b2 = 0.999) {
  net.t++;
  const c1 = 1 - b1 ** net.t, c2 = 1 - b2 ** net.t;
  let k = 0;
  for (const arr of [net.a, net.p, net.q, net.b, net.c]) for (let j = 0; j < arr.length; j++, k++) {
    net.m[k] = b1 * net.m[k] + (1 - b1) * grad[k];
    net.v[k] = b2 * net.v[k] + (1 - b2) * grad[k] * grad[k];
    arr[j] -= (lr * (net.m[k] / c1)) / (Math.sqrt(net.v[k] / c2) + 1e-8);
  }
}

/**
 * Centre (x, z) of the recovered body: mass-weighted over the cells above `frac` of the
 * peak density, so the faint background left by the softplus output does not pull it
 * towards mid-depth. `mass` is the total excess mass per metre (10⁶ kg/m), all cells.
 */
export function centroid(rho: ArrayLike<number>, grid: GravGrid, frac = 0.3): { x: number; z: number; mass: number } {
  let rmax = 0, Mall = 0;
  for (let k = 0; k < rho.length; k++) { rmax = Math.max(rmax, rho[k]); Mall += rho[k]; }
  let M = 0, X = 0, Z = 0;
  for (let k = 0; k < rho.length; k++) {
    if (rho[k] < frac * rmax) continue;
    const x = grid.xc[k % grid.nx], z = grid.zc[Math.floor(k / grid.nx)];
    M += rho[k]; X += rho[k] * x; Z += rho[k] * z;
  }
  return { x: X / (M || 1), z: Z / (M || 1), mass: (Mall * grid.area) / 1e6 };
}
