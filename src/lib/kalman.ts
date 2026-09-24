/**
 * Kalman filter for 2D tracking with a constant-velocity model.
 * State s = [x, y, vx, vy], s_{k+1} = F s_k + w, w ~ N(0, Q) (white acceleration, intensity q),
 * GPS measures position: z_k = H s_k + v, v ~ N(0, r² I).
 */
import { rng } from './linalg.ts';

export type Vec4 = [number, number, number, number];
export type Mat4 = number[]; // 16, row-major

const mm = (A: Mat4, B: Mat4): Mat4 => {
  const C = new Array(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) C[i * 4 + j] += A[i * 4 + k] * B[k * 4 + j];
  return C;
};
const tr = (A: Mat4): Mat4 => A.map((_, k) => A[(k % 4) * 4 + Math.floor(k / 4)]);

export function F(dt: number): Mat4 { return [1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1]; }
export function Q(dt: number, q: number): Mat4 {
  const a = (dt ** 3) / 3, b = (dt ** 2) / 2, c = dt;
  return [a, 0, b, 0, 0, a, 0, b, b, 0, c, 0, 0, b, 0, c].map((v) => v * q);
}

export interface KF { s: Vec4; P: Mat4 }

export function init(z: [number, number], r: number): KF {
  return { s: [z[0], z[1], 0, 0], P: [r * r, 0, 0, 0, 0, r * r, 0, 0, 0, 0, 100, 0, 0, 0, 0, 100] };
}

export function predict(kf: KF, dt: number, q: number): KF {
  const f = F(dt), s = kf.s;
  const sp: Vec4 = [s[0] + dt * s[2], s[1] + dt * s[3], s[2], s[3]];
  const P = mm(mm(f, kf.P), tr(f)).map((v, k) => v + Q(dt, q)[k]);
  return { s: sp, P };
}

/** Measurement update with position-only H; returns the new state and the Kalman gain (4×2). */
export function update(kf: KF, z: [number, number], r: number): KF & { K: number[] } {
  const P = kf.P;
  // S = H P Hᵀ + R (2×2)
  const S00 = P[0] + r * r, S01 = P[1], S10 = P[4], S11 = P[5] + r * r;
  const det = S00 * S11 - S01 * S10;
  const i00 = S11 / det, i01 = -S01 / det, i10 = -S10 / det, i11 = S00 / det;
  // K = P Hᵀ S⁻¹ (4×2); P Hᵀ = columns 0,1 of P
  const K: number[] = [];
  for (let i = 0; i < 4; i++) {
    const a = P[i * 4 + 0], b = P[i * 4 + 1];
    K.push(a * i00 + b * i10, a * i01 + b * i11);
  }
  const y0 = z[0] - kf.s[0], y1 = z[1] - kf.s[1];
  const s = kf.s.map((v, i) => v + K[2 * i] * y0 + K[2 * i + 1] * y1) as Vec4;
  // P = (I − K H) P
  const Pn: Mat4 = new Array(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) Pn[i * 4 + j] = P[i * 4 + j] - K[2 * i] * P[j] - K[2 * i + 1] * P[4 + j];
  // symmetrise
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) { const m = 0.5 * (Pn[i * 4 + j] + Pn[j * 4 + i]); Pn[i * 4 + j] = Pn[j * 4 + i] = m; }
  return { s, P: Pn, K };
}

/** 2σ ellipse of the position covariance: semi-axes and rotation angle. */
export function ellipse(P: Mat4, k = 2) {
  const a = P[0], b = P[1], c = P[5];
  const tmp = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const l1 = (a + c) / 2 + tmp, l2 = Math.max((a + c) / 2 - tmp, 0);
  return { rx: k * Math.sqrt(l1), ry: k * Math.sqrt(l2), angle: 0.5 * Math.atan2(2 * b, a - c) };
}

/** A smooth drone trajectory (metres) sampled every dt seconds, with noisy GPS fixes. */
export function scenario(n: number, dt: number, r: number, seed: number) {
  const R = rng(seed), truth: [number, number][] = [], meas: [number, number][] = [];
  for (let k = 0; k < n; k++) {
    const t = k * dt, w = 0.03;
    const x = 50 + 34 * Math.sin(w * t) + 4 * Math.sin(3 * w * t);
    const y = 50 + 26 * Math.sin(2 * w * t + 0.6);
    truth.push([x, y]);
    meas.push([x + r * R.normal(), y + r * R.normal()]);
  }
  return { truth, meas };
}

/** Run the filter over a whole sequence (for tests and readouts). */
export function runFilter(meas: [number, number][], dt: number, q: number, r: number) {
  let kf: KF = init(meas[0], r);
  const est: [number, number][] = [[kf.s[0], kf.s[1]]];
  for (let k = 1; k < meas.length; k++) {
    kf = update(predict(kf, dt, q), meas[k], r);
    est.push([kf.s[0], kf.s[1]]);
  }
  return est;
}
