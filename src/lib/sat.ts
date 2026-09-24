/**
 * Spline-based Abel transform (SAT) with signal trapping, for the flame tomography tab.
 *
 * Method: Littin et al., Fuel 374 (2024) 132365; released as the Python package pysat-abel.
 * The radial field is a clamped cubic B-spline with zero slope on the axis; its
 * coefficients are fitted to the line-of-sight signal with a curvature (second
 * difference) penalty. For emission, the light of each annulus is attenuated by the soot
 * between it and the camera (signal trapping, or self-absorption), so the forward
 * operator depends on the absorption field; including it removes the bias of the
 * plain Abel model.
 *
 * Geometry: n annuli of width 1/n (radius normalised by R); ray i passes at offset
 * y_i = i/n, as in `chordMatrix`.
 */
import { zeros, gram, cholSolve, type Mat } from './linalg.ts';
import { chordMatrix } from './abel.ts';

/**
 * Emission operator with self-absorption: S_i = Σ_j T_ij J_j, where each annulus j is
 * crossed twice by ray i (far and near halves, length L_ij / 2 each), and each half is
 * attenuated by the optical depth between its midpoint and the camera.
 * kappa: absorption coefficient per annulus, in units of 1/R. kappa = 0 gives the Abel matrix.
 */
export function trappingMatrix(n: number, kappa: ArrayLike<number>): Mat {
  const L = chordMatrix(n, 1);
  const T = zeros(n, n);
  for (let i = 0; i < n; i++) {
    // optical depth of each half chord of ray i
    const h = new Float64Array(n);
    let total = 0; // depth of all near halves (from annulus i out to the edge)
    for (let j = i; j < n; j++) { h[j] = (kappa[j] * L.a[i * n + j]) / 2; total += h[j]; }
    // near halves: light from annulus j crosses the near halves of j+1..n-1
    let outside = 0;
    for (let j = n - 1; j >= i; j--) {
      const near = Math.exp(-(outside + h[j] / 2));
      outside += h[j];
      T.a[i * n + j] += (L.a[i * n + j] / 2) * near;
    }
    // far halves: light from annulus j crosses far halves j-1..i, then every near half
    let inner = 0;
    for (let j = i; j < n; j++) {
      const far = Math.exp(-(total + inner + h[j] / 2));
      inner += h[j];
      T.a[i * n + j] += (L.a[i * n + j] / 2) * far;
    }
  }
  return T;
}

/** Cox–de Boor evaluation of all cubic B-splines on a clamped uniform knot vector. */
export function splineBasis(n: number, segments: number): Mat {
  const k = 3;
  const knots: number[] = [];
  for (let q = 0; q < k; q++) knots.push(0);
  for (let q = 0; q <= segments; q++) knots.push(q / segments);
  for (let q = 0; q < k; q++) knots.push(1);
  const M = knots.length - k - 1; // segments + 3
  const B = zeros(n, M);
  for (let i = 0; i < n; i++) {
    const x = Math.min((i + 0.5) / n, 1 - 1e-12);
    // degree 0
    let N = new Float64Array(knots.length - 1);
    for (let q = 0; q < N.length; q++) N[q] = x >= knots[q] && x < knots[q + 1] ? 1 : 0;
    for (let d = 1; d <= k; d++) {
      const Nn = new Float64Array(knots.length - 1 - d);
      for (let q = 0; q < Nn.length; q++) {
        const a = knots[q + d] - knots[q], b = knots[q + d + 1] - knots[q + 1];
        Nn[q] = (a > 0 ? ((x - knots[q]) / a) * N[q] : 0) + (b > 0 ? ((knots[q + d + 1] - x) / b) * N[q + 1] : 0);
      }
      N = Nn;
    }
    for (let q = 0; q < M; q++) B.a[i * M + q] = N[q];
  }
  // zero slope on the axis: the first two coefficients are equal (merge their columns)
  const S = zeros(n, M - 1);
  for (let i = 0; i < n; i++) {
    S.a[i * (M - 1)] = B.a[i * M] + B.a[i * M + 1];
    for (let q = 2; q < M; q++) S.a[i * (M - 1) + q - 1] = B.a[i * M + q];
  }
  return S;
}

const mul = (A: Mat, B: Mat): Mat => {
  const C = zeros(A.n, B.m);
  for (let i = 0; i < A.n; i++) for (let k = 0; k < A.m; k++) {
    const a = A.a[i * A.m + k];
    if (a) for (let j = 0; j < B.m; j++) C.a[i * B.m + j] += a * B.a[k * B.m + j];
  }
  return C;
};

/**
 * SAT fit: minimise ‖A B c − g‖² + λ s ‖D₂ c‖², return f = B c.
 * s = tr(GᵀG) / tr(D₂ᵀD₂) makes λ dimensionless.
 */
export function satSolve(A: Mat, g: ArrayLike<number>, B: Mat, lambda: number): Float64Array {
  const G = mul(A, B);
  const S = gram(G);
  const M = B.m;
  const DtD = zeros(M, M);
  for (let r = 0; r + 2 < M; r++) {
    const idx = [r, r + 1, r + 2], w = [1, -2, 1];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) DtD.a[idx[a] * M + idx[b]] += w[a] * w[b];
  }
  let tg = 0, td = 0;
  for (let q = 0; q < M; q++) { tg += S.a[q * M + q]; td += DtD.a[q * M + q]; }
  const s = (lambda * tg) / (td || 1);
  for (let q = 0; q < M * M; q++) S.a[q] += s * DtD.a[q] + (q % (M + 1) === 0 ? 1e-12 * tg : 0);
  const rhs = new Float64Array(M);
  for (let q = 0; q < M; q++) { let t = 0; for (let i = 0; i < G.n; i++) t += G.a[i * M + q] * g[i]; rhs[q] = t; }
  const c = cholSolve(S, rhs) ?? new Float64Array(M);
  const f = new Float64Array(B.n);
  for (let i = 0; i < B.n; i++) { let t = 0; for (let q = 0; q < M; q++) t += B.a[i * M + q] * c[q]; f[i] = t; }
  return f;
}

/** Soot spectral emission (Rayleigh regime) up to a constant: f_s E(m)/λ · B_λ(T), λ in m. */
export function sootEmission(fs: number, T: number, lambda = 660e-9): number {
  const c2 = 1.438777e-2;
  return (fs / lambda) / (lambda ** 5 * (Math.exp(c2 / (lambda * T)) - 1));
}
