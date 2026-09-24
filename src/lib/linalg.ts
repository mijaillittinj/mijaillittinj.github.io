/** Minimal dense linear algebra for in-browser demos (row-major Float64Array). */

export type Mat = { n: number; m: number; a: Float64Array };

export function zeros(n: number, m: number): Mat {
  return { n, m, a: new Float64Array(n * m) };
}

export function matvec(A: Mat, x: ArrayLike<number>, out = new Float64Array(A.n)): Float64Array {
  for (let i = 0; i < A.n; i++) {
    let s = 0;
    const row = i * A.m;
    for (let j = 0; j < A.m; j++) s += A.a[row + j] * x[j];
    out[i] = s;
  }
  return out;
}

/** AᵀA (m×m). */
export function gram(A: Mat): Mat {
  const G = zeros(A.m, A.m);
  for (let k = 0; k < A.n; k++) {
    const row = k * A.m;
    for (let i = 0; i < A.m; i++) {
      const aki = A.a[row + i];
      if (aki === 0) continue;
      for (let j = i; j < A.m; j++) G.a[i * A.m + j] += aki * A.a[row + j];
    }
  }
  for (let i = 0; i < A.m; i++) for (let j = 0; j < i; j++) G.a[i * A.m + j] = G.a[j * A.m + i];
  return G;
}

/** Aᵀy. */
export function tvec(A: Mat, y: ArrayLike<number>): Float64Array {
  const out = new Float64Array(A.m);
  for (let k = 0; k < A.n; k++) {
    const row = k * A.m;
    const yk = y[k];
    for (let j = 0; j < A.m; j++) out[j] += A.a[row + j] * yk;
  }
  return out;
}

/** Solve SPD system via Cholesky. Returns null if not positive definite. */
export function cholSolve(S: Mat, b: ArrayLike<number>): Float64Array | null {
  const n = S.n;
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = S.a[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) return null;
        L[i * n + i] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

/** Back substitution for an upper-triangular matrix. */
export function solveUpper(U: Mat, b: ArrayLike<number>): Float64Array {
  const n = U.n;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= U.a[i * n + k] * x[k];
    x[i] = s / U.a[i * n + i];
  }
  return x;
}

/** Seeded PRNG (mulberry32) and Gaussian samples, so demos are reproducible. */
export function rng(seed = 1) {
  let t = seed >>> 0;
  const uni = () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => {
    const u = Math.max(uni(), 1e-12);
    const v = uni();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  return { uni, normal };
}

export function linspace(a: number, b: number, n: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
  return out;
}

export function rms(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / a.length);
}
