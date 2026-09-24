/** Objectives and first-order optimisers for the gradient-descent explorer. */

export type Vec2 = [number, number];
export interface Objective {
  id: string;
  label: string;
  f: (x: number, y: number, k: number) => number;
  g: (x: number, y: number, k: number) => Vec2;
  domain: [number, number, number, number]; // xmin, xmax, ymin, ymax
  start: Vec2;
  minima: Vec2[];
  /** log-spaced contour levels look better for this objective */
  logLevels: boolean;
  /** sensible default log10 step size */
  logAlpha: number;
  note: string;
}

export const objectives: Objective[] = [
  {
    id: 'valley',
    label: 'Elongated quadratic valley',
    f: (x, y, k) => 0.5 * (x * x + k * y * y),
    g: (x, y, k) => [x, k * y],
    domain: [-3, 3, -2, 2],
    start: [-2.6, 1.4],
    minima: [[0, 0]],
    logLevels: true,
    logAlpha: -1.3,
    note: 'f(x, y) = ½(x² + κy²). The Hessian has eigenvalues 1 and κ. Plain gradient descent is stable only when α < 2/κ, yet progress along x scales with α. A large κ (ill-conditioning) forces slow convergence.',
  },
  {
    id: 'rosenbrock',
    label: 'Rosenbrock (curved valley)',
    f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
    domain: [-2, 2, -1, 3],
    start: [-1.5, 2.5],
    minima: [[1, 1]],
    logLevels: true,
    logAlpha: -2.2,
    note: 'f(x, y) = (1 − x)² + 10(y − x²)². The minimum at (1, 1) sits at the end of a narrow, curved valley. The local curvature changes along the path, so no single step size is ideal.',
  },
  {
    id: 'himmelblau',
    label: 'Himmelblau (four minima)',
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    g: (x, y) => [
      4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
      2 * (x * x + y - 11) + 4 * y * (x + y * y - 7),
    ],
    domain: [-5, 5, -5, 5],
    start: [-0.5, -0.5],
    minima: [[3, 2], [-2.805118, 3.131312], [-3.77931, -3.283186], [3.584428, -1.848126]],
    logLevels: true,
    logAlpha: -2.3,
    note: 'Non-convex with four global minima. Which one you reach depends on the starting point and on the step size: the answer of a local method is not unique.',
  },
];

export type Method = 'gd' | 'momentum' | 'adam';
export interface RunOptions { alpha: number; beta?: number; steps: number; k?: number }

/** Returns the path x_0..x_n (stops early on divergence). */
export function run(obj: Objective, x0: Vec2, method: Method, o: RunOptions): { path: Vec2[]; loss: number[]; diverged: boolean } {
  const k = o.k ?? 10;
  const beta = o.beta ?? 0.9;
  let [x, y] = x0;
  let vx = 0, vy = 0; // momentum / first moment
  let sx = 0, sy = 0; // second moment (Adam)
  const b2 = 0.999, eps = 1e-8;
  const path: Vec2[] = [[x, y]];
  const loss: number[] = [obj.f(x, y, k)];
  let diverged = false;
  for (let t = 1; t <= o.steps; t++) {
    const [gx, gy] = obj.g(x, y, k);
    if (method === 'gd') {
      x -= o.alpha * gx; y -= o.alpha * gy;
    } else if (method === 'momentum') {
      vx = beta * vx - o.alpha * gx; vy = beta * vy - o.alpha * gy;
      x += vx; y += vy;
    } else {
      vx = beta * vx + (1 - beta) * gx; vy = beta * vy + (1 - beta) * gy;
      sx = b2 * sx + (1 - b2) * gx * gx; sy = b2 * sy + (1 - b2) * gy * gy;
      const mhx = vx / (1 - beta ** t), mhy = vy / (1 - beta ** t);
      const shx = sx / (1 - b2 ** t), shy = sy / (1 - b2 ** t);
      x -= (o.alpha * mhx) / (Math.sqrt(shx) + eps);
      y -= (o.alpha * mhy) / (Math.sqrt(shy) + eps);
    }
    const fx = obj.f(x, y, k);
    if (!Number.isFinite(fx) || Math.abs(x) > 1e6 || Math.abs(y) > 1e6) { diverged = true; break; }
    path.push([x, y]);
    loss.push(fx);
  }
  return { path, loss, diverged };
}
