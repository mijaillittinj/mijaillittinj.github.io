/**
 * Registry of open learning modules. A module with status 'planned' appears in
 * the index as "in preparation" and has no page yet. See TEACHING_WORKFLOW.md.
 */

export type LearnCategory = 'machine-learning' | 'optimization' | 'inverse-problems' | 'numerical-methods' | 'thermal-systems' | 'uncertainty';

export interface LearnModule {
  slug: string; // route under /learn/
  category: LearnCategory;
  title: string;
  question: string; // what the student can manipulate / discover
  status: 'live' | 'planned';
  minutes?: number;
  related?: { label: string; href: string }[];
}

export const categories: Record<LearnCategory, { title: string; blurb: string }> = {
  'inverse-problems': {
    title: 'Inverse problems',
    blurb: 'Recovering causes from effects: projections, ill-conditioning and regularisation.',
  },
  optimization: {
    title: 'Optimization',
    blurb: 'How iterative methods move through a loss landscape, and why they sometimes fail.',
  },
  'machine-learning': {
    title: 'Machine learning',
    blurb: 'Learning from data, and what changes when a physical law enters the loss.',
  },
  'numerical-methods': {
    title: 'Numerical methods',
    blurb: 'Discretisation, stability and the cost of computing things accurately.',
  },
  'thermal-systems': {
    title: 'Thermal systems',
    blurb: 'Models of heat transfer used in design: from lumped bodies to radiation exchange.',
  },
  uncertainty: {
    title: 'Uncertainty',
    blurb: 'How noise in the data becomes uncertainty in what we infer, and how to combine models with data.',
  },
};

export const modules: LearnModule[] = [
  {
    slug: 'inverse-problems/abel-transform',
    category: 'inverse-problems',
    title: 'Seeing inside an axisymmetric object: the Abel transform',
    question: 'Shape a radial field, project it, add noise, and watch direct inversion fail while regularisation recovers it.',
    status: 'live',
    minutes: 15,
    related: [
      { label: 'pysat-abel', href: '/software/pysat-abel/' },
      { label: 'Carbon 2026 case study', href: '/publications/carbon-2026/' },
    ],
  },
  {
    slug: 'optimization/gradient-descent',
    category: 'optimization',
    title: 'Gradient descent, momentum and Adam on a loss landscape',
    question: 'Pick a starting point and a step size, and compare how three methods travel to (or away from) a minimum.',
    status: 'live',
    minutes: 12,
    related: [{ label: 'Course: thermal systems design', href: '/teaching/thermal-systems-design/' }],
  },
  {
    slug: 'machine-learning/physics-informed-learning',
    category: 'machine-learning',
    title: 'Data only or data plus physics? A small physics-informed network',
    question: 'Train a tiny network on five noisy sensors, then add the heat equation to the loss and infer a hidden parameter.',
    status: 'live',
    minutes: 15,
    related: [{ label: 'Research: physics-informed learning', href: '/research/#physics-informed' }],
  },
  {
    slug: 'inverse-problems/svd-noise-amplification',
    category: 'inverse-problems',
    title: 'Why inversion amplifies noise: the singular value decomposition',
    question: 'Decompose a blur into singular values, see which components the data can resolve, and filter the rest with truncation or Tikhonov.',
    status: 'live',
    minutes: 15,
    related: [
      { label: 'Module: the Abel transform', href: '/learn/inverse-problems/abel-transform/' },
      { label: 'Research: regularised reconstruction', href: '/research/#reconstruction' },
    ],
  },
  {
    slug: 'uncertainty/bayesian-mcmc',
    category: 'uncertainty',
    title: 'Bayesian inference with Markov chain Monte Carlo',
    question: 'Calibrate a cooling law from a few noisy readings and let a Metropolis sampler map every parameter value compatible with the data.',
    status: 'live',
    minutes: 15,
    related: [{ label: 'Module: tracking with a Kalman filter', href: '/learn/uncertainty/kalman-filter/' }],
  },
  {
    slug: 'machine-learning/neural-network-from-scratch',
    category: 'machine-learning',
    title: 'A neural network from scratch: fitting, overfitting and generalisation',
    question: 'Train a small network with hand-written backpropagation and watch the validation error reveal overfitting.',
    status: 'live',
    minutes: 15,
    related: [
      { label: 'Module: physics-informed learning', href: '/learn/machine-learning/physics-informed-learning/' },
      { label: 'Course: thermal systems design', href: '/teaching/thermal-systems-design/' },
    ],
  },
  {
    slug: 'uncertainty/kalman-filter',
    category: 'uncertainty',
    title: 'Tracking a drone with a Kalman filter',
    question: 'Fuse a motion model with noisy GPS fixes and tune how much the filter trusts each of them.',
    status: 'live',
    minutes: 12,
    related: [{ label: 'Module: Bayesian inference with MCMC', href: '/learn/uncertainty/bayesian-mcmc/' }],
  },
  { slug: 'optimization/sgd-adam-bfgs', category: 'optimization', title: 'SGD, Adam and BFGS compared', question: 'Race first- and quasi-second-order methods on the same problem.', status: 'planned' },
  { slug: 'uncertainty/propagation', category: 'uncertainty', title: 'Propagating uncertainty through a model', question: 'Push a distribution of inputs through a nonlinear model and see what comes out.', status: 'planned' },
  { slug: 'numerical-methods/heat-equation', category: 'numerical-methods', title: 'The heat equation and explicit time stepping', question: 'Increase the time step until the scheme becomes unstable.', status: 'planned' },
  { slug: 'thermal-systems/view-factors', category: 'thermal-systems', title: 'View factors and radiation exchange', question: 'Move and tilt a receiver and watch the fraction of radiation it intercepts.', status: 'planned' },
  { slug: 'machine-learning/surrogate-models', category: 'machine-learning', title: 'Surrogate models of expensive simulations', question: 'Replace a slow model with a fast approximation and test where it breaks.', status: 'planned' },
];
