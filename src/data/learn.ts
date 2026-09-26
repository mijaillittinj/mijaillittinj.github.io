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
    blurb: 'Recovering hidden causes from indirect measurements, and why noise makes this hard.',
  },
  optimization: {
    title: 'Optimization',
    blurb: 'How iterative methods search for the minimum of a function, and why they sometimes fail.',
  },
  'machine-learning': {
    title: 'Machine learning',
    blurb: 'Fitting models to data, and what changes when a physical law is added to the training.',
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
    blurb: 'How noise in the data becomes uncertainty in the result, and how to combine a model with measurements.',
  },
};

export const modules: LearnModule[] = [
  {
    slug: 'inverse-problems/abel-transform',
    category: 'inverse-problems',
    title: 'Seeing inside an axisymmetric object: the Abel transform',
    question: 'A camera sees an axisymmetric object only as a projection. Shape a radial profile, project it, add noise, and compare direct inversion, which amplifies the noise, with a regularised inversion that recovers the profile.',
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
    question: 'Choose a starting point and a step size, and compare how three optimisation methods approach the minimum of a function, or fail to.',
    status: 'live',
    minutes: 12,
    related: [{ label: 'Course: thermal systems design', href: '/teaching/thermal-systems-design/' }],
  },
  {
    slug: 'machine-learning/physics-informed-learning',
    category: 'machine-learning',
    title: 'Data only or data plus physics? A small physics-informed network',
    question: 'Fit a small neural network to four noisy temperature readings, then add the heat equation to its training and use it to estimate a parameter that is not measured.',
    status: 'live',
    minutes: 15,
    related: [{ label: 'Research: physics-informed learning', href: '/research/#physics-informed' }],
  },
  {
    slug: 'inverse-problems/svd-noise-amplification',
    category: 'inverse-problems',
    title: 'Why inversion amplifies noise: the singular value decomposition',
    question: 'Break a blurring operator into independent components, see which of them the data can resolve, and filter out the rest with truncation or Tikhonov regularisation.',
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
    question: 'Estimate the two parameters of a cooling law from a few noisy readings, and let a random sampler map every combination of values that is compatible with the data.',
    status: 'live',
    minutes: 15,
    related: [{ label: 'Module: tracking with a Kalman filter', href: '/learn/uncertainty/kalman-filter/' }],
  },
  {
    slug: 'machine-learning/neural-network-from-scratch',
    category: 'machine-learning',
    title: 'A neural network from scratch: fitting, overfitting and generalisation',
    question: 'Train a small neural network written from first principles and watch the error on unseen data reveal when it starts to memorise the noise.',
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
    question: 'Combine a simple model of motion with noisy GPS positions, and adjust how much the filter trusts each of them.',
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
