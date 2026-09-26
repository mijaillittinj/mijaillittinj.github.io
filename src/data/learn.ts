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
  /** Spanish title and question (the Spanish page lives under /es/learn/<slug>/). */
  es?: { title: string; question: string; related?: { label: string; href: string }[] };
}

export const categories: Record<LearnCategory, { title: string; blurb: string; es?: { title: string; blurb: string } }> = {
  'inverse-problems': {
    title: 'Inverse problems',
    blurb: 'Recovering hidden causes from indirect measurements, and why noise makes this hard.',
    es: { title: 'Problemas inversos', blurb: 'Recuperar causas ocultas a partir de mediciones indirectas, y por qué el ruido lo hace difícil.' },
  },
  optimization: {
    title: 'Optimization',
    blurb: 'How iterative methods search for the minimum of a function, and why they sometimes fail.',
    es: { title: 'Optimización', blurb: 'Cómo buscan los métodos iterativos el mínimo de una función, y por qué a veces fallan.' },
  },
  'machine-learning': {
    title: 'Machine learning',
    blurb: 'Fitting models to data, and what changes when a physical law is added to the training.',
    es: { title: 'Aprendizaje automático', blurb: 'Ajustar modelos a datos, y qué cambia cuando se agrega una ley física al entrenamiento.' },
  },
  'numerical-methods': {
    title: 'Numerical methods',
    blurb: 'Discretisation, stability and the cost of computing things accurately.',
    es: { title: 'Métodos numéricos', blurb: 'Discretización, estabilidad y el costo de calcular con precisión.' },
  },
  'thermal-systems': {
    title: 'Thermal systems',
    blurb: 'Models of heat transfer used in design: from lumped bodies to radiation exchange.',
    es: { title: 'Sistemas térmicos', blurb: 'Modelos de transferencia de calor usados en diseño: de los cuerpos concentrados al intercambio por radiación.' },
  },
  uncertainty: {
    title: 'Uncertainty',
    blurb: 'How noise in the data becomes uncertainty in the result, and how to combine a model with measurements.',
    es: { title: 'Incertidumbre', blurb: 'Cómo el ruido de los datos se convierte en incertidumbre del resultado, y cómo combinar un modelo con mediciones.' },
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
    es: {
      title: 'Ver el interior de un objeto axisimétrico: la transformada de Abel',
      question: 'Una cámara solo ve un objeto axisimétrico como una proyección. Dé forma a un perfil radial, proyéctelo, agregue ruido y compare la inversión directa, que amplifica el ruido, con una inversión regularizada que recupera el perfil.',
      related: [
        { label: 'pysat-abel', href: '/software/pysat-abel/' },
        { label: 'Caso de estudio Carbon 2026', href: '/publications/carbon-2026/' },
      ],
    },
  },
  {
    slug: 'optimization/gradient-descent',
    category: 'optimization',
    title: 'Gradient descent, momentum and Adam on a loss landscape',
    question: 'Choose a starting point and a step size, and compare how three optimisation methods approach the minimum of a function, or fail to.',
    status: 'live',
    minutes: 12,
    related: [{ label: 'Course: thermal systems design', href: '/teaching/thermal-systems-design/' }],
    es: {
      title: 'Descenso de gradiente, momentum y Adam sobre una función de pérdida',
      question: 'Elija un punto de partida y un tamaño de paso, y compare cómo tres métodos de optimización se acercan al mínimo de una función, o no lo logran.',
      related: [{ label: 'Curso: diseño de sistemas térmicos', href: '/teaching/thermal-systems-design/' }],
    },
  },
  {
    slug: 'machine-learning/physics-informed-learning',
    category: 'machine-learning',
    title: 'Data only or data plus physics? A small physics-informed network',
    question: 'Fit a small neural network to four noisy temperature readings, then add the heat equation to its training and use it to estimate a parameter that is not measured.',
    status: 'live',
    minutes: 15,
    related: [{ label: 'Research: physics-informed learning', href: '/research/#physics-informed' }],
    es: {
      title: '¿Solo datos o datos más física? Una pequeña red informada por la física',
      question: 'Ajuste una pequeña red neuronal a cuatro lecturas de temperatura con ruido, luego agregue la ecuación del calor a su entrenamiento y úsela para estimar un parámetro que no se mide.',
      related: [{ label: 'Investigación: aprendizaje informado por la física', href: '/es/research/#physics-informed' }],
    },
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
    es: {
      title: 'Por qué la inversión amplifica el ruido: la descomposición en valores singulares',
      question: 'Descomponga un operador de desenfoque en componentes independientes, vea cuáles pueden resolver los datos y filtre el resto con truncamiento o regularización de Tikhonov.',
      related: [
        { label: 'Módulo: la transformada de Abel', href: '/learn/inverse-problems/abel-transform/' },
        { label: 'Investigación: reconstrucción regularizada', href: '/es/research/#reconstruction' },
      ],
    },
  },
  {
    slug: 'uncertainty/bayesian-mcmc',
    category: 'uncertainty',
    title: 'Bayesian inference with Markov chain Monte Carlo',
    question: 'Estimate the two parameters of a cooling law from a few noisy readings, and let a random sampler map every combination of values that is compatible with the data.',
    status: 'live',
    minutes: 15,
    related: [{ label: 'Module: tracking with a Kalman filter', href: '/learn/uncertainty/kalman-filter/' }],
    es: {
      title: 'Inferencia bayesiana con Monte Carlo por cadenas de Markov',
      question: 'Calibre la ley de enfriamiento de Newton a partir de lecturas con ruido y obtenga no un valor, sino la distribución de valores compatibles con los datos.',
      related: [{ label: 'Módulo: seguimiento con un filtro de Kalman', href: '/learn/uncertainty/kalman-filter/' }],
    },
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
    es: {
      title: 'Una red neuronal desde cero: ajuste, sobreajuste y generalización',
      question: 'Entrene una pequeña red neuronal escrita desde sus principios y observe cómo el error sobre datos no vistos revela cuándo empieza a memorizar el ruido.',
      related: [
        { label: 'Módulo: aprendizaje informado por la física', href: '/learn/machine-learning/physics-informed-learning/' },
        { label: 'Curso: diseño de sistemas térmicos', href: '/teaching/thermal-systems-design/' },
      ],
    },
  },
  {
    slug: 'uncertainty/kalman-filter',
    category: 'uncertainty',
    title: 'Tracking a drone with a Kalman filter',
    question: 'Combine a simple model of motion with noisy GPS positions, and adjust how much the filter trusts each of them.',
    status: 'live',
    minutes: 12,
    related: [{ label: 'Module: Bayesian inference with MCMC', href: '/learn/uncertainty/bayesian-mcmc/' }],
    es: {
      title: 'Seguimiento de un dron con un filtro de Kalman',
      question: 'Combine un modelo simple del movimiento con posiciones GPS con ruido y ajuste cuánto confía el filtro en cada uno.',
      related: [{ label: 'Módulo: inferencia bayesiana con MCMC', href: '/learn/uncertainty/bayesian-mcmc/' }],
    },
  },
  // --- new modules (session 4): each agent edits only its own entry ---
  {
    slug: 'inverse-problems/ct-tomography',
    category: 'inverse-problems',
    title: 'Seeing inside the body: computed tomography',
    question: 'Watch a scanner record X-ray shadows from every angle, then rebuild the slice by back-projection, filtered back-projection or an iterative method, and see what few views, noise and a missing angle do.',
    status: 'live',
    minutes: 15,
    related: [
      { label: 'Module: the Abel transform', href: '/learn/inverse-problems/abel-transform/' },
      { label: 'Module: why inversion amplifies noise (SVD)', href: '/learn/inverse-problems/svd-noise-amplification/' },
      { label: 'Module: compressed sensing', href: '/learn/inverse-problems/compressed-sensing/' },
    ],
    es: {
      title: 'Ver dentro del cuerpo: tomografía computarizada',
      question: 'Vea cómo un escáner registra sombras de rayos X desde todos los ángulos y reconstruya el corte por retroproyección, retroproyección filtrada o un método iterativo, observando el efecto de pocas vistas, del ruido y de un ángulo faltante.',
      related: [
        { label: 'Módulo: la transformada de Abel', href: '/learn/inverse-problems/abel-transform/' },
        { label: 'Módulo: por qué la inversión amplifica el ruido (SVD)', href: '/learn/inverse-problems/svd-noise-amplification/' },
        { label: 'Módulo: muestreo compresivo', href: '/learn/inverse-problems/compressed-sensing/' },
      ],
    },
  },
  {
    slug: 'inverse-problems/compressed-sensing',
    category: 'inverse-problems',
    title: 'Recovering a signal from too few measurements: compressed sensing',
    question: 'Sample a signal made of a few tones at a handful of random times and recover it exactly with L1 minimisation, where least squares fails, then map how many samples are really needed.',
    status: 'live',
    minutes: 12,
    related: [
      { label: 'Module: computed tomography', href: '/learn/inverse-problems/ct-tomography/' },
      { label: 'Module: why inversion amplifies noise (SVD)', href: '/learn/inverse-problems/svd-noise-amplification/' },
    ],
    es: {
      title: 'Recuperar una señal con muy pocas mediciones: muestreo compresivo',
      question: 'Muestree una señal formada por unos pocos tonos en un puñado de instantes aleatorios y recupérela exactamente con minimización L1, donde fallan los mínimos cuadrados; luego vea cuántas muestras hacen falta realmente.',
      related: [
        { label: 'Módulo: tomografía computarizada', href: '/learn/inverse-problems/ct-tomography/' },
        { label: 'Módulo: por qué la inversión amplifica el ruido (SVD)', href: '/learn/inverse-problems/svd-noise-amplification/' },
      ],
    },
  },
  {
    slug: 'uncertainty/gaussian-processes',
    category: 'uncertainty',
    title: 'Gaussian processes: a model that knows what it does not know',
    question: 'Add measurements with a click and watch a curve with error bars adapt; tune the length scale and let the model choose where to measure next.',
    status: 'live',
    minutes: 12,
    related: [
      { label: 'Module: Bayesian inference with MCMC', href: '/learn/uncertainty/bayesian-mcmc/' },
      { label: 'Module: why inversion amplifies noise (SVD)', href: '/learn/inverse-problems/svd-noise-amplification/' },
    ],
    es: {
      title: 'Procesos gaussianos: un modelo que sabe lo que no sabe',
      question: 'Agregue mediciones con un clic y vea cómo se adapta una curva con barras de error; ajuste la escala de longitud y deje que el modelo elija dónde medir a continuación.',
      related: [
        { label: 'Módulo: inferencia bayesiana con MCMC', href: '/learn/uncertainty/bayesian-mcmc/' },
        { label: 'Módulo: por qué la inversión amplifica el ruido (SVD)', href: '/learn/inverse-problems/svd-noise-amplification/' },
      ],
    },
  },
  {
    slug: 'uncertainty/chaos-data-assimilation',
    category: 'uncertainty',
    title: 'Chaos and weather forecasting: the butterfly effect and data assimilation',
    question: 'Watch two almost identical trajectories of the Lorenz system diverge, then keep an ensemble forecast on track with noisy observations and an ensemble Kalman filter.',
    status: 'live',
    minutes: 15,
    related: [
      { label: 'Module: tracking with a Kalman filter', href: '/learn/uncertainty/kalman-filter/' },
      { label: 'Module: Bayesian inference with MCMC', href: '/learn/uncertainty/bayesian-mcmc/' },
    ],
    es: {
      title: 'Caos y pronóstico del tiempo: el efecto mariposa y la asimilación de datos',
      question: 'Vea cómo se separan dos trayectorias casi idénticas del sistema de Lorenz y mantenga un pronóstico por conjuntos en curso con observaciones ruidosas y un filtro de Kalman por conjuntos.',
      related: [
        { label: 'Módulo: seguimiento con un filtro de Kalman', href: '/learn/uncertainty/kalman-filter/' },
        { label: 'Módulo: inferencia bayesiana con MCMC', href: '/learn/uncertainty/bayesian-mcmc/' },
      ],
    },
  },
  {
    slug: 'machine-learning/discovering-equations',
    category: 'machine-learning',
    title: 'Discovering the equations of motion from data',
    question: 'Give a sparse regression noisy measurements of a pendulum, a predator–prey system or the Lorenz system, and see whether it writes down the right differential equations.',
    status: 'live',
    minutes: 12,
    related: [
      { label: 'Module: physics-informed learning', href: '/learn/machine-learning/physics-informed-learning/' },
      { label: 'Module: chaos and data assimilation', href: '/learn/uncertainty/chaos-data-assimilation/' },
    ],
    es: {
      title: 'Descubrir las ecuaciones de movimiento a partir de datos',
      question: 'Entregue a una regresión dispersa mediciones con ruido de un péndulo, de un sistema depredador–presa o del sistema de Lorenz, y vea si escribe las ecuaciones diferenciales correctas.',
      related: [
        { label: 'Módulo: aprendizaje informado por la física', href: '/learn/machine-learning/physics-informed-learning/' },
        { label: 'Módulo: caos y asimilación de datos', href: '/learn/uncertainty/chaos-data-assimilation/' },
      ],
    },
  },
  {
    slug: 'machine-learning/decision-boundaries',
    category: 'machine-learning',
    title: 'How a neural network learns to classify',
    question: 'Train a small network to separate two classes of points and watch its decision boundary bend, overfit and stay overconfident far from the data.',
    status: 'live',
    minutes: 10,
    related: [
      { label: 'Module: a neural network from scratch', href: '/learn/machine-learning/neural-network-from-scratch/' },
      { label: 'Module: Gaussian processes', href: '/learn/uncertainty/gaussian-processes/' },
      { label: 'Trustworthy AI (home page, computing tab)', href: '/#computing' },
    ],
    es: {
      title: 'Cómo aprende a clasificar una red neuronal',
      question: 'Entrene una red pequeña para separar dos clases de puntos y vea cómo su frontera de decisión se curva, se sobreajusta y sigue demasiado segura lejos de los datos.',
      related: [
        { label: 'Módulo: una red neuronal desde cero', href: '/learn/machine-learning/neural-network-from-scratch/' },
        { label: 'Módulo: procesos gaussianos', href: '/learn/uncertainty/gaussian-processes/' },
        { label: 'IA confiable (pestaña de computación, inicio)', href: '/es/#computing' },
      ],
    },
  },
  { slug: 'optimization/sgd-adam-bfgs', category: 'optimization', title: 'SGD, Adam and BFGS compared', question: 'Race first- and quasi-second-order methods on the same problem.', status: 'planned' },
  { slug: 'uncertainty/propagation', category: 'uncertainty', title: 'Propagating uncertainty through a model', question: 'Push a distribution of inputs through a nonlinear model and see what comes out.', status: 'planned' },
  { slug: 'numerical-methods/heat-equation', category: 'numerical-methods', title: 'The heat equation and explicit time stepping', question: 'Increase the time step until the scheme becomes unstable.', status: 'planned' },
  { slug: 'thermal-systems/view-factors', category: 'thermal-systems', title: 'View factors and radiation exchange', question: 'Move and tilt a receiver and watch the fraction of radiation it intercepts.', status: 'planned' },
  { slug: 'machine-learning/surrogate-models', category: 'machine-learning', title: 'Surrogate models of expensive simulations', question: 'Replace a slow model with a fast approximation and test where it breaks.', status: 'planned' },
];

/** Localised module and category texts (fall back to English). */
export function moduleText(m: LearnModule, lang: 'en' | 'es') {
  return lang === 'es' && m.es ? { title: m.es.title, question: m.es.question, related: m.es.related ?? m.related } : { title: m.title, question: m.question, related: m.related };
}
export function categoryText(c: LearnCategory, lang: 'en' | 'es') {
  const k = categories[c];
  return lang === 'es' && k.es ? k.es : { title: k.title, blurb: k.blurb };
}
