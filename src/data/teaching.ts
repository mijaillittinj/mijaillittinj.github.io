/**
 * Teaching record. Source: verified CV (2026-09-23) and course material in the
 * local teaching folders (see CONTENT_INVENTORY.md). Only verified information.
 */

export interface Course {
  code?: string;
  title: { en: string; es: string };
  institution: { en: string; es: string };
  role: { en: string; es: string };
  period: string;
  level: { en: string; es: string };
  description?: { en: string; es: string };
  page?: string;
  current: boolean;
}

export const courses: Course[] = [
  {
    code: 'ILN223',
    title: { en: 'Design and Optimization of Thermal Systems', es: 'Diseño y Optimización de Sistemas Térmicos' },
    institution: { en: 'UTFSM, Department of Industrial Engineering', es: 'UTFSM, Departamento de Industrias' },
    role: { en: 'Instructor in charge', es: 'Profesor de cátedra' },
    period: '2026-2',
    level: { en: 'Undergraduate', es: 'Pregrado' },
    description: {
      en: 'How to build a numerical model of a thermal system, check that it is right, find which inputs matter most, and optimise the design, from constrained problems to the stochastic gradient methods used in machine learning. Weekly computational laboratories, and a unit on machine learning for models built from data.',
      es: 'Cómo construir un modelo numérico de un sistema térmico, comprobar que sea correcto, identificar qué variables pesan más y optimizar el diseño, desde problemas con restricciones hasta los métodos de gradiente estocástico que se usan en aprendizaje automático. Laboratorios computacionales semanales y una unidad de aprendizaje automático para modelos construidos a partir de datos.',
    },
    page: '/teaching/thermal-systems-design/',
    current: true,
  },
  {
    code: 'ILN222',
    title: { en: 'Energy Management II (Heat and Mass Transfer)', es: 'Gestión Energética II (Transferencia de Calor y Masa)' },
    institution: { en: 'UTFSM, Department of Industrial Engineering', es: 'UTFSM, Departamento de Industrias' },
    role: { en: 'Instructor in charge', es: 'Profesor de cátedra' },
    period: '2026',
    level: { en: 'Undergraduate', es: 'Pregrado' },
    description: {
      en: 'The three modes of heat transfer (conduction, convection and radiation) and mass transfer, taught with short computational demonstrations and a semester-long modelling project.',
      es: 'Los tres modos de transferencia de calor (conducción, convección y radiación) y la transferencia de masa, con demostraciones computacionales breves y un proyecto de modelación durante el semestre.',
    },
    current: true,
  },
  {
    code: 'ILN221',
    title: { en: 'Energy Management I (Fluid Mechanics)', es: 'Gestión Energética I (Mecánica de Fluidos)' },
    institution: { en: 'UTFSM, Department of Industrial Engineering', es: 'UTFSM, Departamento de Industrias' },
    role: { en: 'Instructor in charge', es: 'Profesor de cátedra' },
    period: '2026',
    level: { en: 'Undergraduate', es: 'Pregrado' },
    description: {
      en: 'Fluids at rest and in motion: conservation of mass, momentum and energy, applied to pipe flow, friction losses and the pumping systems found in industry, with interactive demonstrations.',
      es: 'Fluidos en reposo y en movimiento: conservación de masa, cantidad de movimiento y energía, aplicada al flujo en tuberías, las pérdidas por fricción y los sistemas de bombeo de la industria, con demostraciones interactivas.',
    },
    current: true,
  },
  {
    code: 'ICN367',
    title: { en: 'Introduction to Energy Project Management', es: 'Introducción a la Gestión de Proyectos Energéticos' },
    institution: { en: 'UTFSM, Department of Industrial Engineering', es: 'UTFSM, Departamento de Industrias' },
    role: { en: 'Instructor in charge', es: 'Profesor de cátedra' },
    period: '2026-1',
    level: { en: 'Undergraduate', es: 'Pregrado' },
    description: {
      en: 'How an energy project is formulated and evaluated: the available resource and the demand, technical sizing, costs, financial indicators and emissions, with case studies built in RETScreen, a standard software tool for pre-feasibility studies of energy projects.',
      es: 'Cómo se formula y evalúa un proyecto energético: el recurso disponible y la demanda, el dimensionamiento técnico, los costos, los indicadores financieros y las emisiones, con casos de estudio desarrollados en RETScreen, una herramienta de uso habitual para estudios de prefactibilidad de proyectos de energía.',
    },
    current: true,
  },
];

export const pastTeaching = [
  {
    where: 'INSA Rouen Normandie, France',
    period: '2023–2025',
    what: {
      en: 'Laboratory and tutorial classes in electricity, geometrical optics and wave optics (2024–2025), energy and fluids (2023–2024), and supervised student science projects (2023–2025).',
      es: 'Trabajos prácticos y dirigidos de electricidad, óptica geométrica y óptica ondulatoria (2024–2025), energía y fluidos (2023–2024), y proyectos científicos supervisados (2023–2025).',
    },
  },
  {
    where: 'UTFSM, Chile',
    period: '2019–2022',
    what: {
      en: 'Teaching assistant for graduate-level Optimization (2019–2022), Operations Research (2020), Operations Management II (2021), Fluid Mechanics (2019–2022) and Differential Calculus (2019).',
      es: 'Ayudante de Optimización de magíster (2019–2022), Investigación de Operaciones (2020), Gestión de Operaciones II (2021), Mecánica de Fluidos (2019–2022) y Cálculo Diferencial (2019).',
    },
  },
  {
    where: 'Universidad Andrés Bello, Chile',
    period: '2017–2018',
    what: {
      en: 'Teaching assistant for Differential and Integral Calculus.',
      es: 'Ayudante de Cálculo Diferencial y Cálculo Integral.',
    },
  },
];
