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
      en: 'Numerical modelling and simulation, verification and sensitivity analysis, and optimization from constrained problems to stochastic gradient methods, with computational laboratories. A unit on machine learning for data-driven modelling is scheduled for October 2026.',
      es: 'Modelación numérica y simulación, verificación y análisis de sensibilidad, y optimización desde problemas con restricciones hasta métodos de gradiente estocástico, con laboratorios computacionales. En octubre de 2026 se incluye una unidad de aprendizaje automático para modelado basado en datos.',
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
      en: 'Conduction, convection and radiation, taught with small computational demonstrations and a semester modelling project.',
      es: 'Conducción, convección y radiación, con demostraciones computacionales y un proyecto semestral de modelación.',
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
      en: 'Fluid statics and the conservation of mass, momentum and energy, applied to pipe flow, head losses and fluid systems in industry, supported by interactive demonstrations.',
      es: 'Estática de fluidos y conservación de masa, cantidad de movimiento y energía, aplicadas a flujo en tuberías, pérdidas de carga y sistemas de fluidos en la industria, con demostraciones interactivas.',
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
      en: 'How energy projects are formulated and evaluated: energy resource and demand, technical sizing, costs, financial indicators and emissions, with case studies built in RETScreen.',
      es: 'Cómo se formulan y evalúan proyectos energéticos: recurso y demanda energética, dimensionamiento técnico, costos, indicadores financieros y emisiones, con casos de estudio desarrollados en RETScreen.',
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
