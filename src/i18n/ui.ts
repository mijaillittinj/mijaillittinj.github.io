/**
 * Bilingual infrastructure.
 * English is the primary language. Spanish pages live under /es/ (see `translatedRoutes`,
 * detected from the files). Pages without a Spanish version link to the English page.
 */
export const languages = { en: 'English', es: 'Español' } as const;
export type Lang = keyof typeof languages;
export const defaultLang: Lang = 'en';

/**
 * Route keys (English path without base, with trailing slash) that have a Spanish version.
 * Detected automatically from the pages under src/pages/es/, so adding a Spanish page is
 * enough to make the language switch and the Spanish navigation point to it.
 */
const esPages = Object.keys(import.meta.glob('../pages/es/**/index.astro'));
export const translatedRoutes = new Set<string>(
  esPages.map((f) => f.replace('../pages/es', '').replace(/index\.astro$/, '') || '/'),
);

export const ui = {
  en: {
    'site.title': 'Mijail Littin',
    'site.description':
      'Mijail Littin, PhD. Computational methods for inverse problems in physical systems, physics-informed machine learning and scientific computing. Research, publications, teaching and open learning material.',
    'nav.home': 'Home',
    'nav.research': 'Research',
    'nav.publications': 'Publications',
    'nav.teaching': 'Teaching',
    'nav.learn': 'Learn',
    'nav.talks': 'Talks',
    'nav.software': 'Software',
    'nav.cv': 'CV',
    'nav.about': 'About',
    'nav.menu': 'Menu',
    'nav.skip': 'Skip to content',
    'nav.primary': 'Primary',
    'lang.switch': 'Español',
    'lang.switchLabel': 'Ver esta página en español',
    'lang.onlyEn': 'This page is available in English only.',
    'footer.contact': 'Contact',
    'footer.profiles': 'Profiles',
    'footer.colophon':
      'Static site built with Astro. No tracking. Figures are generated from the author’s own data or synthetic examples unless stated otherwise.',
    'footer.updated': 'Last updated',
    'footer.license': 'Text © Mijail Littin, CC BY 4.0 unless noted; code MIT.',
    'common.readMore': 'Read more',
    'common.all': 'All',
    'common.pdf': 'PDF',
    'common.doi': 'DOI',
    'common.slides': 'Slides',
    'common.code': 'Code',
    'common.data': 'Data',
    'common.bibtex': 'BibTeX',
    'common.overview': 'Overview',
  },
  es: {
    'site.title': 'Mijail Littin',
    'site.description':
      'Mijail Littin, PhD. Métodos computacionales para problemas inversos en sistemas físicos, aprendizaje automático informado por la física y computación científica. Investigación, publicaciones, docencia y material abierto de aprendizaje.',
    'nav.home': 'Inicio',
    'nav.research': 'Investigación',
    'nav.publications': 'Publicaciones',
    'nav.teaching': 'Docencia',
    'nav.learn': 'Aprender',
    'nav.talks': 'Charlas',
    'nav.software': 'Software',
    'nav.cv': 'CV',
    'nav.about': 'Perfil',
    'nav.menu': 'Menú',
    'nav.skip': 'Ir al contenido',
    'nav.primary': 'Principal',
    'lang.switch': 'English',
    'lang.switchLabel': 'View this page in English',
    'lang.onlyEn': 'Esta página está disponible solo en inglés.',
    'footer.contact': 'Contacto',
    'footer.profiles': 'Perfiles',
    'footer.colophon':
      'Sitio estático construido con Astro. Sin rastreo. Las figuras se generan con datos propios del autor o con ejemplos sintéticos, salvo indicación.',
    'footer.updated': 'Última actualización',
    'footer.license': 'Texto © Mijail Littin, CC BY 4.0 salvo indicación; código MIT.',
    'common.readMore': 'Leer más',
    'common.all': 'Todo',
    'common.pdf': 'PDF',
    'common.doi': 'DOI',
    'common.slides': 'Diapositivas',
    'common.code': 'Código',
    'common.data': 'Datos',
    'common.bibtex': 'BibTeX',
    'common.overview': 'Resumen',
  },
} as const;

export type UIKey = keyof (typeof ui)['en'];

export function t(lang: Lang, key: UIKey): string {
  return ui[lang][key] ?? ui.en[key];
}
