/**
 * Canonical publication database. Every page that lists papers reads from here.
 *
 * Sources: verified CV (2026-09-23), Crossref metadata (volumes, DOIs, licences
 * queried 2026-09-24), HAL records for accepted papers. Summaries are written for
 * this site (not publisher abstracts). See PUBLICATION_WORKFLOW.md.
 *
 * `curated` is the hand-written list. `publications` adds what the automatic
 * ORCID/Crossref sync found (publications.auto.json, refreshed weekly by CI):
 * new papers appear without a summary, and accepted/in-press entries are
 * upgraded once Crossref has the DOI or volume. Merge rules: src/lib/pubmerge.ts.
 */
import { mergePublications, type AutoWork } from '../lib/pubmerge.ts';
import autoData from './publications.auto.json';

export type PubStatus = 'published' | 'in-press' | 'accepted' | 'under-review';
export type ThemeId = 'reconstruction' | 'physics-informed' | 'forward-models' | 'uncertainty' | 'experiments';
export type DomainId = 'combustion' | 'radiation' | 'aerosols' | 'fire' | 'fuels' | 'optical-diagnostics';

export interface Publication {
  /** URL slug, also used as BibTeX key. */
  id: string;
  title: string;
  authors: string[];
  year: number;
  journal: string;
  volume?: string;
  article?: string;
  doi?: string;
  status: PubStatus;
  /** Candidate's position: first author, corresponding author. */
  firstAuthor?: boolean;
  corresponding?: boolean;
  /** Uses neural networks / machine learning. */
  ml?: boolean;
  themes: ThemeId[];
  domains: DomainId[];
  summary: string;
  access: {
    /** 'open' = version of record under an open licence. */
    kind: 'open' | 'closed' | 'unknown';
    license?: string;
    /** Only link a PDF when the licence allows it. */
    pdf?: string;
  };
  links?: { label: string; href: string }[];
  /** Internal detail page (e.g. '/publications/carbon-2026/'). */
  page?: string;
  slides?: string;
  featured?: boolean;
  /** Hidden from public lists until the candidate confirms. */
  hidden?: boolean;
}

const ME = 'M. Littin';

const curated: Publication[] = [
  {
    id: 'littin2026carbon',
    title: 'Physics-informed neural network retrieval of spatially-resolved soot properties from multi-wavelength optical diagnostics',
    authors: [ME, 'F. Escudero', 'E. Magaña', 'C. López', 'M. Mazur', 'A. Fuentes', 'J. Yon'],
    year: 2026, journal: 'Carbon', volume: '250', article: '121296',
    doi: '10.1016/j.carbon.2026.121296', status: 'published', firstAuthor: true, ml: true,
    themes: ['physics-informed', 'reconstruction', 'forward-models'],
    domains: ['optical-diagnostics', 'combustion', 'aerosols', 'radiation'],
    summary:
      'Three optical measurements (attenuation, emission and multi-angle scattering at five wavelengths) are inverted jointly by one coordinate-based neural network whose loss contains the light-scattering theory and Planck’s law. The network returns temperature, volume fraction, aggregate size, composition and optical-index fields, together with the calibration factors of the instruments.',
    access: { kind: 'open', license: 'CC BY 4.0', pdf: 'https://doi.org/10.1016/j.carbon.2026.121296' },
    links: [{ label: 'SSRN preprint', href: 'https://doi.org/10.2139/ssrn.5726644' }],
    page: '/publications/carbon-2026/',
    slides: '/talks/carbon-2026/',
    featured: true,
  },
  {
    id: 'yap2026dual',
    title: 'Dual-wavelength neural reconstruction of soot volume fraction, temperature and maturity in inverse and normal diffusion flames',
    authors: ['V. Yap', 'J. Gallardo', ME, 'F. Escudero', 'J. Yon', 'A. Fuentes', 'R. Demarco'],
    year: 2026, journal: 'Proceedings of the Combustion Institute', volume: '42', article: '106371',
    doi: '10.1016/j.proci.2026.106371', status: 'published', ml: true,
    themes: ['physics-informed', 'reconstruction'],
    domains: ['optical-diagnostics', 'combustion'],
    summary:
      'Neural reconstruction of soot volume fraction, temperature and a maturity indicator from measurements at only two wavelengths, applied to inverse and normal diffusion flames.',
    access: { kind: 'closed' },
  },
  {
    id: 'mercado2026view',
    title: 'Machine-learning surrogate models for view factor estimation in benchmark fire configurations',
    authors: ['M. Mercado', 'P. E. Pinto', 'I. Verdugo', ME, 'F. Escudero', 'R. Demarco', 'A. Fuentes'],
    year: 2026, journal: 'International Journal of Heat and Mass Transfer', volume: '269', article: '129097',
    doi: '10.1016/j.ijheatmasstransfer.2026.129097', status: 'published', ml: true,
    themes: ['forward-models'],
    domains: ['fire', 'radiation'],
    summary:
      'Machine-learning surrogates for radiative view factors between benchmark fire geometries and targets, as fast approximations of the geometric integrals.',
    access: { kind: 'closed' },
  },
  {
    id: 'pinto2026spatially',
    title: 'Spatially and temporally resolved soot radiation and temperature in laminar PMMA flames',
    authors: ['P. E. Pinto', ME, 'I. Verdugo', 'G. Severino', 'F. Escudero', 'J. J. Cruz', 'J. L. Urban', 'M. Thomsen', 'A. Fuentes', 'R. Demarco'],
    year: 2026, journal: 'International Journal of Heat and Mass Transfer', article: '129542',
    doi: '10.1016/j.ijheatmasstransfer.2026.129542', status: 'in-press',
    themes: ['experiments', 'reconstruction'],
    domains: ['fire', 'radiation', 'optical-diagnostics'],
    summary: 'Time-resolved measurements of soot temperature and radiation in flames of a burning solid polymer (PMMA).',
    access: { kind: 'closed' },
  },
  {
    id: 'raynaud2026structure',
    title: 'A pair-correlation-based structure factor for fractal agglomerates accounting for primary-particle polydispersity',
    authors: ['Y. Raynaud-Diarra', 'R. Ceolato', 'A. Poux', ME, 'C. Argentin', 'M. Mazur', 'J. Yon'],
    year: 2026, journal: 'Journal of Quantitative Spectroscopy and Radiative Transfer', volume: '364', article: '110072',
    doi: '10.1016/j.jqsrt.2026.110072', status: 'published',
    themes: ['forward-models'],
    domains: ['aerosols', 'radiation'],
    summary: 'A structure-factor model for light and X-ray scattering by fractal agglomerates that accounts for the spread in primary-particle sizes.',
    access: { kind: 'open', license: 'CC BY 4.0', pdf: 'https://doi.org/10.1016/j.jqsrt.2026.110072' },
  },
  {
    id: 'yon2026xray',
    title: 'X-ray to visible light scattering by fractal agglomerates: a model comparison applied to soot',
    authors: ['J. Yon', ME, 'Y. Raynaud-Diarra', 'G. Lefevre', 'M. Mazur', 'M. Sztucki', 'R. Ceolato', 'A. Fuentes'],
    year: 2026, journal: 'Journal of Quantitative Spectroscopy and Radiative Transfer', volume: '354', article: '109860',
    doi: '10.1016/j.jqsrt.2026.109860', status: 'published',
    themes: ['forward-models'],
    domains: ['aerosols', 'radiation'],
    summary: 'Compares scattering models for fractal soot agglomerates from the X-ray to the visible range, which are the forward models used to interpret scattering measurements.',
    access: { kind: 'open', license: 'CC BY 4.0', pdf: 'https://doi.org/10.1016/j.jqsrt.2026.109860' },
  },
  {
    id: 'weiss2026hychem',
    title: 'Assessment of extended HyChem-based gasoline kinetic mechanisms for soot production in coflow diffusion flames',
    authors: ['M. Weiss', ME, 'F. Escudero', 'F. Liu', 'R. Demarco'],
    year: 2026, journal: 'Case Studies in Thermal Engineering', volume: '84', article: '108217',
    doi: '10.1016/j.csite.2026.108217', status: 'published',
    themes: ['forward-models'],
    domains: ['combustion', 'fuels'],
    summary: 'Evaluates extended reaction mechanisms for predicting soot in simulations of vaporised-gasoline laminar flames.',
    access: { kind: 'open', license: 'CC BY-NC 4.0', pdf: 'https://doi.org/10.1016/j.csite.2026.108217' },
  },
  {
    id: 'pinto2026pmma',
    title: 'Sooting behavior of laminar flames produced with clear and black cast PMMA',
    authors: ['P. E. Pinto', ME, 'I. Verdugo', 'G. Severino', 'J. J. Cruz', 'F. Escudero', 'J. L. Urban', 'M. Thomsen', 'A. Fuentes', 'R. Demarco'],
    year: 2026, journal: 'Fire Safety Journal', volume: '163', article: '104875',
    doi: '10.1016/j.firesaf.2026.104875', status: 'published',
    themes: ['experiments'],
    domains: ['fire', 'optical-diagnostics'],
    summary: 'Compares soot production in laminar flames of clear and black cast PMMA, a solid polymer.',
    access: { kind: 'closed' },
  },
  {
    id: 'pinto2026eucalyptus',
    title: 'Soot propensity of Eucalyptus globulus laminar flames at different coflow indices',
    authors: ['P. E. Pinto', 'I. Verdugo', ME, 'G. Severino', 'J. J. Cruz', 'F. Escudero', 'R. Demarco', 'A. Fuentes'],
    year: 2026, journal: 'Fuel', volume: '415', article: '138357',
    doi: '10.1016/j.fuel.2026.138357', status: 'published',
    themes: ['experiments'],
    domains: ['fire', 'combustion'],
    summary: 'Soot measurements in flames of a wildland fuel under different oxidiser coflow conditions.',
    access: { kind: 'closed' },
  },
  {
    id: 'pinto2026fov',
    title: 'Improving radiation estimation from circular pool fires through field-of-view angle analysis',
    authors: ['P. E. Pinto', ME],
    year: 2026, journal: 'International Journal of Heat and Mass Transfer', volume: '258', article: '128312',
    doi: '10.1016/j.ijheatmasstransfer.2025.128312', status: 'published',
    themes: ['forward-models'],
    domains: ['fire', 'radiation'],
    summary: 'Uses the field-of-view angle of a receiver to improve estimates of the thermal radiation it receives from a circular pool fire.',
    access: { kind: 'open', license: 'CC BY 4.0', pdf: 'https://doi.org/10.1016/j.ijheatmasstransfer.2025.128312' },
  },
  {
    id: 'littin2025scatt',
    title: 'In situ determination of 2D spatially-resolved size distribution of soot aggregates in a laminar diffusion flame',
    authors: [ME, 'G. Lefevre', 'M. Mazur', 'A. Fuentes', 'J. Yon'],
    year: 2025, journal: 'Proceedings of the Combustion Institute', volume: '41', article: '105946',
    doi: '10.1016/j.proci.2025.105946', status: 'published', firstAuthor: true,
    themes: ['reconstruction', 'forward-models'],
    domains: ['optical-diagnostics', 'aerosols'],
    summary:
      'Recovers the size distribution of nanoparticle aggregates at every point of an axisymmetric flame from multi-angle, multi-wavelength scattering. The line-of-sight signals are first inverted with trapping correction, then interpreted with two competing size-distribution models.',
    access: { kind: 'open', license: 'CC BY 4.0', pdf: 'https://doi.org/10.1016/j.proci.2025.105946' },
    featured: true,
  },
  {
    id: 'littin2025saxs',
    title: 'Soot primary particle radial profiles in laminar diffusion flames for Jet A-1/SAF fuels: a SAXS study',
    authors: [ME, 'M. Mazur', 'G. Lefevre', 'M. Sztucki', 'A. Fuentes', 'J. Yon'],
    year: 2025, journal: 'Proceedings of the Combustion Institute', volume: '41', article: '105852',
    doi: '10.1016/j.proci.2025.105852', status: 'published', firstAuthor: true,
    themes: ['reconstruction', 'forward-models', 'experiments'],
    domains: ['aerosols', 'fuels', 'combustion'],
    summary:
      'Synchrotron small-angle X-ray scattering, deconvolved with the spline-based Abel transform, gives radial profiles of primary-particle size in flames of jet fuel and sustainable-aviation-fuel blends.',
    access: { kind: 'closed' },
  },
  {
    id: 'lefevre2025oxidation',
    title: 'Experimental investigation of the impact of organic contents and fuel (kerosene and SAF) on soot oxidation',
    authors: ['G. Lefevre', 'G. Godard', ME, 'M. Mazur', 'S. Richard', 'N. Detomaso', 'J. Yon'],
    year: 2025, journal: 'Combustion and Flame', volume: '282', article: '114503',
    doi: '10.1016/j.combustflame.2025.114503', status: 'published',
    themes: ['experiments'],
    domains: ['aerosols', 'fuels'],
    summary: 'Studies how organic content and fuel type change the oxidation of soot particles.',
    access: { kind: 'open', license: 'CC BY 4.0', pdf: 'https://doi.org/10.1016/j.combustflame.2025.114503' },
  },
  {
    id: 'alarcon2025dme',
    title: 'Mechanistic investigation of soot and NOx suppression in iso-carbon propane coflow flames by dimethyl ether substitution',
    authors: ['F. Alarcón', ME, 'F. Escudero', 'I. Verdugo', 'A. Fuentes', 'R. Demarco'],
    year: 2025, journal: 'Journal of the Energy Institute', volume: '123', article: '102302',
    doi: '10.1016/j.joei.2025.102302', status: 'published',
    themes: ['forward-models'],
    domains: ['combustion', 'fuels'],
    summary: 'Investigates the mechanisms by which substituting dimethyl ether for part of a propane fuel suppresses both soot and NOx at constant carbon flow.',
    access: { kind: 'closed' },
  },
  {
    id: 'littin2024sat',
    title: 'Spline-based Abel Transform (SAT) radial property reconstruction for noise and trapping correction: application to axisymmetric sooting flames',
    authors: [ME, 'A. Poux', 'G. Lefevre', 'M. Mazur', 'F. Escudero', 'A. Fuentes', 'J. Yon'],
    year: 2024, journal: 'Fuel', volume: '374', article: '132365',
    doi: '10.1016/j.fuel.2024.132365', status: 'published', firstAuthor: true,
    themes: ['reconstruction'],
    domains: ['optical-diagnostics', 'combustion'],
    summary:
      'Abel inversion recovers a radial field from line-of-sight measurements but amplifies noise near the axis. SAT represents the field as a constrained cubic spline fitted with a curvature penalty and includes the self-absorption (trapping) of the signal inside the forward model. It is released as the Python package pysat-abel.',
    access: { kind: 'closed' },
    links: [{ label: 'pysat-abel', href: '/software/pysat-abel/' }],
    featured: true,
  },
  {
    id: 'littin2024trapping',
    title: 'On the consideration of signal trapping for soot sizing by angular light scattering in laminar flames',
    authors: [ME, 'A. Poux', 'G. Lefevre', 'M. Mazur', 'A. Fuentes', 'J. Yon'],
    year: 2024, journal: 'Journal of Aerosol Science', volume: '181', article: '106429',
    doi: '10.1016/j.jaerosci.2024.106429', status: 'published', firstAuthor: true,
    themes: ['reconstruction', 'forward-models'],
    domains: ['optical-diagnostics', 'aerosols'],
    summary:
      'Shows how attenuation of scattered light along the path to the detector biases aggregate sizes inferred from angular scattering, and how to correct it inside the reconstruction.',
    access: { kind: 'closed' },
  },
  {
    id: 'littin2024jeta1',
    title: 'Understanding soot production in a Jet A-1 laminar coflow non-premixed flame',
    authors: [ME, 'F. Escudero', 'J. J. Cruz', 'I. Verdugo', 'D. Chen', 'A. Fuentes', 'R. Demarco'],
    year: 2024, journal: 'Proceedings of the Combustion Institute', volume: '40', article: '105534',
    doi: '10.1016/j.proci.2024.105534', status: 'published', firstAuthor: true,
    themes: ['forward-models', 'experiments'],
    domains: ['combustion', 'fuels'],
    summary:
      'Combines multi-wavelength attenuation and emission measurements with detailed simulations based on the HyChem mechanism to explain soot formation in a jet-fuel flame.',
    access: { kind: 'closed' },
  },
  {
    id: 'escudero2023water',
    title: 'Sooting properties of laminar coflow non-premixed ethylene/hydrogen flames influenced by water vapor addition to the oxidizer',
    authors: ['F. Escudero', ME, 'R. Demarco', 'F. Liu', 'A. Fuentes'],
    year: 2023, journal: 'Fire Safety Journal', volume: '141', article: '103997',
    doi: '10.1016/j.firesaf.2023.103997', status: 'published',
    themes: ['forward-models'],
    domains: ['combustion', 'fire', 'radiation'],
    summary: 'Simulations separating the dilution, thermal and chemical effects of water vapour on soot and radiation in hydrogen-enriched flames.',
    access: { kind: 'closed' },
  },
  {
    id: 'pinto2023contour',
    title: 'A simpler tractable contour technique to model thermal radiation from buoyant diffusion flames',
    authors: ['P. Pinto', ME, 'J. I. Rivera', 'G. Severino', 'J. J. Cruz', 'A. Fuentes'],
    year: 2023, journal: 'Experimental Thermal and Fluid Science', volume: '149', article: '111027',
    doi: '10.1016/j.expthermflusci.2023.111027', status: 'published',
    themes: ['forward-models'],
    domains: ['fire', 'radiation'],
    summary: 'Estimates the view factor between an unsteady flame and a target from parametric flame contours and Stokes’ theorem.',
    access: { kind: 'closed' },
  },
  // Accepted, no DOI yet (public HAL records exist for the Kyoto 2026 presentations)
  {
    id: 'littin2026saf',
    title: 'Impact of sustainable aviation fuel on soot properties: a physics-informed neural network enhanced in-situ characterization',
    authors: [ME, 'B. Kaźmierski', 'M. Mazur', 'F. Escudero', 'A. Fuentes', 'J. Yon'],
    year: 2026, journal: 'Proceedings of the Combustion Institute', status: 'accepted', firstAuthor: true, ml: true,
    themes: ['physics-informed'],
    domains: ['fuels', 'optical-diagnostics'],
    summary: 'Applies physics-informed neural inversion to compare soot properties in flames of conventional jet fuel and sustainable aviation fuel blends.',
    access: { kind: 'unknown' },
    links: [{ label: 'HAL record', href: 'https://hal.science/hal-05718642' }],
  },
  {
    id: 'gutierrez2026maturity',
    title: 'Soot maturity evolution in a periodically-forced diffusion flame via neural network assisted inversion',
    authors: ['N. Gutiérrez', 'N. Mancilla', 'G. Severino', ME, 'J. Solis Igor', 'A. García', 'F. Escudero', 'J. J. Cruz', 'B. Herrmann', 'J. Yon', 'R. Demarco', 'A. Fuentes'],
    year: 2026, journal: 'Proceedings of the Combustion Institute', status: 'accepted', ml: true,
    themes: ['physics-informed', 'reconstruction'],
    domains: ['combustion', 'optical-diagnostics'],
    summary: 'Neural-network-assisted inversion used to follow the evolution of soot maturity in a periodically forced diffusion flame.',
    access: { kind: 'unknown' },
    links: [{ label: 'HAL record', href: 'https://hal.science/hal-05718633' }],
  },
  // Manuscripts under review are not stored in this public repository (see private/ notes and CONTENT_QUESTIONS Q4).
];

/** Date of the last change detected by the automatic sync. */
export const autoSyncDate: string = autoData.generated;

export const publications: Publication[] = mergePublications(curated, autoData.works as AutoWork[]);

export const theses = [
  {
    id: 'littin2025phd',
    kind: 'PhD thesis',
    title: 'Characterization of soot properties in laminar diffusion flames by coupling optical techniques: application to sustainable aeronautic fuels',
    institution: 'Normandie Université / INSA Rouen Normandie (CORIA, UMR 6614 CNRS)',
    year: 2025,
    doi: '10.70675/0d4a7bfaz5f21z4970zae63zb2bfe79c5a5f',
    note: 'Defended 11 December 2025. Supervision: J. Yon and A. Fuentes.',
  },
  {
    id: 'littin2024msc',
    kind: 'MSc thesis',
    title: 'Investigación detallada de la formación de hollín en una llama de difusión laminar de jet fuel basada en un mecanismo HyChem extendido',
    institution: 'Universidad Técnica Federico Santa María',
    year: 2024,
    doi: '10.71959/eze4-k527',
    note: '',
  },
];

export const themeLabels: Record<ThemeId, { en: string; es: string }> = {
  reconstruction: { en: 'Regularised reconstruction', es: 'Reconstrucción regularizada' },
  'physics-informed': { en: 'Physics-informed learning', es: 'Aprendizaje informado por la física' },
  'forward-models': { en: 'Forward and surrogate models', es: 'Modelos directos y sustitutos' },
  uncertainty: { en: 'Uncertainty and identifiability', es: 'Incertidumbre e identificabilidad' },
  experiments: { en: 'Measurements and applications', es: 'Mediciones y aplicaciones' },
};

export const domainLabels: Record<DomainId, string> = {
  combustion: 'Combustion',
  radiation: 'Thermal radiation',
  aerosols: 'Aerosols and nanoparticles',
  fire: 'Fire',
  fuels: 'Fuels and energy',
  'optical-diagnostics': 'Optical diagnostics',
};

export const statusLabel: Record<PubStatus, string> = {
  published: 'Published',
  'in-press': 'In press',
  accepted: 'Accepted',
  'under-review': 'Under review',
};

export const visiblePublications = publications.filter((p) => !p.hidden);

export function byId(id: string): Publication | undefined {
  return publications.find((p) => p.id === id);
}

export function citation(p: Publication): string {
  const loc = [p.volume, p.article].filter(Boolean).join(', ');
  return `${p.authors.join(', ')} (${p.year}). ${p.title}. ${p.journal}${loc ? ` ${loc}` : ''}.${p.doi ? ` doi:${p.doi}` : ''}`;
}

export function bibtex(p: Publication): string {
  const fields: [string, string | undefined][] = [
    ['title', `{${p.title}}`],
    ['author', p.authors.map((a) => (a === ME ? 'Littin, Mijail' : a)).join(' and ')],
    ['journal', p.journal],
    ['year', String(p.year)],
    ['volume', p.volume],
    ['pages', p.article],
    ['doi', p.doi],
    ['note', p.status === 'accepted' ? 'Accepted for publication' : p.status === 'in-press' ? 'In press' : undefined],
  ];
  const body = fields.filter(([, v]) => v).map(([k, v]) => `  ${k} = {${v}}`).join(',\n');
  return `@article{${p.id},\n${body}\n}`;
}
