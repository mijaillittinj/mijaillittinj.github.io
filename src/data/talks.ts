/**
 * Conference contributions and presentations.
 * Source: verified CV (2026-09-23) and HAL records. The CV states that contributions
 * are oral unless marked as poster; who presented each one is not documented, so the
 * site lists author lists and does not claim the candidate as speaker.
 */

export type TalkKind = 'conference' | 'poster' | 'workshop' | 'seminar' | 'teaching' | 'web';
export type TalkTopic = 'research' | 'teaching';

export interface Talk {
  id: string;
  title: string;
  authors: string;
  event: string;
  place?: string;
  year: number;
  date?: string;
  kind: TalkKind;
  topic: TalkTopic;
  invited?: boolean;
  firstAuthor?: boolean;
  doi?: string;
  hal?: string;
  /** Interactive slide deck on this site */
  slides?: string;
  related?: string[]; // publication ids
  abstract?: string;
}

const ME = 'M. Littin';

export const talks: Talk[] = [
  {
    id: 'carbon-2026-web',
    title: 'Inverting several optical measurements at once with a physics-informed neural network',
    authors: ME,
    event: 'Interactive web presentation (prepared for this site, not tied to a specific event)',
    year: 2026,
    date: '2026-09',
    kind: 'web',
    topic: 'research',
    firstAuthor: true,
    slides: '/talks/carbon-2026/',
    related: ['littin2026carbon', 'littin2024sat'],
    abstract:
      'A 15-minute scientific presentation of the Carbon 2026 paper: why separate inversions of attenuation, emission and scattering disagree, how a coordinate network constrained by light-scattering theory reconciles them, and what the reconstructed fields show.',
  },
  { id: 'lipl2026-radiative', title: 'Experimental determination of the radiative power emitted by a laminar diffusion flame by considering the soot optical index spatial and spectral variations', authors: `${ME}, F. Escudero, R. Demarco, M. Mazur, J. Yon`, event: 'Light Interaction with Particles / Laser-Induced Incandescence (LIP-LII 2026)', place: 'Rouen, France', year: 2026, kind: 'conference', topic: 'research', firstAuthor: true },
  { id: 'lipl2026-tirelii', title: 'Physics-based inference of soot properties from time-resolved laser-induced incandescence', authors: `F. Escudero, ${ME}, V. Castro, J. J. Cruz, R. Demarco, A. Fuentes, F. Liu, J. Yon`, event: 'LIP-LII 2026', place: 'Rouen, France', year: 2026, kind: 'conference', topic: 'research' },
  { id: 'lipl2026-oxidation', title: 'Coupled optical diagnostics for in-situ characterization of soot oxidation: application to kerosene (Jet A-1) and sustainable aviation fuel (SAF)', authors: `G. Lefevre, G. Godard, ${ME}, M. Mazur, S. Richard, N. Detomaso, J. Yon`, event: 'LIP-LII 2026', place: 'Rouen, France', year: 2026, kind: 'poster', topic: 'research', hal: 'hal-05692865' },
  { id: 'eac2025-saxs', title: 'Characterization of soot emissions from aviation fuels: a SAXS study of Jet A-1 and SAF combustion aerosols', authors: `${ME}, M. Mazur, G. Lefevre, M. Sztucki, A. Fuentes, J. Yon`, event: 'European Aerosol Conference (EAC 2025)', place: 'Lecce, Italy', year: 2025, kind: 'conference', topic: 'research', firstAuthor: true },
  { id: 'rad25-saxs', title: 'Comparison of SAXS models with the help of pair correlation functions', authors: `${ME}, M. Mazur, G. Lefevre, M. Sztucki, A. Fuentes, J. Yon`, event: '11th International Symposium on Radiative Transfer (RAD-25)', place: 'Kuşadası, Türkiye', year: 2025, kind: 'conference', topic: 'research', firstAuthor: true, doi: '10.1615/rad-25.490' },
  { id: 'els2025-structure', title: 'Structure factor of fractal aggregates based on pair correlation modeling: comparison with current modeling and impacts', authors: `J. Yon, ${ME}, G. Lefevre, Y. Raynaud-Diarra, M. Mazur, A. Fuentes, M. Sztucki, R. Ceolato`, event: '21st Electromagnetic and Light Scattering Conference (ELS XXI)', place: 'Milazzo, Italy', year: 2025, kind: 'conference', topic: 'research' },
  { id: 'mcs2025-sizing', title: 'In situ determination of the spatially-resolved size distribution of soot aggregates in a laminar diffusion flame', authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'Mediterranean Combustion Symposium (MCS 2025)', place: 'Corfu, Greece', year: 2025, kind: 'conference', topic: 'research', firstAuthor: true, related: ['littin2025scatt'] },
  { id: 'mcs2025-moisture', title: 'Assessing the effect of wildland fuel moisture content on the spatially-resolved soot properties in laminar flames', authors: `${ME}, P. Pinto, I. Verdugo, G. Severino, F. Escudero, J. J. Cruz, R. Demarco, A. Fuentes`, event: 'MCS 2025', place: 'Corfu, Greece', year: 2025, kind: 'conference', topic: 'research', firstAuthor: true },
  { id: 'mcs2025-eucalyptus', title: 'Soot propensity of Eucalyptus globulus laminar flames at different coflow indices', authors: `P. Pinto, I. Verdugo, ${ME}, G. Severino, J. J. Cruz, F. Escudero, R. Demarco, A. Fuentes`, event: 'MCS 2025', place: 'Corfu, Greece', year: 2025, kind: 'conference', topic: 'research', related: ['pinto2026eucalyptus'] },
  { id: 'isf2025-saxs', title: 'Soot primary particle radial profiles in laminar diffusion flames for Jet A-1/SAF fuels: a SAXS study', authors: `J. Yon, ${ME}, G. Lefevre, M. Mazur, A. Fuentes, F. Escudero, M. Sztucki`, event: 'International Sooting Flame (ISF) Workshop, online meeting', year: 2025, kind: 'workshop', topic: 'research', hal: 'hal-05037260', related: ['littin2025saxs'] },
  { id: 'ecm2025-saxs', title: 'Soot primary particle radial profiles in laminar diffusion flames for Jet A-1/SAF fuels: a SAXS study', authors: `${ME}, M. Mazur, G. Lefevre, M. Sztucki, A. Fuentes, J. Yon`, event: 'European Combustion Meeting (ECM 2025)', place: 'Edinburgh, UK', year: 2025, kind: 'conference', topic: 'research', firstAuthor: true, related: ['littin2025saxs'] },
  { id: 'cfa2025-sizing', title: 'Détermination in situ de la distribution de taille des agrégats de suie dans une flamme de diffusion laminaire', authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'Congrès Français sur les Aérosols (CFA 2025)', place: 'Paris, France', year: 2025, kind: 'conference', topic: 'research', firstAuthor: true, doi: '10.25576/asfera-cfa2025-43927' },
  { id: 'lip2024-sizing', title: 'Soot sizing in laminar diffusion flames: impact of self absorption', authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'Laser-light and Interaction with Particles (LIP 2024)', place: "Xi'an, China", year: 2024, kind: 'conference', topic: 'research', firstAuthor: true },
  { id: 'symp2024-poster', title: 'Impact of self-absorption on soot sizing by angular light scattering in a laminar axisymmetric diffusion flame', authors: `${ME}, A. Poux, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: '40th International Symposium on Combustion', place: 'Milan, Italy', year: 2024, kind: 'poster', topic: 'research', firstAuthor: true, hal: 'hal-04691699' },
  { id: 'workshop2024-milan', title: 'Soot sizing in laminar diffusion flame by angular light scattering', authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'Workshop on Measurement and Computation of Reacting Flows with Carbon Nanoparticles', place: 'Milan, Italy', year: 2024, kind: 'poster', topic: 'research', firstAuthor: true, hal: 'hal-04691692' },
  { id: 'optique2024', title: 'On the consideration of self-absorption for soot sizing by angular light scattering in laminar flames', authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'OPTIQUE Normandie 2024', place: 'Rouen, France', year: 2024, kind: 'conference', topic: 'research', firstAuthor: true, hal: 'hal-04639572' },
  { id: 'lii2024', title: 'Impact of self-absorption on soot sizing by angular light scattering in a laminar axisymmetric diffusion flame', authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'International Workshop on Laser-Induced Incandescence (LII 2024)', place: 'Muskoka, Canada', year: 2024, kind: 'workshop', topic: 'research', firstAuthor: true },
  { id: 'lii2024-oxidation', title: 'Impact of the soot composition and morphology on its oxidation', authors: `G. Lefevre, ${ME}, M. Mazur, J. Yon`, event: 'LII 2024', place: 'Muskoka, Canada', year: 2024, kind: 'poster', topic: 'research' },
  { id: 'osca2024-cavity', title: 'Cavity-enhanced multi-wavelength light extinction and scattering measurements for drone-based soot particle characterization', authors: `G. Lefevre, ${ME}, J. Yon, M. Mazur`, event: "Optique et Capteurs pour l'Atmosphère (OSCA 2024)", place: 'Toulouse, France', year: 2024, kind: 'conference', topic: 'research' },
  { id: 'osca2024-scattering', title: 'Angular light scattering in an axisymmetric laminar flame: influence of self-absorption on soot sizing', authors: `J. Yon, M. Mazur, ${ME}, G. Lefevre`, event: 'OSCA 2024', place: 'Toulouse, France', year: 2024, kind: 'conference', topic: 'research' },
  { id: 'afvl2024', title: "Diagnostic optique in-situ des particules de suie: de la flamme académique aux foyers aéronautiques", authors: `J. Yon, ${ME}, G. Lefevre, M. Bouvier, M. Mazur`, event: "Journée thématique de l'AFVL", place: 'France', year: 2024, kind: 'seminar', topic: 'research' },
  { id: 'cfa2024-trapping', title: "Correction de l'auto-absorption dans les mesures de diffusion angulaire de la lumière par les particules de suie", authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'Congrès Français sur les Aérosols (CFA 2024)', place: 'Paris, France', year: 2024, kind: 'conference', topic: 'research', firstAuthor: true, doi: '10.25576/asfera-cfa2024-38920' },
  { id: 'cfa2024-bench', title: "Développement d'un banc d'essai pour la caractérisation du processus d'oxydation de nanoparticules de suie", authors: `G. Lefevre, ${ME}, M. Mazur, J. Yon`, event: 'CFA 2024', place: 'Paris, France', year: 2024, kind: 'conference', topic: 'research' },
  { id: 'gdr2023-sat', title: 'Spline-based radial property reconstruction for noise and trapping correction for emission, light scattering and extinction measurements in axisymmetric sooting flames', authors: `${ME}, G. Lefevre, M. Mazur, A. Fuentes, J. Yon`, event: 'GDR SUIE meeting (CNRS)', place: 'Saint-Étienne-du-Rouvray, France', year: 2023, kind: 'seminar', topic: 'research', firstAuthor: true, hal: 'hal-04259081', related: ['littin2024sat'] },
  { id: 'grc2023-sat', title: 'Spline-based Abel Transform (SAT) for radial property reconstruction for noise and trapping correction in axisymmetric sooting flames', authors: `${ME}, M. Mazur, A. Fuentes, J. Yon`, event: 'Gordon Research Conference: Laser Diagnostics in Energy and Combustion Science', place: 'Newry, Maine, USA', year: 2023, kind: 'poster', topic: 'research', firstAuthor: true, related: ['littin2024sat'] },
  { id: 'ecm2023-coalescence', title: 'Effects of soot coalescence process on a non-premixed Jet A-1 coflow flame using a sectional particle dynamics model', authors: `${ME}, R. Demarco, F. Escudero, G. Lefevre, J. Yon, M. Mazur, A. Fuentes`, event: 'European Combustion Meeting (ECM 2023)', place: 'Rouen, France', year: 2023, kind: 'poster', topic: 'research', firstAuthor: true },
  { id: 'ecm2023-coupling', title: 'Coupling extinction and angular static light scattering measurements for the characterization (2D) of soot nanoparticles in a flame', authors: `${ME}, J. Yon, G. Lefevre, A. Fuentes, M. Mazur`, event: 'ECM 2023', place: 'Rouen, France', year: 2023, kind: 'poster', topic: 'research', firstAuthor: true },
  { id: 'ecm2023-uav', title: 'Development of a multi-wavelength optical cavity embeddable on a UAV for the characterization of soot particles in smoke plumes during industrial fires', authors: `G. Lefevre, ${ME}, M. Mazur, J. Yon`, event: 'ECM 2023', place: 'Rouen, France', year: 2023, kind: 'poster', topic: 'research', hal: 'hal-04085436' },
  { id: 'cfa2023-coupling', title: "Couplage de mesures d'extinction et de diffusion angulaire de la lumière pour la caractérisation (2D) de nanoparticules de suie dans une flamme", authors: `G. Lefevre, J. Yon, A. Fuentes, ${ME}, M. Mazur`, event: 'Congrès Français sur les Aérosols (CFA 2023)', place: 'Paris, France', year: 2023, kind: 'conference', topic: 'research', doi: '10.25576/ASFERA-CFA2023-32916' },
];

export const kindLabel: Record<TalkKind, string> = {
  conference: 'Conference',
  poster: 'Poster',
  workshop: 'Workshop',
  seminar: 'Seminar',
  teaching: 'Teaching',
  web: 'Interactive',
};
