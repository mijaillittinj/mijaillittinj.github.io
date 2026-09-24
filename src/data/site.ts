/**
 * Identity and profile links. Only verified identifiers are listed here.
 * Sources: CV (application_uandes/02_cv/academic_cv.tex, verified 2026-09-23), PyPI.
 */
export const person = {
  name: 'Mijail Littin',
  fullName: 'Mijail Farid Littin Jadell',
  honorific: 'PhD',
  // Public contact email. See CONTENT_QUESTIONS.md (institutional address preferred?).
  email: 'mijaillittin@icloud.com',
  orcid: '0000-0002-5577-5060',
  scopus: '58550141200',
  affiliation: {
    en: 'Department of Industrial Engineering, Universidad Técnica Federico Santa María (UTFSM), Chile',
    es: 'Departamento de Industrias, Universidad Técnica Federico Santa María (UTFSM), Chile',
  },
  role: {
    en: 'Adjunct instructor (profesor de cátedra), UTFSM, 2026–',
    es: 'Profesor de cátedra, UTFSM, 2026–',
  },
} as const;

export const profiles = [
  { label: 'ORCID', href: `https://orcid.org/${person.orcid}`, note: person.orcid },
  { label: 'Scopus', href: `https://www.scopus.com/authid/detail.uri?authorId=${person.scopus}`, note: person.scopus },
  { label: 'PyPI', href: 'https://pypi.org/project/pysat-abel/', note: 'pysat-abel' },
  { label: 'theses.fr', href: 'https://theses.fr/2025NORMIR38', note: 'PhD thesis record' },
] as const;

/** Date shown in the footer; bump when content changes. */
export const lastUpdated = '2026-09-24';
