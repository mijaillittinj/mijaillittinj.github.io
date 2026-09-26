/**
 * Research software. Only packages whose public release has been verified.
 * pysat-abel: PyPI release 0.1.3 (2025-05-09) checked on 2026-09-24.
 */
export interface Software {
  slug: string;
  name: string;
  summary: string;
  language: string;
  status: 'released' | 'in-development';
  links: { label: string; href: string }[];
  paper?: string; // publication id
  license: string;
  licenseNote?: string;
}

export const software: Software[] = [
  {
    slug: 'pysat-abel',
    name: 'pysat-abel',
    summary: 'Spline-based Abel transform (SAT): regularised reconstruction of radial profiles from line-of-sight measurements of axisymmetric objects, with optional correction of signal trapping (self-absorption).',
    language: 'Python',
    status: 'released',
    links: [
      { label: 'PyPI', href: 'https://pypi.org/project/pysat-abel/' },
      { label: 'Source (CORIA GitLab)', href: 'https://gitlab.coria-cfd.fr/littinm/pysat' },
    ],
    paper: 'littin2024sat',
    license: 'MIT (package metadata)',
    licenseNote: 'The 0.1.3 release also ships a GPL-3.0 licence file; the intended licence is being clarified.',
  },
];

/**
 * Methods described without code or equations. Only pysat-abel is distributed.
 * Published methods cite their paper; the soot–gas separation method is unpublished
 * so it is described in general terms only.
 */
export interface Method {
  id: string;
  name: string;
  summary: string;
  status: string;
  paper?: string; // publication id
  page?: string;  // internal explainer
}

export const methods: Method[] = [
  {
    id: 'pinn-multidiagnostic',
    name: 'Physics-informed inversion of several optical diagnostics',
    summary: 'One neural network takes a position in the flame and returns the local soot properties there. Its loss contains the physics of light scattering by fractal aggregates and thermal emission, so it is trained directly on the measurements, without labelled data. Attenuation, emission and multi-angle scattering at several wavelengths are inverted together, and the calibration factors of the instruments are recovered as part of the solution. Outputs: temperature, volume fraction, aggregate volume, composition and optical-index fields.',
    status: 'Published. Code available upon request.',
    paper: 'littin2026carbon',
    page: '/publications/carbon-2026/',
  },
  {
    id: 'ann-maturity',
    name: 'Neural network for soot maturity in normal and inverse diffusion flames',
    summary: 'A neural network that takes a position in the flame reconstructs soot volume fraction, temperature and an indicator of soot maturity (how far the particles have evolved towards graphitic carbon) from flame emission recorded at two wavelengths. The wavelength dependence of the soot optical properties is built into the model, which is what makes maturity visible in the data. Applied to normal (NDF) and inverse (IDF) diffusion flames.',
    status: 'Published. Code available upon request.',
    paper: 'yap2026dual',
  },
  {
    id: 'aggregate-sizing-2d',
    name: 'Two-dimensional aggregate size distributions from angular light scattering',
    summary: 'Images of light scattered at several angles and of light extinction are converted into local values, with signal trapping taken into account, and interpreted with a scattering model for fractal soot aggregates. The result is a map of the aggregate size distribution across the flame instead of a single line-of-sight average.',
    status: 'Published. Code available upon request.',
    paper: 'littin2025scatt',
  },
  {
    id: 'soot-gas-voxel',
    name: 'Volumetric separation of soot and gas radiation',
    summary: 'A radiometer measures the total radiation of a flame, soot and combustion gases together. This method divides the flame into small volume elements (voxels), estimates the soot emission of each element from locally resolved soot fields retrieved from flame images, and carries every contribution to the detector through the exact geometry of the set-up. The gas contribution is what remains. The method is validated on synthetic flames with known answers before it is applied to experiments.',
    status: 'Unpublished method. Description only.',
  },
];
