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
  /** Spanish texts for /es/software/. */
  es?: { summary: string; license: string; licenseNote?: string; links?: { label: string; href: string }[] };
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
    es: {
      summary: 'Transformada de Abel basada en splines (SAT): reconstrucción regularizada de perfiles radiales a partir de mediciones integradas a lo largo de la línea de visión en objetos axisimétricos, con corrección opcional del atrapamiento de señal (autoabsorción).',
      license: 'MIT (metadatos del paquete)',
      licenseNote: 'La versión 0.1.3 incluye además un archivo de licencia GPL-3.0; se está aclarando cuál es la licencia prevista.',
      links: [
        { label: 'PyPI', href: 'https://pypi.org/project/pysat-abel/' },
        { label: 'Código fuente (GitLab de CORIA)', href: 'https://gitlab.coria-cfd.fr/littinm/pysat' },
      ],
    },
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
  /** Spanish texts for /es/software/. */
  es?: { name: string; summary: string; status: string };
}

export const methods: Method[] = [
  {
    id: 'pinn-multidiagnostic',
    name: 'Physics-informed inversion of several optical diagnostics',
    summary: 'One neural network takes a position in the flame and returns the local soot properties there. Its loss contains the physics of light scattering by fractal aggregates and thermal emission, so it is trained directly on the measurements, without labelled data. Attenuation, emission and multi-angle scattering at several wavelengths are inverted together, and the calibration factors of the instruments are recovered as part of the solution. Outputs: temperature, volume fraction, aggregate volume, composition and optical-index fields.',
    status: 'Published. Code available upon request.',
    paper: 'littin2026carbon',
    page: '/publications/carbon-2026/',
    es: {
      name: 'Inversión informada por la física de varios diagnósticos ópticos',
      summary: 'Una red neuronal recibe una posición en la llama y entrega las propiedades locales del hollín en ese punto. Su función de pérdida contiene la física de la dispersión de luz por agregados fractales y de la emisión térmica, de modo que se entrena directamente con las mediciones, sin datos etiquetados. La atenuación, la emisión y la dispersión en varios ángulos, en varias longitudes de onda, se invierten en conjunto, y los factores de calibración de los instrumentos se obtienen como parte de la solución. Resultados: campos de temperatura, fracción volumétrica, volumen de agregados, composición e índice óptico.',
      status: 'Publicado. Código disponible a solicitud.',
    },
  },
  {
    id: 'ann-maturity',
    name: 'Neural network for soot maturity in normal and inverse diffusion flames',
    summary: 'A neural network that takes a position in the flame reconstructs soot volume fraction, temperature and an indicator of soot maturity (how far the particles have evolved towards graphitic carbon) from flame emission recorded at two wavelengths. The wavelength dependence of the soot optical properties is built into the model, which is what makes maturity visible in the data. Applied to normal (NDF) and inverse (IDF) diffusion flames.',
    status: 'Published. Code available upon request.',
    paper: 'yap2026dual',
    es: {
      name: 'Red neuronal para la madurez del hollín en llamas de difusión normales e inversas',
      summary: 'Una red neuronal que recibe una posición en la llama reconstruye la fracción volumétrica de hollín, la temperatura y un indicador de la madurez del hollín (cuánto han evolucionado las partículas hacia carbono grafítico) a partir de la emisión de la llama registrada en dos longitudes de onda. La dependencia de las propiedades ópticas del hollín con la longitud de onda está incluida en el modelo, y es lo que hace visible la madurez en los datos. Aplicada a llamas de difusión normales (NDF) e inversas (IDF).',
      status: 'Publicado. Código disponible a solicitud.',
    },
  },
  {
    id: 'aggregate-sizing-2d',
    name: 'Two-dimensional aggregate size distributions from angular light scattering',
    summary: 'Images of light scattered at several angles and of light extinction are converted into local values, with signal trapping taken into account, and interpreted with a scattering model for fractal soot aggregates. The result is a map of the aggregate size distribution across the flame instead of a single line-of-sight average.',
    status: 'Published. Code available upon request.',
    paper: 'littin2025scatt',
    es: {
      name: 'Distribuciones bidimensionales del tamaño de agregados a partir de la dispersión angular de la luz',
      summary: 'Imágenes de la luz dispersada en varios ángulos y de la extinción de la luz se convierten en valores locales, considerando el atrapamiento de señal, y se interpretan con un modelo de dispersión para agregados fractales de hollín. El resultado es un mapa de la distribución de tamaños de los agregados en toda la llama, en lugar de un único promedio a lo largo de la línea de visión.',
      status: 'Publicado. Código disponible a solicitud.',
    },
  },
  {
    id: 'soot-gas-voxel',
    name: 'Volumetric separation of soot and gas radiation',
    summary: 'A radiometer measures the total radiation of a flame, soot and combustion gases together. This method divides the flame into small volume elements (voxels), estimates the soot emission of each element from locally resolved soot fields retrieved from flame images, and carries every contribution to the detector through the exact geometry of the set-up. The gas contribution is what remains. The method is validated on synthetic flames with known answers before it is applied to experiments.',
    status: 'Unpublished method. Description only.',
    es: {
      name: 'Separación volumétrica de la radiación del hollín y de los gases',
      summary: 'Un radiómetro mide la radiación total de una llama, del hollín y de los gases de combustión en conjunto. Este método divide la llama en pequeños elementos de volumen (vóxeles), estima la emisión de hollín de cada elemento a partir de campos de hollín resueltos localmente, obtenidos de imágenes de la llama, y lleva cada contribución hasta el detector con la geometría exacta del montaje. La contribución de los gases es lo que queda. El método se valida con llamas sintéticas de respuesta conocida antes de aplicarlo a experimentos.',
      status: 'Método no publicado. Solo descripción.',
    },
  },
];
