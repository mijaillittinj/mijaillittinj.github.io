/**
 * Language, numbers and messages of the tool (WEB_SPEC §8). Spanish first, English a faithful
 * translation, the rules of §9 of the brief in both (no colons, semicolons or dashes in the prose, no
 * adversatives, messages say what happened and what to do, no programming jargon). Decimal comma in
 * Spanish. Codes come from the Worker and the core, the sentence is written here.
 */
export type Lang = 'es' | 'en';
let lang: Lang = 'es';
export const getLang = () => lang;
export const setLang = (l: Lang) => { lang = l; };
export const L = (es: string, en: string) => (lang === 'es' ? es : en);

export const CH_NAME = { es: ['rojo', 'verde', 'azul'], en: ['red', 'green', 'blue'] };
export const CH_LETTER = ['R', 'G', 'B'] as const;
export const chName = (c: number) => CH_NAME[lang][c];

/** Decimal comma in Spanish, thin space every three digits from 10 000 on, a true minus sign. */
export function num(x: number, digits = 0): string {
  if (!Number.isFinite(x)) return '–';
  let s = x.toFixed(digits);
  const [i, f] = s.split('.');
  const neg = i.startsWith('-');
  let int = neg ? i.slice(1) : i;
  if (int.length > 4) int = int.replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  s = (neg && Number(s) !== 0 ? '−' : '') + int + (f ? (lang === 'es' ? ',' : '.') + f : '');
  return s;
}
/** Significant digits, with × 10ⁿ for very large or small values. */
export function sig(x: number, d = 3): string {
  if (!Number.isFinite(x)) return '–';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e >= 5 || e <= -3) {
    const m = x / 10 ** e;
    const sup = String(e).replace(/-/g, '⁻').replace(/\d/g, (c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]);
    return `${num(m, d - 1)} × 10${sup}`;
  }
  return num(x, Math.max(0, d - 1 - e));
}
export const pct = (x: number, digits = 1) => `${num(100 * x, digits)} %`;
export const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));

type T2 = { es: string; en: string };

/** WEB_SPEC §8, rewritten for the redesign (DESIGN.md §7). What happened, then what to do, in one or two sentences. */
const MESSAGES: Record<string, T2> = {
  not_tiff: { es: 'Este archivo no es un TIFF. Exporten la imagen desde pylon como TIFF de 16 bits.', en: 'This file is not a TIFF. Export the image from pylon as a 16-bit TIFF.' },
  bigtiff: { es: 'El archivo es un BigTIFF y la herramienta no lo lee. Guárdenlo como TIFF normal desde pylon.', en: 'The file is a BigTIFF, which the tool cannot read. Save it from pylon as a standard TIFF.' },
  compression: { es: 'El archivo usa una compresión que la herramienta no lee. Guárdenlo sin compresión o con LZW.', en: 'The file uses a compression the tool cannot read. Save it uncompressed or with LZW.' },
  predictor: { es: 'El archivo usa un predictor que la herramienta no lee. Guárdenlo sin compresión desde pylon.', en: 'The file uses a predictor the tool cannot read. Save it uncompressed from pylon.' },
  sample_type: { es: 'El tipo de dato de la imagen no se puede leer. Guárdenla desde pylon como TIFF de 16 bits.', en: 'The image data type cannot be read. Save it from pylon as a 16-bit TIFF.' },
  samples: { es: 'La imagen no tiene uno ni tres canales. Guárdenla desde pylon como TIFF RGB de 16 bits.', en: 'The image has neither one nor three channels. Save it from pylon as a 16-bit RGB TIFF.' },
  truncated: { es: 'El archivo está incompleto. Cópienlo de nuevo desde el computador de la cámara.', en: 'The file is incomplete. Copy it again from the camera computer.' },
  u8_gamma: { es: 'La imagen tiene 8 bits y casi seguro trae gamma. Usen el TIFF de 12 o 16 bits sin gamma ni balance de blancos.', en: 'The image has 8 bits and almost certainly has gamma applied. Use the 12- or 16-bit TIFF without gamma or white balance.' },
  float_nosat: { es: 'La imagen está en punto flotante y no se puede saber dónde satura. Revisen la exposición en pylon.', en: 'The image is floating point, so saturation cannot be detected. Check the exposure in pylon.' },
  rewritten: { es: 'Otro programa guardó de nuevo este archivo ({program}). Revisen que no esté girado ni convertido, o copien el original de pylon.', en: 'Another program saved this file again ({program}). Check that it was not rotated or converted, or copy the original from pylon.' },
  rewritten_example: { es: 'Los archivos del ejemplo son recortes, así que este aviso es esperado. Con sus propias fotos sí importa.', en: 'The example files are crops, so this notice is expected. With your own photos it does matter.' },
  name_unread: { es: 'El nombre {file} no trae la temperatura ni el tiempo de exposición. Escríbanlos en la tabla de archivos.', en: 'The name {file} carries neither the temperature nor the exposure time. Type them in the file table.' },
  name_unread_T: { es: 'El nombre {file} no trae la temperatura del horno. Escríbanla en la tabla de archivos o usen una carpeta como horno/1000C/.', en: 'The name {file} does not carry the furnace temperature. Type it in the file table or use a folder such as horno/1000C/.' },
  name_unread_texp: { es: 'El nombre {file} no trae el tiempo de exposición. Escríbanlo en la tabla de archivos o nombren el archivo como 500exp.tiff.', en: 'The name {file} does not carry the exposure time. Type it in the file table or name the file like 500exp.tiff.' },
  name_unread_flame: { es: 'El nombre {file} no trae el tiempo de exposición. Escríbanlo en el campo Tiempo de exposición.', en: 'The name {file} does not carry the exposure time. Type it in the Exposure time field.' },
  saturated: { es: 'Hay píxeles saturados en {channel}. Esas imágenes quedan fuera. Si siguen en el laboratorio, repítanlas con menos tiempo de exposición.', en: 'There are saturated pixels in {channel}. Those images are left out. If you are still in the lab, retake them with a shorter exposure time.' },
  hot_pixels: { es: 'Hay {n} píxeles calientes del sensor, {a} %. Se reemplazan por la mediana de sus vecinos. Bajo 0,1 % es normal.', en: 'The sensor has {n} hot pixels, {a} %. They are replaced by the median of their neighbours. Below 0.1 % is normal.' },
  below_noise: { es: 'En la zona plana el canal {channel} queda bajo el ruido. Alarguen el tiempo de exposición.', en: 'In the flat zone the {channel} channel is below the noise. Use a longer exposure time.' },
  plateau_not_flat: { es: 'La zona plana de {T} varía {a} % de borde a centro. Hagan clic en el centro de la abertura en la imagen o achiquen el radio.', en: 'The flat zone at {T} varies by {a} % from edge to centre. Click the centre of the aperture in the image or make the radius smaller.' },
  plateau_outside: { es: 'La zona plana de {T} no queda dentro de la abertura. Hagan clic en el centro de la abertura en la imagen.', en: 'The flat zone at {T} is not inside the aperture. Click the centre of the aperture in the image.' },
  plateau_off_centre: { es: 'El centro de la zona plana está a {a} píxeles del centro de la abertura. Si no hay un motivo, vuelvan a la propuesta.', en: 'The flat zone centre is {a} pixels from the aperture centre. Unless there is a reason, go back to the proposal.' },
  plateau_moved: { es: 'El horno o la cámara se movieron {a} píxeles entre series. Si siguen en el laboratorio, fíjenlos y repitan.', en: 'The furnace or camera moved {a} pixels between series. If you are still in the lab, fix them and retake.' },
  dark_estimated: { es: 'No hay oscuros. La calibración usa la pendiente de la señal con el tiempo de exposición, que no depende del oscuro.', en: 'There are no darks. The calibration uses the slope of the signal against exposure time, which does not depend on the dark.' },
  dark_disagree: { es: 'Las dos estimaciones del oscuro difieren en el canal {channel}. Tomen un oscuro con el lente tapado.', en: 'The two dark estimates disagree in the {channel} channel. Take a dark with the lens covered.' },
  bg_texp: { es: 'El fondo {file} tiene {a} µs y la llama {b} µs. Usen un fondo con el mismo tiempo de exposición, o pongan ignorar en su rol.', en: 'The background {file} has {a} µs and the flame {b} µs. Use a background with the same exposure time, or set its role to ignore.' },
  dark_texp: { es: 'El oscuro tiene {a} µs y la llama {b} µs. Usen un oscuro con el mismo tiempo de exposición.', en: 'The dark has {a} µs and the flame {b} µs. Use a dark with the same exposure time.' },
  short_exposure: { es: 'El tiempo de {a} µs es menor que 100 µs y el obturador pierde exactitud. Esa imagen queda fuera.', en: 'The {a} µs exposure is shorter than 100 µs, where the shutter is not accurate. That image is left out.' },
  nonlinear: { es: 'La señal del canal {channel} se aparta {a} % de la recta. Revisen las imágenes sobre el 90 % de la escala.', en: 'The {channel} channel deviates by {a} % from the line. Check the images above 90 % of full scale.' },
  cal_one_channel: { es: 'Solo el canal {channel} quedó calibrado y la temperatura necesita dos. Repitan el horno con el canal más brillante cerca del 60 % de la escala.', en: 'Only the {channel} channel is calibrated and temperature needs two. Repeat the furnace with the brightest channel near 60 % of full scale.' },
  sensor_mismatch: { es: 'La calibración es de un sensor de {a} y la llama de {b}. Calibren con la misma cámara. Mientras tanto la llama se calcula sin calibración.', en: 'The calibration comes from a {a} sensor and the flame from {b}. Calibrate with the same camera. Until then the flame is computed without calibration.' },
  method_mismatch: { es: 'La calibración usó {a} y la llama {b}. Usen el mismo formato de pylon en las dos.', en: 'The calibration used {a} and the flame {b}. Use the same pylon format for both.' },
  series_mismatch: { es: 'Las imágenes de esta serie no tienen el mismo tamaño. Pongan ignorar en el rol de las que sobran.', en: 'The images in this series do not share one size. Set the role of the extra ones to ignore.' },
  texp_unconfirmed: { es: 'Confirmen que el tiempo de exposición es el de la toma. Sin eso f_s queda provisional.', en: 'Confirm that the exposure time matches the shot. Without it f_s stays provisional.' },
  scale_missing: { es: 'Falta la escala. Hagan clic en dos marcas de la regla o escriban el valor.', en: 'The scale is missing. Click two marks on the ruler or type the value.' },
  fs_provisional: { es: 'f_s es provisional. Confirmen el tiempo de exposición y la escala en el paso 2.', en: 'f_s is provisional. Confirm the exposure time and the scale in step 2.' },
  fs_provisional_cal: { es: 'f_s es provisional porque la calibración no se pudo probar. Carguen dos temperaturas o más en el paso 1.', en: 'f_s is provisional because the calibration could not be tested. Load two temperatures or more in step 1.' },
  tilt_warn_1: { es: 'Un cuadro del promedio está algo inclinado. Mírenlo en Opciones antes de seguir.', en: 'One frame in the average is slightly tilted. Look at it in Options before going on.' },
  tilted_1: { es: 'Quedó fuera 1 de {m} cuadros porque la llama estaba inclinada. Pueden incluirlo de nuevo en Opciones.', en: '1 of {m} frames was left out because the flame was tilted. You can include it again in Options.' },
  tilted: { es: 'Quedaron fuera {n} de {m} cuadros porque la llama estaba inclinada. Pueden incluirlos de nuevo en Opciones.', en: '{n} of {m} frames were left out because the flame was tilted. You can include them again in Options.' },
  tilt_warn: { es: '{n} cuadros del promedio están algo inclinados. Mírenlos en Opciones antes de seguir.', en: '{n} frames in the average are slightly tilted. Look at them in Options before going on.' },
  axis_spread: { es: 'El eje se mueve {a} píxeles entre cuadros y quedaron fuera {n}. Pueden incluirlos de nuevo en Opciones.', en: 'The axis moves {a} pixels between frames and {n} were left out. You can include them again in Options.' },
  times_in_names: { es: 'Los nombres traen tiempos de adquisición. Indiquen en Opciones si son repeticiones o instantes distintos.', en: 'The names carry acquisition times. Say in Options whether they are repeats or different instants.' },
  axis_edge: { es: 'El eje quedó en el borde de la búsqueda. Amplíen el semiancho o fijen el eje a mano en Opciones.', en: 'The axis is at the edge of the search. Widen the half-width or set the axis by hand in Options.' },
  axis_ambiguous: { es: 'Hay dos posiciones del eje casi igual de buenas. Fijen el eje a mano en Opciones.', en: 'Two axis positions are almost equally good. Set the axis by hand in Options.' },
  axis_low_signal: { es: 'La señal es baja para ubicar el eje. Revisen el fondo o fijen el eje a mano en Opciones.', en: 'The signal is low for finding the axis. Check the background or set the axis by hand in Options.' },
  axis_search_reduced: { es: 'La búsqueda del eje se achicó para que quepa en la imagen. No hay que hacer nada.', en: 'The axis search was narrowed to fit the image. There is nothing to do.' },
  axis_none: { es: 'No se encontró el eje porque casi no hay señal. Revisen el giro y el fondo, o fijen el eje a mano en Opciones.', en: 'No axis was found because there is almost no signal. Check the rotation and the background, or set the axis by hand in Options.' },
  crop_outside: { es: 'El recorte se sale de la imagen. Reduzcan el semiancho o la altura en Opciones.', en: 'The crop goes outside the image. Reduce the half-width or the height in Options.' },
  edge_signal: { es: 'En el borde del recorte la señal es {a} % del pico. Amplíen los radios por lado en Opciones.', en: 'At the edge of the crop the signal is {a} % of the peak. Widen the radii per side in Options.' },
  edge_signal_file: { es: 'En el borde de la foto la señal es {a} % del pico. En el próximo encuadre dejen más margen a los lados de la llama.', en: 'At the edge of the photo the signal is {a} % of the peak. Next time, leave more margin on each side of the flame.' },
  fine_rotation: { es: 'La rotación fina interpola y suaviza un poco la imagen. Úsenla solo si la llama está inclinada.', en: 'Fine rotation interpolates and smooths the image a little. Use it only if the flame is tilted.' },
  chemilum: { es: 'En la base de la llama hay luz azul de más, probablemente quimioluminiscencia. Esa zona no entra en el azul.', en: 'Near the burner there is extra blue light, probably chemiluminescence. The blue channel is not used there.' },
  uncalibrated: { es: 'Sin calibración la temperatura es solo orientativa y no se calcula f_s.', en: 'Without a calibration the temperature is only indicative and f_s is not computed.' },
  no_ir_cut: { es: 'Sin filtro infrarrojo la razón casi no cambia con la temperatura. Marquen el corte infrarrojo.', en: 'Without the infrared filter the ratio hardly changes with temperature. Tick the infrared cut.' },
  out_of_memory: { es: 'El navegador se quedó sin memoria. Carguen menos cuadros o cierren otras pestañas.', en: 'The browser ran out of memory. Load fewer frames or close other tabs.' },
  zip_unsupported: { es: 'Este navegador no abre archivos .zip. Descompriman la carpeta y elíjanla.', en: 'This browser cannot open .zip files. Unzip the folder and choose it.' },
  zip_bad: { es: 'El .zip está dañado o es muy grande. Descompriman la carpeta y elíjanla.', en: 'The .zip is damaged or too large. Unzip the folder and choose it.' },
  phone: { es: 'Esta herramienta necesita un computador. En un teléfono la memoria no alcanza.', en: 'This tool needs a computer. A phone does not have enough memory.' },
  browser_missing: { es: 'Este navegador no puede correr la herramienta. Usen Chrome, Edge o Firefox recientes.', en: 'This browser cannot run the tool. Use a recent Chrome, Edge or Firefox.' },
  annabel_slow: { es: 'En este computador la red neuronal tardaría cerca de {a} minutos. Bajen las iteraciones o usen otro método.', en: 'On this computer the neural network would take about {a} minutes. Lower the iterations or use another method.' },
  annabel_stopped: { es: 'La red neuronal se detuvo antes de terminar. Se usa el mejor resultado, de la iteración {n}.', en: 'The neural network stopped before the end. The best result, from iteration {n}, is used.' },
  webgpu_lost: { es: 'La tarjeta gráfica dejó de responder y la red neuronal se detuvo. Vuelvan a intentar o usen otro método.', en: 'The graphics card stopped responding and the neural network stopped. Try again or use another method.' },
  worker_failed: { es: 'El cálculo se detuvo. Descarguen el registro en Opciones del paso 5 y envíenlo al profesor.', en: 'The calculation stopped. Download the log in step 5 Options and send it to the instructor.' },
  net_slow: { es: 'En este computador la red neuronal tardaría cerca de {a} minutos. Bajen las iteraciones o usen la razón rojo/verde.', en: 'On this computer the neural network would take about {a} minutes. Lower the iterations or use the red/green ratio.' },
  net_stopped: { es: 'La red neuronal se detuvo antes de terminar. Se usa el mejor resultado, de la iteración {n}.', en: 'The neural network stopped before the end. The best result, from iteration {n}, is used.' },
  net_xcheck_bad: { es: 'La red no coincide con el cálculo directo en los píxeles con más señal. {a} Suban las iteraciones o usen la razón rojo/verde.', en: 'The network does not match the direct calculation on the pixels with most signal. {a} Raise the iterations or use the red/green ratio.' },
  net_bounds_T: { es: 'Algunas temperaturas quedan en el límite de la tabla. Son límites del ajuste y no mediciones.', en: 'Some temperatures sit at the limit of the table. They are limits of the fit, not measurements.' },
  net_bounds_fs: { es: 'f_s llega a su límite en algunos puntos. Ese límite es de la red y no una medición.', en: 'f_s reaches its limit at some points. That limit belongs to the network, not to the measurement.' },
  net_two_channels: { es: 'La red necesita dos canales calibrados. Usen la razón rojo/verde o calibren otro canal.', en: 'The network needs two calibrated channels. Use the red/green ratio or calibrate another channel.' },
  net_no_signal: { es: 'Ningún píxel tiene señal suficiente para la red. Revisen la inversión del paso 3.', en: 'No pixel has enough signal for the network. Check the inversion of step 3.' },
  dark_clipped: { es: 'Hay píxeles en cero en el oscuro. En la próxima sesión suban el nivel de negro de la cámara.', en: 'Some dark pixels are at zero. Next session, raise the camera black level.' },
  dark_clipped_nodark: { es: 'Hay píxeles en cero fuera de la abertura. En la próxima sesión suban el nivel de negro de la cámara.', en: 'Some pixels outside the aperture are at zero. Next session, raise the camera black level.' },
  stray_light: { es: 'Fuera de la abertura hay luz difusa, {a} de la zona plana. Bajo 1 % no cambia nada.', en: 'There is stray light outside the aperture, {a} of the flat zone. Below 1 % it changes nothing.' },
  oven_fit_missing: { es: 'Falta la temperatura o el tiempo de exposición de {n} archivos. Escríbanlo en la tabla.', en: '{n} files lack their temperature or exposure time. Type it in the table.' },
  chem_off_uncal: { es: 'Sin calibración no se busca quimioluminiscencia con el azul.', en: 'Without a calibration chemiluminescence is not searched for with blue.' },
  burner_needed: { es: 'Confirmen que el quemador queda abajo en la imagen orientada. Si no, usen «Girar».', en: 'Confirm that the burner is at the bottom of the oriented image. If not, use «Rotate».' },
  fs_outliers: { es: 'Algunos píxeles cerca del eje dan f_s muy alta. Lean la mediana y el percentil 95, o suban el Suavizado en el paso 3.', en: 'Some pixels near the axis give a very high f_s. Read the median and the 95th percentile, or raise the Smoothing in step 3.' },
  lcurve_edge: { es: 'No se encontró un suavizado automático claro. Súbanlo en Opciones hasta que el ruido cerca del eje desaparezca.', en: 'No clear automatic smoothing was found. Raise it in Options until the noise near the axis disappears.' },
  texp_typed: { es: 'Los nombres no traen el tiempo de exposición. Se usa el escrito, {a} µs.', en: 'The names do not carry the exposure time. The typed one, {a} µs, is used.' },
  upside_down: { es: 'Confirmen que el quemador queda abajo en la imagen orientada. Si no, usen «Girar».', en: 'Confirm that the burner is at the bottom of the oriented image. If not, use «Rotate».' },
  example_preset: { es: 'El ejemplo llenó el tiempo de exposición, la escala, el giro y la altura del quemador.', en: 'The example filled in the exposure time, the scale, the rotation and the burner height.' },
  implausible: { es: 'T o f_s salen altos para una llama con hollín. Revisen el giro, el quemador y la escala del paso 2.', en: 'T or f_s come out high for a sooting flame. Check the rotation, the burner and the scale in step 2.' },
  cal_untested: { es: 'Con una sola temperatura no se puede probar la calibración. Carguen al menos dos.', en: 'With a single temperature the calibration cannot be tested. Load at least two.' },
  cal_inconsistent: { es: 'La calibración no pasa la prueba y no se calcula f_s.', en: 'The calibration does not pass the test and f_s is not computed.' },
  ratio_test_fail: { es: 'No pasa. La cámara se aleja hasta {a} K del termopar. Carguen las cuatro temperaturas y dejen activa «Corregir la respuesta sobre 700 nm» en Opciones.', en: 'Does not pass. The camera is up to {a} K off the thermocouple. Load all four temperatures and keep «Correct the response above 700 nm» on in Options.' },
  ratio_test_fail_all: { es: 'No pasa. La cámara se aleja hasta {a} K del termopar. Revisen la zona plana y la emisividad en Opciones.', en: 'Does not pass. The camera is up to {a} K off the thermocouple. Check the flat zone and the emissivity in Options.' },
  ir_tail: { es: 'Más del 10 % de la señal viene de sobre 700 nm en {a}. La corrección sobre 700 nm se encarga de eso.', en: 'More than 10 % of the signal comes from above 700 nm in {a}. The correction above 700 nm takes care of it.' },
  ir_tail_ok: { es: 'Más del 10 % de la señal viene de sobre 700 nm en {a}. Con estas imágenes no se nota.', en: 'More than 10 % of the signal comes from above 700 nm in {a}. It has no visible effect on these images.' },
  tail_off: { es: 'La corrección sobre 700 nm está apagada.', en: 'The correction above 700 nm is off.' },
  tail_three: { es: 'Con tres temperaturas la prueba es menos exigente. Carguen las cuatro.', en: 'With three temperatures the test is less demanding. Load all four.' },
  tail_few: { es: 'La corrección sobre 700 nm necesita tres temperaturas o más. Carguen las cuatro.', en: 'The correction above 700 nm needs three temperatures or more. Load all four.' },
  tail_fail: { es: 'La corrección sobre 700 nm no pasa la prueba, queda a {a} % con el criterio de 1 %. Revisen la zona plana y la emisividad.', en: 'The correction above 700 nm does not pass the test, it is {a} % off with a 1 % criterion. Check the flat zone and the emissivity.' },
  tail_ok: { es: 'La corrección sobre 700 nm pasa la prueba.', en: 'The correction above 700 nm passes the test.' },
  bayer_order: { es: 'En el horno los canales no van rojo, verde y azul de mayor a menor. Revisen el orden de los canales en pylon.', en: 'At the furnace the channels do not go red, green and blue from high to low. Check the channel order in pylon.' },
  preset_mismatch: { es: 'Los archivos son de otra cámara que la curva elegida. Cambien la cámara en Opciones del paso 1.', en: 'The files come from another camera than the chosen curve. Change the camera in step 1 Options.' },
  encoding_mismatch: { es: 'La calibración y la llama tienen codificaciones distintas ({a} y {b}). Usen la misma configuración de pylon.', en: 'The calibration and the flame use different encodings ({a} and {b}). Use the same pylon settings.' },
  transposed: { es: 'La imagen está girada respecto de la calibración. Elijan la posición del sensor en Opciones.', en: 'The image is rotated relative to the calibration. Choose the sensor orientation in Options.' },
  texp_outside_cal: { es: 'El tiempo de exposición de la llama está fuera del rango del horno. Repitan una foto del horno con ese tiempo.', en: 'The flame exposure time is outside the furnace range. Retake a furnace photo with that time.' },
  rows_saturated: { es: '{n} alturas están saturadas y quedan fuera. En la próxima toma bajen el tiempo de exposición.', en: '{n} heights are saturated and left out. Lower the exposure time in the next shot.' },
  rows_saturated_1: { es: 'Una altura está saturada y queda fuera. En la próxima toma bajen el tiempo de exposición.', en: 'One height is saturated and left out. Lower the exposure time in the next shot.' },
  flat_near_ceiling: { es: 'En el campo plano el canal {channel} tiene {a} % del disco sobre el 90 % de la escala. Bajen su tiempo de exposición.', en: 'In the flat field the {channel} channel has {a} % of the disc above 90 % of full scale. Shorten its exposure time.' },
  ratio_ambiguous: { es: 'En {n} píxeles la razón da dos temperaturas posibles. Quedan fuera del mapa.', en: 'In {n} pixels the ratio gives two possible temperatures. They are left out of the map.' },
  ratio_outside: { es: 'En {n} píxeles la razón queda fuera de la tabla de 1000 a 2600 K y quedan fuera del mapa.', en: 'In {n} pixels the ratio falls outside the 1000 to 2600 K table and they are left out of the map.' },
  lambda_out_of_range: { es: 'El Suavizado quedó fuera del rango. Vuelvan al automático.', en: 'The Smoothing is out of range. Go back to automatic.' },
  over_regularised: { es: 'Entre {a} y {b} mm la reconstrucción se separa de la imagen. Bajen el Suavizado en Opciones y vuelvan a invertir.', en: 'Between {a} and {b} mm the reconstruction departs from the image. Lower the Smoothing in Options and invert again.' },
  over_regularised_1: { es: 'A {a} mm la reconstrucción se separa de la imagen. Bajen el Suavizado en Opciones y vuelvan a invertir.', en: 'At {a} mm the reconstruction departs from the image. Lower the Smoothing in Options and invert again.' },
  over_regularised_corner: { es: 'Entre {a} y {b} mm la reconstrucción se separa algo de la imagen con el suavizado automático. Si la diferencia cambia de signo de un radio al siguiente, es ruido y se puede seguir.', en: 'Between {a} and {b} mm the reconstruction departs a little from the image with automatic smoothing. If the difference changes sign from one radius to the next, it is noise and you can go on.' },
  over_regularised_corner_1: { es: 'A {a} mm la reconstrucción se separa algo de la imagen con el suavizado automático. Si la diferencia cambia de signo de un radio al siguiente, es ruido y se puede seguir.', en: 'At {a} mm the reconstruction departs a little from the image with automatic smoothing. If the difference changes sign from one radius to the next, it is noise and you can go on.' },
  frames_mixed_size_1: { es: 'Un cuadro tiene otro tamaño que la serie de {a} píxeles y quedó fuera. Si es de la serie, guárdenlo con el mismo recorte.', en: 'One frame has another size than the {a} pixel series and was left out. If it belongs to the series, save it with the same crop.' },
  frames_mixed_size: { es: '{n} cuadros tienen otro tamaño que la serie de {a} píxeles y quedaron fuera. Si son de la serie, guárdenlos con el mismo recorte.', en: '{n} frames have another size than the {a} pixel series and were left out. If they belong to the series, save them with the same crop.' },
  implausible_T: { es: 'T o f_s salen altos para una llama con hollín. Revisen el giro, el quemador y la escala del paso 2.', en: 'T or f_s come out high for a sooting flame. Check the rotation, the burner and the scale in step 2.' },
  fs_negative: { es: 'f_s sale negativa en {n} píxeles. Revisen que el fondo sea del mismo tiempo de exposición.', en: 'f_s comes out negative in {n} pixels. Check that the background has the same exposure time.' },
  T_at_grid_edge: { es: 'En {n} píxeles la temperatura quedó en el borde de la tabla y quedan fuera del mapa.', en: 'In {n} pixels the temperature sits at the edge of the table and they are left out of the map.' },
  bg_estimated: { es: 'No hay fondo. Se estimó con los bordes de la imagen. En la próxima sesión tomen un fondo con la llama apagada.', en: 'There is no background. It was estimated from the image edges. Next session, take a background with the flame off.' },
  bg_darkline: { es: 'No hay fondo. Se restó el oscuro de la calibración.', en: 'There is no background. The dark of the calibration was subtracted.' },
  gtab_mismatch: { es: 'La tabla del archivo no coincide con su curva de cámara. Se usa la tabla recalculada. Revisen que el archivo sea de esta cámara.', en: 'The table of the file does not match its camera curve. The recomputed table is used. Check that the file belongs to this camera.' },
  cal_bad: { es: 'Este archivo no es una calibración de esta herramienta. Calibren con el horno.', en: 'This file is not a calibration from this tool. Calibrate with the furnace.' },
  cal_basis: { es: 'La calibración cargada lee la curva {a} y la herramienta {b}. Se usa la del archivo.', en: 'The loaded calibration reads the curve {a} and the tool {b}. The file one is used.' },
  run_bad: { es: 'El archivo no es un registro de esta herramienta. Elijan un run.json descargado en el paso 5.', en: 'The file is not a log of this tool. Choose a run.json downloaded in step 5.' },
  run_loaded: { es: 'Se cargaron los ajustes de {file}. Carguen los mismos archivos y la herramienta comprobará que sean idénticos.', en: 'The settings of {file} were loaded. Load the same files and the tool will check that they are identical.' },
  run_sha_ok: { es: 'Los {n} archivos cargados coinciden con el registro.', en: 'The {n} loaded files match the log.' },
  run_sha_mismatch: { es: '{n} archivos no coinciden con el registro y el cálculo no será el mismo. Carguen los originales.', en: '{n} files do not match the log and the calculation will not be the same. Load the originals.' },
  example_missing: { es: 'Este ejemplo todavía no está publicado. Usen la copia del pendrive o sus propias imágenes.', en: 'This example is not published yet. Use the copy on the USB stick or your own images.' },
  example_failed: { es: 'No se pudo bajar el ejemplo. Revisen la red o carguen la copia del pendrive.', en: 'The example could not be downloaded. Check the network or load the copy on the USB stick.' },
  need_stage: { es: 'Falta un paso anterior. Hagan primero el paso {a}.', en: 'A previous step is missing. Do step {a} first.' },
  kernel_nonpositive: { es: 'La curva de la cámara da señal nula en algún canal. Revisen el archivo CSV de la respuesta.', en: 'The camera curve gives no signal in some channel. Check the response CSV file.' },
  response_bad: { es: 'El archivo de respuesta no se pudo leer. Usen columnas lambda_nm, R, G y B.', en: 'The response file could not be read. Use columns lambda_nm, R, G and B.' },
  oven_no_aperture: { es: 'No se encontró la abertura del horno. Revisen que el horno esté en el cuadro y que la imagen no esté oscura.', en: 'The furnace aperture was not found. Check that the furnace is in the frame and that the image is not dark.' },
  sphere_no_disc: { es: 'No se encontró el disco de la esfera. Revisen que la salida de la esfera esté en el cuadro.', en: 'The sphere disc was not found. Check that the sphere exit port is in the frame.' },
  sphere_short: { es: 'El espectro de la lámpara no cubre todo el canal {channel}. Usen un espectro de 400 a 1000 nm.', en: 'The lamp spectrum does not cover the whole {channel} channel. Use a spectrum from 400 to 1000 nm.' },
  nd_required: { es: 'Elijan el filtro neutro usado en la esfera. Si no hubo, elijan sin filtro.', en: 'Choose the neutral density filter used on the sphere. If there was none, choose no filter.' },
  spectrum_missing: { es: 'Falta el espectro de la lámpara en CSV. Elíjanlo con su botón.', en: 'The lamp spectrum CSV is missing. Choose it with its button.' },
  quick_zeros: { es: 'Hay píxeles en cero en el oscuro. En la próxima sesión suban el nivel de negro de la cámara.', en: 'Some dark pixels are at zero. Next session, raise the camera black level.' },
  quick_high: { es: 'En el canal {channel} la imagen pasa el 90 % de la escala. Ese canal quedará fuera de la calibración.', en: 'In the {channel} channel the image is above 90 % of full scale. That channel will be left out of the calibration.' },
  far_narrow: { es: 'Estas imágenes son recortes sin la zona lejos de la abertura. El oscuro se estima con la recta.', en: 'These images are crops without the region far from the aperture. The dark is estimated from the line.' },
  log_differs: { es: 'En {n} carpetas la temperatura de registro.csv difiere de la del nombre. Se usa la del termopar.', en: 'In {n} folders the temperature in registro.csv differs from the name. The thermocouple one is used.' },
  pfs_texp: { es: 'El tiempo del archivo .pfs ({a} µs) difiere del nombre ({b} µs) en {file}. Se usa el del .pfs.', en: 'The time in the .pfs file ({a} µs) differs from the name ({b} µs) in {file}. The .pfs value is used.' },
  pfs_differs: { es: 'Los ajustes de pylon de la llama difieren de los del horno en {a}. Revisen que sean los mismos.', en: 'The pylon settings of the flame differ from those of the furnace in {a}. Check that they are the same.' },
  sphere_near_ceiling: { es: 'En la esfera el canal {channel} tiene {a} % del disco sobre el 90 % de la escala y queda dudoso. En la próxima sesión bajen el tiempo de exposición.', en: 'In the sphere the {channel} channel has {a} % of the disc above 90 % of full scale and is doubtful. Next session, shorten the exposure time.' },
  scale_none: { es: 'No hay escala, así que las alturas, los radios y f_s no tienen sentido físico. Escriban la escala o marquen la regla.', en: 'There is no scale, so heights, radii and f_s have no physical meaning. Type the scale or mark the ruler.' },
  scale_needed: { es: 'Falta la escala. Hagan clic en dos marcas de la regla o escriban el valor.', en: 'The scale is missing. Click two marks on the ruler or type the value.' },
  oven_fit_auto: { es: 'La calibración se calculó sola al pulsar «Calcular los pasos 3, 4 y 5».', en: 'The calibration was computed automatically by «Compute steps 3, 4 and 5».' },
  example_cleared: { es: 'Se vació la tabla antes de cargar el ejemplo, para no mezclar cámaras.', en: 'The table was cleared before loading the example, so cameras are not mixed.' },
  average_only: { es: 'Hay un archivo de promedio ({file}). Se usa solo el promedio y quedan fuera los {n} cuadros sueltos.', en: 'There is an averaged file ({file}). Only the average is used and the {n} single frames are left out.' },
  orient_first: { es: 'La llama parece acostada. Elijan el giro y la herramienta vuelve a medir cada cuadro.', en: 'The flame seems to lie on its side. Choose the rotation and the tool measures each frame again.' },
  tilt_warn_one: { es: 'El instante elegido está algo inclinado, asimetría {a}. Revisen la diferencia entre mitades.', en: 'The chosen instant is slightly tilted, asymmetry {a}. Check the difference between halves.' },
  no_frames: { es: 'No hay cuadros de la llama marcados. Revisen la lista en Opciones del paso 2.', en: 'No flame frames are ticked. Check the list in step 2 Options.' },
  no_projections: { es: 'Faltan las imágenes preparadas. Pulsen «Preparar las imágenes» en el paso 2.', en: 'The prepared images are missing. Press «Prepare the images» in step 2.' },
  no_emission: { es: 'Falta la inversión del paso 3.', en: 'The inversion of step 3 is missing.' },
  sat_not_spd: { es: 'El cálculo no converge con este suavizado. Súbanlo un poco y vuelvan a invertir.', en: 'The calculation does not converge with this smoothing. Raise it a little and invert again.' },
  tikhonov_not_spd: { es: 'El cálculo no converge con este suavizado. Súbanlo un poco y vuelvan a invertir.', en: 'The calculation does not converge with this smoothing. Raise it a little and invert again.' },
  odd_width: { es: 'El recorte tiene un número impar de columnas. Cambien los radios por lado en Opciones.', en: 'The crop has an odd number of columns. Change the radii per side in Options.' },
  too_large: { es: 'La imagen es demasiado grande para este navegador. Carguen un recorte hecho en pylon.', en: 'The image is too large for this browser. Load a crop made in pylon.' },
  mixed_bits: { es: 'Los canales tienen distinta profundidad de bits. Guarden la imagen desde pylon como TIFF RGB de 16 bits.', en: 'The channels have different bit depths. Save the image from pylon as a 16-bit RGB TIFF.' },
  missing_tag: { es: 'Al archivo le falta información que todo TIFF trae. Guárdenlo otra vez desde pylon.', en: 'The file lacks information every TIFF carries. Save it again from pylon.' },
  lzw_bad_code: { es: 'La compresión del archivo está dañada. Cópienlo de nuevo o guárdenlo sin compresión.', en: 'The compression of the file is damaged. Copy it again or save it uncompressed.' },
  ycbcr: { es: 'La imagen está en YCbCr, como una foto comprimida. Guárdenla desde pylon como TIFF RGB de 16 bits.', en: 'The image is stored as YCbCr, like a compressed photo. Save it from pylon as a 16-bit RGB TIFF.' },
  cal_unit_dn12: { es: 'El archivo trae la calibración en la unidad de la herramienta.', en: 'The file gives the calibration in the unit of the tool.' },
  cal_unit_dn16: { es: 'Este archivo no es una calibración de esta herramienta. Se usa convertido a su unidad. Si pueden, calibren con el horno.', en: 'This file is not a calibration from this tool. It is used converted to its unit. If you can, calibrate with the furnace.' },
  cal_unit_missing: { es: 'Este archivo no es una calibración de esta herramienta. Se usa convertido a su unidad. Si pueden, calibren con el horno.', en: 'This file is not a calibration from this tool. It is used converted to its unit. If you can, calibrate with the furnace.' },
  unknown: { es: 'La herramienta encontró un problema que no sabe describir. Descarguen el registro en el paso 5 y envíenlo al profesor.', en: 'The tool ran into a problem it cannot describe. Download the log in step 5 and send it to the instructor.' },
};

/** Second sentence for whoever already left the lab (WEB_SPEC §8, last paragraph). */
const AFTER: Record<string, T2> = {
  below_noise: { es: 'Si ya salieron del laboratorio, ese canal de esa imagen queda fuera.', en: 'If you have already left the lab, that channel of that image is left out.' },
  dark_disagree: { es: 'Si ya salieron del laboratorio, la herramienta usa la recta como oscuro.', en: 'If you have already left the lab, the tool uses the line as the dark.' },
};

export interface Msg { code: string; vars?: Record<string, string | number>; files?: string[]; level?: 'warn' | 'error' | 'info' }

export function msgText(m: Msg): { text: string; after?: string } {
  // one of a thing has its own sentence where the plural would read wrong (GB3r2-estudiante-24)
  const e = (m.vars?.n === 1 && MESSAGES[`${m.code}_1`]) || MESSAGES[m.code];
  if (!e) return { text: MESSAGES.unknown[lang] };
  let s = e[lang];
  for (const [k, v] of Object.entries(m.vars ?? {})) {
    const val = k === 'channel' && typeof v === 'number' ? CH_NAME[lang][v] : typeof v === 'number' ? num(v, Number.isInteger(v) ? 0 : 1) : v;
    s = s.split(`{${k}}`).join(String(val));
  }
  if (lang === 'es') s = s.replace(/p99\.9/g, 'p99,9');
  const a = AFTER[m.code];
  return { text: s, after: a ? a[lang] : undefined };
}

/** One message per code and values, with the files it concerns. */
const RANK: Record<string, number> = { info: 0, warn: 1, error: 2 };
export function groupMsgs(list: Msg[]): Msg[] {
  const g = new Map<string, Msg>();
  for (const m of list) {
    const k = m.code + JSON.stringify(m.vars ?? {});
    const x = g.get(k);
    // the same code twice keeps the highest level (GB3r2-adv-13)
    if (x) { x.files = [...(x.files ?? []), ...(m.files ?? [])]; if (RANK[m.level ?? 'warn'] > RANK[x.level ?? 'warn']) x.level = m.level; } else g.set(k, { ...m, files: [...(m.files ?? [])] });
  }
  return [...g.values()];
}

export function msgHtml(list: Msg[]): string {
  return groupMsgs(list).map((m) => {
    const t = msgText(m);
    const files = m.files?.length ? `<span class="wr-msg__files">${L('Afecta a', 'Concerns')} ${m.files.slice(0, 6).map(esc).join(', ')}${m.files.length > 6 ? L(` y ${m.files.length - 6} más`, ` and ${m.files.length - 6} more`) : ''}</span>` : '';
    const lv = m.level ?? 'warn';
    const lab = lv === 'error' ? L('Error.', 'Error.') : lv === 'info' ? L('Nota.', 'Note.') : L('Aviso.', 'Warning.');
    return `<p class="wr-msg wr-msg--${lv}" data-code="${m.code}"><em class="wr-msg__lab">${lab}</em> ${esc(t.text).replace(/f_s/g, 'f<sub>s</sub>').replace(/K_j/g, 'K<sub>j</sub>').replace(/K_R\/K_G/g, 'K<sub>R</sub>/K<sub>G</sub>').replace(/G_j/g, 'G<sub>j</sub>').replace(/ψ_j/g, 'ψ<sub>j</sub>').replace(/ψ_R/g, 'ψ<sub>R</sub>').replace(/K_B/g, 'K<sub>B</sub>')}${t.after ? ` ${esc(t.after)}` : ''} ${files}</p>`;
  }).join('');
}
