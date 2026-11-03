// File and folder names of the workshop tool (names.ts), the oven folders as typed on Spanish and English
// keyboards and the crop file that goes with them (GB3r3-adv-06):
// run with `node --experimental-strip-types tests/workshop-rgb-names.test.ts`
import { parseName, parseCrop, cropFor } from '../src/components/workshop-rgb/names.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

// 1. temperature read from the folder in every spelling
for (const d of ['1050C', '1050 C', '1050ºC', '1050 °C', '1050 grados']) {
  const n = parseName(`horno/${d}/20000exp.tiff`);
  check(`T from "${d}"`, n.tC === 1050, `tC = ${n.tC}`);
}

// 2. recorte.json written with the plain names finds the crop of renamed folders
const crops = parseCrop(JSON.stringify({ '800C': { x0: 654, y0: 306 }, '900C': { x0: 654, y0: 306 }, '1000C': { x0: 654, y0: 306 }, '1050C': { x0: 654, y0: 306 } }));
for (const d of ['800C', '900 °C', '1000 C', '1050ºC']) {
  const c = cropFor(`horno_ejemplo/${d}/20000exp.tiff`, crops);
  check(`crop for "${d}"`, !!c && c.x0 === 654 && c.y0 === 306, JSON.stringify(c));
}
check('no crop for another folder', cropFor('horno_ejemplo/oscuro/20000exp.tiff', crops) === undefined);
check('crop for every folder with "*"', cropFor('a/b.tiff', parseCrop('{"x0": 3, "y0": 4}'))?.x0 === 3);

// 3. the laboratory flame set, the ruler under either name, the flame with its exposure time and its crop
for (const f of ['llama_laboratorio/regla.tiff', 'real_flame/Escala.tiff']) check(`ruler "${f}"`, parseName(f).role === 'ruler', parseName(f).role);
{
  const n = parseName('llama_laboratorio/llama/200exp.tiff');
  check('lab flame role and t_exp', n.role === 'flame' && n.texpUs === 200, `${n.role} ${n.texpUs}`);
  const c = cropFor('llama_laboratorio/llama/200exp.tiff', parseCrop('{"llama":{"y0":400,"x0":500,"sensor":[1200,1920],"camera":"a2A1920-160ucPRO"}}'));
  check('lab flame crop', !!c && c.x0 === 500 && c.y0 === 400, JSON.stringify(c));
}

if (fails) { console.log(`\n${fails} FAIL`); process.exit(1); }
console.log('\nall names tests pass');
