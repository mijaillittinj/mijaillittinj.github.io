// Checks for the publication merge: run with `node tests/publications.test.ts`
import { readFileSync } from 'node:fs';
import { mergePublications, titleSimilarity, slugFor, type AutoWork } from '../src/lib/pubmerge.ts';
import type { Publication } from '../src/data/publications.ts';

let fails = 0;
const check = (name: string, ok: boolean, info = '') => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`); if (!ok) fails++; };

const base = (o: Partial<Publication>): Publication => ({
  id: 'x', title: 'T', authors: ['M. Littin'], year: 2026, journal: 'J', status: 'published',
  themes: ['reconstruction'], domains: [], summary: 'curated summary', access: { kind: 'closed' }, ...o,
});
const work = (o: Partial<AutoWork>): AutoWork => ({
  doi: '10.1/x', title: 'T', authors: ['A. Author', 'M. Littin'], year: 2026, journal: 'J', type: 'journal-article', volume: '1', article: '1', ...o,
});

// 1. title matching tolerates case, punctuation and accents, rejects different papers
check('similar titles match', titleSimilarity('Soot maturity evolution in a periodically-forced diffusion flame', 'Soot Maturity Evolution in a Periodically Forced Diffusion Flame') >= 0.8);
check('different titles do not match', titleSimilarity('Soot maturity evolution in a forced flame', 'X-ray scattering by fractal agglomerates') < 0.2);

// 2. curated entries win; auto duplicates are not appended
{
  const out = mergePublications([base({ id: 'a', doi: '10.1/A', title: 'Curated title' })], [work({ doi: '10.1/a', title: 'Crossref title' })]);
  check('curated wins over auto', out.length === 1 && out[0].title === 'Curated title' && out[0].summary === 'curated summary');
}

// 3. accepted without DOI is upgraded by title; in-press with DOI upgraded when a volume appears
{
  const out = mergePublications(
    [
      base({ id: 'acc', status: 'accepted', title: 'Impact of sustainable aviation fuel on soot properties' }),
      base({ id: 'inp', status: 'in-press', doi: '10.1/p', title: 'In press paper' }),
    ],
    [
      work({ doi: '10.1/new', title: 'Impact of Sustainable Aviation Fuel on Soot Properties', volume: '42', article: '99' }),
      work({ doi: '10.1/p', title: 'In press paper', volume: '7' }),
    ],
  );
  const acc = out.find((p) => p.id === 'acc')!, inp = out.find((p) => p.id === 'inp')!;
  check('accepted upgraded by title', acc.status === 'published' && acc.doi === '10.1/new' && acc.volume === '42' && out.length === 2);
  check('in-press upgraded by DOI', inp.status === 'published' && inp.volume === '7');
}

// 4. new works appended with stable unique ids
{
  const out = mergePublications([base({ id: 'pinto2027flame', doi: '10.1/c' })], [
    work({ doi: '10.1/n1', authors: ['P. E. Pinto', 'M. Littin'], year: 2027, title: 'Flame radiation in pools' }),
    work({ doi: '10.1/n2', authors: ['M. Littin'], year: 2027, title: 'The neural inversion', volume: undefined, inPress: true }),
  ]);
  const ids = out.map((p) => p.id);
  check('unique ids', new Set(ids).size === ids.length, ids.join(','));
  check('slug collision resolved', ids.includes('pinto2027flame2'));
  check('new entry flags', out[2].firstAuthor === true && out[2].status === 'in-press' && out[1].themes.length === 0);
  check('slug deterministic', slugFor(work({ title: 'The neural inversion', authors: ['M. Littin'], year: 2027 }), new Set()) === 'littin2027neural');
}

// 5. the committed auto file is well formed and every work names the candidate
{
  const auto = JSON.parse(readFileSync(new URL('../src/data/publications.auto.json', import.meta.url), 'utf8'));
  const ok = Array.isArray(auto.works) && auto.works.length > 0 && auto.works.every((w: AutoWork) => w.doi && w.title && w.year && w.authors.includes('M. Littin'));
  check('auto file well formed', ok, `${auto.works?.length} works`);
}

if (fails) { console.error(`${fails} failure(s)`); process.exit(1); }
console.log('All publication checks passed.');
