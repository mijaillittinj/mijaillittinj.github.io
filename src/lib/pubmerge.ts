/**
 * Merge of the hand-curated publication list with the automatic ORCID/Crossref
 * list (src/data/publications.auto.json, written by scripts/sync-publications.mjs).
 *
 * Rules
 *  - Curated entries win: summaries, themes, flags and links are never touched.
 *  - A curated entry that is 'accepted' or 'in-press' is upgraded when Crossref has
 *    caught up: matched by DOI or, without a DOI, by title (token Jaccard ≥ 0.8).
 *  - Auto works whose DOI is not curated are appended as published entries,
 *    without summary, themes or domains, under a stable unique slug.
 * Pure functions, tested in tests/publications.test.ts.
 */
import type { Publication } from '../data/publications.ts';

export interface AutoWork {
  doi: string;
  title: string;
  authors: string[];
  year: number;
  journal: string;
  volume?: string;
  article?: string;
  type: string;
  inPress?: boolean;
  license?: string;
}

const STOP = new Set(['a', 'an', 'the', 'of', 'in', 'on', 'for', 'and', 'to', 'by', 'with', 'from', 'at', 'via', 'as', 'its', 'into', 'using']);

export function tokens(title: string): string[] {
  return title
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .split(' ')
    .filter((t) => t && !STOP.has(t));
}

export function titleSimilarity(a: string, b: string): number {
  const A = new Set(tokens(a)), B = new Set(tokens(b));
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter++;
  return inter / (A.size + B.size - inter);
}

const normDoi = (d?: string) => (d ?? '').trim().toLowerCase();

/** 'P. E. Pinto' -> 'pinto'; keeps only ASCII letters. */
function surname(author: string): string {
  const last = author.trim().split(/\s+/).pop() ?? 'anon';
  return last.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z]/g, '') || 'anon';
}

export function slugFor(w: AutoWork, taken: Set<string>): string {
  const word = tokens(w.title).find((t) => t.length > 3 && !/^\d+$/.test(t)) ?? 'paper';
  const base = `${surname(w.authors[0] ?? 'anon')}${w.year}${word}`;
  let id = base;
  for (let k = 2; taken.has(id); k++) id = `${base}${k}`;
  taken.add(id);
  return id;
}

export function mergePublications(curated: Publication[], auto: AutoWork[]): Publication[] {
  const byDoi = new Map(auto.map((w) => [normDoi(w.doi), w]));
  const used = new Set<string>();

  const merged = curated.map((p): Publication => {
    let w = p.doi ? byDoi.get(normDoi(p.doi)) : undefined;
    if (!w && !p.doi && p.status === 'accepted') {
      w = auto.find((a) => !used.has(normDoi(a.doi)) && titleSimilarity(a.title, p.title) >= 0.8);
    }
    if (p.doi) used.add(normDoi(p.doi));
    if (!w) return p;
    used.add(normDoi(w.doi));
    if (p.status !== 'accepted' && p.status !== 'in-press') return p;
    const published = !w.inPress && !!w.volume;
    return {
      ...p,
      doi: p.doi ?? w.doi,
      volume: p.volume ?? w.volume,
      article: p.article ?? w.article,
      year: w.year,
      status: published ? 'published' : 'in-press',
    };
  });

  const taken = new Set(curated.map((p) => p.id));
  const extra = auto
    .filter((w) => !used.has(normDoi(w.doi)))
    .map((w): Publication => ({
      id: slugFor(w, taken),
      title: w.title,
      authors: w.authors,
      year: w.year,
      journal: w.journal,
      volume: w.volume,
      article: w.article,
      doi: w.doi,
      status: w.inPress ? 'in-press' : 'published',
      firstAuthor: w.authors[0] === 'M. Littin',
      themes: [],
      domains: [],
      summary: '',
      access: w.license
        ? { kind: 'open', license: w.license, pdf: `https://doi.org/${w.doi}` }
        : { kind: 'unknown' },
    }));

  return [...merged, ...extra];
}
