#!/usr/bin/env node
/**
 * Automatic publication sync: ORCID + Crossref -> src/data/publications.auto.json
 *
 *   node scripts/sync-publications.mjs
 *
 * Sources
 *   (a) ORCID public API: every work with a DOI on the candidate's record.
 *   (b) Crossref: works carrying the candidate's ORCID iD, plus an author search
 *       for "Mijail Littin". The search also returns other people named Littin,
 *       so a work is accepted only if an author has family name Littin AND a given
 *       name starting with "Mijail" or the candidate's ORCID iD.
 *   (c) The DOIs already in the curated list, so their metadata stays current.
 * Preprints, theses, peer reviews and components are excluded (theses are listed
 * by hand in publications.ts). Metadata of every accepted DOI comes from Crossref.
 *
 * Safety: if a source fails, or the result would drop works that the committed
 * file already lists, the old file is kept (exit 0, with a warning), so a network
 * problem never empties the publication list. Output is sorted and has no
 * timestamp churn: the file is rewritten only when the works change.
 * The merge with the curated list happens at build time (src/lib/pubmerge.ts).
 */
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const ORCID = '0000-0002-5577-5060';
const MAILTO = 'mijaillittin@icloud.com';
const OUT = fileURLToPath(new URL('../src/data/publications.auto.json', import.meta.url));
const UA = `mijaillittin-site/1.0 (mailto:${MAILTO})`;
const KEEP_TYPES = new Set(['journal-article', 'proceedings-article', 'book-chapter']);

async function getJson(url, headers = {}) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'application/json', ...headers }, signal: AbortSignal.timeout(30000) });
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      if (attempt === 3) throw new Error(`${url}: ${e.message}`);
      await new Promise((res) => setTimeout(res, 1500 * attempt));
    }
  }
}

const normDoi = (d) => d.trim().toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, '').replace(/^doi:/, '');

/**
 * Strict: family Littin and given name Mijail or the ORCID iD.
 * Loose (only for DOIs the candidate already vouches for, on ORCID or in the
 * curated list): family Littin with initial M., as many publishers print it.
 */
function isCandidate(author, loose = false) {
  if (!author || (author.family ?? '').trim().toLowerCase() !== 'littin') return false;
  const given = (author.given ?? '').trim().toLowerCase();
  if (given.startsWith('mijail') || (author.ORCID ?? '').includes(ORCID)) return true;
  return loose && given.startsWith('m');
}

/** DOIs already in the hand-curated list (src/data/publications.ts). */
async function curatedDois() {
  const src = await readFile(fileURLToPath(new URL('../src/data/publications.ts', import.meta.url)), 'utf8');
  const papers = src.split('export const theses')[0]; // theses are listed by hand
  return new Set([...papers.matchAll(/doi:\s*'([^']+)'/g)].map((m) => normDoi(m[1])));
}

async function orcidDois() {
  const d = await getJson(`https://pub.orcid.org/v3.0/${ORCID}/works`);
  const out = new Set();
  for (const g of d?.group ?? []) {
    const type = g['work-summary']?.[0]?.type;
    if (!['journal-article', 'conference-paper', 'book-chapter'].includes(type)) continue;
    for (const e of g['external-ids']?.['external-id'] ?? []) {
      if (e['external-id-type'] === 'doi' && e['external-id-relationship'] !== 'part-of') out.add(normDoi(e['external-id-value']));
    }
  }
  return out;
}

async function crossrefDois() {
  const out = new Set();
  const base = `https://api.crossref.org/works?rows=200&select=DOI,type,author&mailto=${MAILTO}`;
  const byOrcid = await getJson(`${base}&filter=orcid:${ORCID}`);
  for (const w of byOrcid?.message?.items ?? []) if (KEEP_TYPES.has(w.type)) out.add(normDoi(w.DOI));
  const bySearch = await getJson(`${base}&query.author=${encodeURIComponent('Mijail Littin')}`);
  for (const w of bySearch?.message?.items ?? []) {
    if (KEEP_TYPES.has(w.type) && (w.author ?? []).some(isCandidate)) out.add(normDoi(w.DOI));
  }
  return out;
}

const initials = (given) =>
  given
    .replace(/\./g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.split('-').map((p) => p[0].toUpperCase() + '.').join('-'))
    .join(' ');

function formatAuthor(a) {
  if (isCandidate(a, true)) return 'M. Littin';
  if (!a.family) return a.name ?? '';
  return a.given ? `${initials(a.given)} ${a.family}` : a.family;
}

const clean = (s) => (s ?? '').replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();

async function metadata(doi, trusted) {
  const d = await getJson(`https://api.crossref.org/works/${encodeURIComponent(doi)}?mailto=${MAILTO}`);
  const m = d?.message;
  if (!m || !KEEP_TYPES.has(m.type)) return null;
  if (!(m.author ?? []).some((a) => isCandidate(a, trusted))) return null; // not the candidate's work (e.g. a wrong ORCID link)
  const date = m['published-print'] ?? m['published-online'] ?? m.issued;
  const year = date?.['date-parts']?.[0]?.[0];
  if (!year) return null;
  const vor = (m.license ?? []).find((l) => l['content-version'] === 'vor' && /creativecommons\.org/.test(l.URL));
  const lic = vor ? vor.URL.match(/licenses\/([a-z-]+)\/([\d.]+)/i) : null;
  const work = {
    doi: normDoi(m.DOI),
    title: clean(m.title?.[0]),
    authors: (m.author ?? []).map(formatAuthor),
    year,
    journal: clean(m['container-title']?.[0]),
    type: m.type,
  };
  if (m.volume) work.volume = String(m.volume);
  const art = m['article-number'] ?? m.page;
  if (art) work.article = String(art);
  if (!m.volume && !m['published-print']) work.inPress = true;
  if (vor) work.license = lic ? `CC ${lic[1].toUpperCase()} ${lic[2]}` : 'CC';
  return work;
}

async function main() {
  let old = { works: [] };
  try { old = JSON.parse(await readFile(OUT, 'utf8')); } catch { /* first run */ }

  let dois, trusted;
  try {
    const [a, b, c] = await Promise.all([orcidDois(), crossrefDois(), curatedDois()]);
    trusted = new Set([...a, ...c]);
    dois = [...new Set([...a, ...b, ...c])].sort();
    const unlisted = [...c].filter((d) => !a.has(d) && !b.has(d));
    console.log(`ORCID: ${a.size} DOIs, Crossref: ${b.size} DOIs, curated: ${c.size}, union: ${dois.length}`);
    if (unlisted.length) console.log(`  curated but found by neither source (add them to ORCID): ${unlisted.join(', ')}`);
  } catch (e) {
    console.warn(`WARNING: source unavailable, keeping ${OUT} (${e.message})`);
    return;
  }

  const works = [];
  for (const doi of dois) {
    try {
      const w = await metadata(doi, trusted.has(doi));
      if (w) works.push(w);
      else console.log(`  skipped ${doi} (type or authorship)`);
    } catch (e) {
      console.warn(`WARNING: metadata failed for ${doi}, keeping ${OUT} (${e.message})`);
      return;
    }
  }
  works.sort((x, y) => y.year - x.year || x.doi.localeCompare(y.doi));

  const lost = old.works.filter((o) => !works.some((w) => w.doi === o.doi));
  if (lost.length && !process.argv.includes('--allow-removal')) {
    console.warn(`WARNING: ${lost.length} previously listed work(s) not returned (${lost.map((w) => w.doi).join(', ')}); keeping ${OUT}. Re-run with --allow-removal to accept.`);
    return;
  }
  if (JSON.stringify(old.works) === JSON.stringify(works)) {
    console.log(`No change (${works.length} works).`);
    return;
  }
  const out = {
    generated: new Date().toISOString().slice(0, 10),
    sources: `ORCID ${ORCID} public record and Crossref (works with this ORCID iD or authored by Mijail Littin). Generated by scripts/sync-publications.mjs; do not edit by hand.`,
    works,
  };
  await writeFile(OUT, JSON.stringify(out, null, 1) + '\n');
  console.log(`Wrote ${works.length} works to ${OUT}`);
}

main().catch((e) => { console.warn(`WARNING: ${e.message}`); });
