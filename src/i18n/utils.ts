import { defaultLang, translatedRoutes, type Lang } from './ui';

const BASE = import.meta.env.BASE_URL.replace(/\/$/, '');

/** Prefix an internal path with the configured base path. `url('/research/')`. */
export function url(path: string): string {
  if (/^(https?:|mailto:|#)/.test(path)) return path;
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${BASE}${p}`;
}

/** Strip base path and /es prefix: returns the English route key. */
export function routeKey(pathname: string): string {
  let p = pathname;
  if (BASE && p.startsWith(BASE)) p = p.slice(BASE.length) || '/';
  if (p === '/es' || p.startsWith('/es/')) p = p.slice(3) || '/';
  if (!p.endsWith('/')) p += '/';
  return p;
}

export function langFromPath(pathname: string): Lang {
  let p = pathname;
  if (BASE && p.startsWith(BASE)) p = p.slice(BASE.length);
  return p === '/es' || p.startsWith('/es/') ? 'es' : defaultLang;
}

/** Localized href for a route key. Falls back to English when no translation exists. */
export function localized(key: string, lang: Lang): string {
  if (lang === 'es' && translatedRoutes.has(key)) return url(`/es${key === '/' ? '/' : key}`);
  return url(key);
}

export function hasTranslation(key: string): boolean {
  return translatedRoutes.has(key);
}
