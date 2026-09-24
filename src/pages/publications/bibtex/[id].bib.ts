import type { APIRoute, GetStaticPaths } from 'astro';
import { visiblePublications, bibtex } from '../../../data/publications';

export const getStaticPaths: GetStaticPaths = () =>
  visiblePublications.map((p) => ({ params: { id: p.id }, props: { p } }));

export const GET: APIRoute = ({ props }) =>
  new Response(bibtex(props.p), { headers: { 'Content-Type': 'application/x-bibtex; charset=utf-8' } });
