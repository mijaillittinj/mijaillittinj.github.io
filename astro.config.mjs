// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Deployment target. Override with env vars for a project repository or a custom domain:
//   SITE_URL=https://example.org BASE_PATH=/ npm run build
const site = process.env.SITE_URL ?? 'https://mijaillittinj.github.io';
const base = process.env.BASE_PATH ?? '/';

export default defineConfig({
  site,
  base,
  trailingSlash: 'always',
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'es'],
    routing: { prefixDefaultLocale: false },
  },
  integrations: [
    mdx(),
    sitemap({
      filter: (page) => !/\/cv\/(print\/)?$/.test(page),
      i18n: { defaultLocale: 'en', locales: { en: 'en', es: 'es' } },
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { output: 'htmlAndMathml' }]],
    shikiConfig: { theme: 'github-light' },
  },
  build: { inlineStylesheets: 'always' },
  prefetch: { prefetchAll: false, defaultStrategy: 'hover' },
});
