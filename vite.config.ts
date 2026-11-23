import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import mdx from '@mdx-js/rollup';
import remarkFrontmatter from 'remark-frontmatter';
import remarkMdxFrontmatter from 'remark-mdx-frontmatter';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { parse as parseYaml } from 'yaml';
import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

/** Erlaubt `import data from './x.yaml'` (Glossar, Credits). */
function yamlPlugin(): Plugin {
  return {
    name: 'igai-yaml',
    transform(code, id) {
      if (!/\.ya?ml$/.test(id)) return null;
      return { code: `export default ${JSON.stringify(parseYaml(code))};`, map: null };
    },
  };
}

/**
 * `virtual:chapter-meta`: Frontmatter aller Kapitel, ohne die Kapitel selbst zu bündeln.
 * So bleibt jedes Kapitel ein eigener Chunk, die Navigation kennt trotzdem alle Titel.
 */
function chapterMetaPlugin(): Plugin {
  const ID = 'virtual:chapter-meta';
  const RESOLVED = '\0' + ID;
  const dir = resolve(process.cwd(), 'content/chapters');
  return {
    name: 'igai-chapter-meta',
    resolveId: (id) => (id === ID ? RESOLVED : null),
    load(id) {
      if (id !== RESOLVED) return null;
      const files = readdirSync(dir).filter((f) => f.endsWith('.mdx')).sort();
      const metas = files.map((f) => {
        const full = join(dir, f);
        this.addWatchFile(full);
        const m = /^---\n([\s\S]*?)\n---/.exec(readFileSync(full, 'utf8'));
        return { ...(m ? parseYaml(m[1]) : {}), file: `/content/chapters/${f}` };
      });
      return `export default ${JSON.stringify(metas)};`;
    },
    handleHotUpdate({ file, server }) {
      if (file.startsWith(dir)) {
        const mod = server.moduleGraph.getModuleById(RESOLVED);
        if (mod) server.moduleGraph.invalidateModule(mod);
      }
    },
  };
}

// BASE_PATH wird im Deploy-Workflow aus actions/configure-pages gesetzt
// (z. B. "/Inside-GenAI/"). Lokal läuft die Seite unter "/".
export default defineConfig({
  base: process.env.BASE_PATH ?? '/',
  plugins: [
    yamlPlugin(),
    chapterMetaPlugin(),
    {
      enforce: 'pre',
      ...mdx({
        providerImportSource: '@mdx-js/react',
        remarkPlugins: [remarkFrontmatter, [remarkMdxFrontmatter, { name: 'frontmatter' }], remarkMath],
        rehypePlugins: [[rehypeKatex, { throwOnError: false, strict: false }]],
      }),
    },
    react({ include: /\.(mdx|tsx?|jsx?)$/ }),
  ],
  build: {
    outDir: 'dist',
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 900,
  },
});
