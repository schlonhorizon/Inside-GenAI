declare module '*.yaml' {
  const data: unknown;
  export default data;
}

declare module '*.mdx' {
  import type { ComponentType } from 'react';
  export const frontmatter: Record<string, unknown>;
  const MDXComponent: ComponentType<{ components?: Record<string, unknown> }>;
  export default MDXComponent;
}

declare module 'virtual:chapter-meta' {
  const metas: unknown[];
  export default metas;
}
