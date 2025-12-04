declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<object, object, unknown>
  export default component
}

declare module 'markdown-it' {
  const MarkdownIt: typeof import('@types/markdown-it')
  export = MarkdownIt
}

declare module 'mermaid' {
  interface MermaidRenderOptions {
    nodes: Iterable<Element>
  }

  interface MermaidInstance {
    initialize: (config?: Record<string, unknown>) => void
    run: (options: MermaidRenderOptions) => Promise<void>
  }

  const mermaid: MermaidInstance
  export default mermaid
}
