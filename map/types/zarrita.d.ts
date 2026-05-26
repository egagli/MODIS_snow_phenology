// Minimal type shims for zarrita and @zarrita/storage/fetch.
// These packages live in zarr-layer's node_modules and are aliased via webpack
// in next.config.js. TypeScript's moduleResolution:"node" can't follow sub-path
// exports, so we declare the shapes we actually use here.

declare module 'zarrita' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function open(store: any, opts?: { kind?: string; path?: string }): Promise<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function get(arr: any, sel: (number | null)[]): Promise<any>
}

declare module '@zarrita/storage/fetch' {
  export default class FetchStore {
    constructor(url: string)
  }
}
