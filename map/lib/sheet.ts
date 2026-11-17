import { createContext, useContext } from 'react'

// Phone / tablet layout: one bottom sheet instead of the floating sidebar and
// cards. The query is shared by every map on egagli.github.io: below 1024px
// wide (theme-ui's 64em breakpoint, where these maps already switched to a
// "mobile" layout), or a short coarse-pointer screen (landscape phones).
export const SHEET_MEDIA = '(max-width: 1023px), (pointer: coarse) and (max-height: 520px)'
export const SHEET_MQ = `@media ${SHEET_MEDIA}`
// Landscape phones dock the sheet on the left instead of the bottom.
export const SHEET_LANDSCAPE_MEDIA =
  '(pointer: coarse) and (max-height: 520px) and (orientation: landscape)'

export type SheetMode = 'phone' | 'tablet' | 'land'
export type SheetState = 'peek' | 'half' | 'full'

// Shared visual tokens (same on all maps of the site).
export const SHEET = {
  bg: 'rgb(22,25,30)', // the desktop card colour, opaque
  border: '#2e3138',
  text: '#d0d0d0',
  dim: '#8f95a0', // 5.9:1 on bg; the desktop DIM (#6b7280) is 3.7:1
  accent: '#1dbd8f',
  ease: 'cubic-bezier(.2,.8,.2,1)',
  ms: 220,
  maxTabletWidth: 600,
  landWidth: 380,
  searchClearance: 68, // map padding that keeps results below the search pill
}

// True inside the sheet: components render touch-sized, >=12px, higher-contrast.
export const SheetContext = createContext(false)

/** Colour and type helpers that switch with the sheet context. */
export function useTone(desktopDim: string) {
  const sheet = useContext(SheetContext)
  return {
    sheet,
    dim: sheet ? SHEET.dim : desktopDim,
    fs: (px: number) => (sheet ? Math.max(12, px) : px),
  }
}

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
