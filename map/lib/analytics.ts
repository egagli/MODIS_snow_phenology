// Google Analytics 4 helpers. The gtag.js snippet itself is loaded in
// pages/_app.tsx; these helpers never throw if it is absent (ad blockers,
// local dev, SSR during static export).

export const GA_MEASUREMENT_ID = 'G-M59931E34K'

type GtagParams = Record<string, string | number | boolean>

declare global {
  interface Window {
    gtag?: (...args: unknown[]) => void
  }
}

export const track = (name: string, params: GtagParams = {}) => {
  if (typeof window === 'undefined' || typeof window.gtag !== 'function') return
  try {
    window.gtag('event', name, params)
  } catch {
    // analytics must never break the map
  }
}

// Debounced variant for controls that fire continuously (e.g. the water-year
// slider): only the value the user settles on is sent.
const timers: Record<string, ReturnType<typeof setTimeout>> = {}
export const trackDebounced = (name: string, params: GtagParams = {}, waitMs = 800) => {
  if (typeof window === 'undefined') return
  clearTimeout(timers[name])
  timers[name] = setTimeout(() => track(name, params), waitMs)
}
