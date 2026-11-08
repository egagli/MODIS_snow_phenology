// View state <-> URL. Every setting worth sending to someone lives in the
// page's URL fragment, is restored on load, and is kept current with
// history.replaceState. The "Share this view" button turns it into a link to
// the map's wrapper page on egagli.github.io, and when the map is embedded in
// that page we tell it about each change so its own address bar stays current.
//
// Why the fragment and not the query string: GA4's enhanced measurement sends
// a page_view for every history.replaceState that changes the path or query,
// so a query-string view would log a page view on every map move. Fragment-only
// changes are not counted. The same keys are still read from the query string
// (hash wins), so ?var=SDD_DOWY&wy=2021 links work too; they are moved into the
// fragment on load.
//
// Keys (all optional; any other query parameters are left alone):
//   map=zoom/lat/lng[/bearing/pitch]  map view
//   var=SAD_DOWY|SDD_DOWY|max_consec_snow_days
//   wy=2021                           water year (calendar year it ends in)
//   cmap=reds   clim=1,366            colormap and range (omitted when default)
//   opacity=0.8                       layer opacity (omitted when 1)
//   basemap=satellite|topography      (omitted when dark)
//   proj=mercator                     (omitted when globe)
//   pt=lat,lng                        selected point (point-query mode)
//   grid=1   tile=h10v04              processing-grid mode and selected tile

import {
  useStore,
  VARIABLE_CONFIGS,
  WATER_YEARS,
  COLORMAPS,
  type Variable,
  type Basemap,
} from './store'

export const SHARE_BASE = 'https://egagli.github.io/maps/modis-snow-phenology.html'
const SITE_ORIGIN = 'https://egagli.github.io'

export type MapView = { lng: number; lat: number; zoom: number; bearing: number; pitch: number }

export type UrlState = {
  view?: MapView
  variable?: Variable
  waterYearIndex?: number
  colormap?: string
  clim?: [number, number]
  opacity?: number
  basemap?: Basemap
  globeProjection?: boolean
  point?: { lng: number; lat: number }
  showTiles?: boolean
  tile?: string
}

const OWN_KEYS = ['map', 'var', 'wy', 'cmap', 'clim', 'opacity', 'basemap', 'proj', 'pt', 'grid', 'tile']
const BASEMAPS: Basemap[] = ['dark', 'satellite', 'topography']

const num = (s: string | null | undefined): number | null => {
  if (s === null || s === undefined || s.trim() === '') return null
  const n = Number(s)
  return Number.isFinite(n) ? n : null
}

const round = (n: number, d: number) => String(Number(n.toFixed(d)))

function ownParams(): URLSearchParams {
  const p = new URLSearchParams()
  const query = new URLSearchParams(window.location.search)
  const hash = new URLSearchParams(window.location.hash.replace(/^#/, ''))
  for (const src of [query, hash]) {
    OWN_KEYS.forEach((k) => { const v = src.get(k); if (v !== null) p.set(k, v) })
  }
  return p
}

export function readUrlState(): UrlState {
  if (typeof window === 'undefined') return {}
  const p = ownParams()
  const out: UrlState = {}

  const view = p.get('map')?.split('/').map(num)
  if (view && view.length >= 3 && view.slice(0, 3).every((v) => v !== null)) {
    const [zoom, lat, lng, bearing, pitch] = view as number[]
    if (Math.abs(lat) <= 90 && zoom >= 0 && zoom <= 24) {
      out.view = { zoom, lat, lng: ((((lng + 180) % 360) + 360) % 360) - 180, bearing: bearing ?? 0, pitch: pitch ?? 0 }
    }
  }

  const v = p.get('var')
  if (v && v in VARIABLE_CONFIGS) out.variable = v as Variable

  const wy = num(p.get('wy'))
  const wyIdx = wy === null ? -1 : (WATER_YEARS as readonly number[]).indexOf(wy)
  if (wyIdx >= 0) out.waterYearIndex = wyIdx

  const cmap = p.get('cmap')
  if (cmap && COLORMAPS.includes(cmap)) out.colormap = cmap

  const clim = p.get('clim')?.split(',').map(num)
  if (clim && clim.length === 2 && clim[0] !== null && clim[1] !== null) {
    out.clim = [clim[0], clim[1]]
  }

  const opacity = num(p.get('opacity'))
  if (opacity !== null && opacity >= 0 && opacity <= 1) out.opacity = opacity

  const basemap = p.get('basemap')
  if (basemap && (BASEMAPS as string[]).includes(basemap)) out.basemap = basemap as Basemap

  const proj = p.get('proj')
  if (proj === 'mercator') out.globeProjection = false
  if (proj === 'globe') out.globeProjection = true

  const pt = p.get('pt')?.split(',').map(num)
  if (pt && pt.length === 2 && pt[0] !== null && pt[1] !== null && Math.abs(pt[0]) <= 90 && Math.abs(pt[1]) <= 180) {
    out.point = { lat: pt[0], lng: pt[1] }
  }

  const grid = p.get('grid')
  if (grid === '1') out.showTiles = true

  const tile = p.get('tile')
  if (tile && /^h\d{2}v\d{2}$/.test(tile)) out.tile = tile

  return out
}

// A tile named in the URL is only selected once the map has loaded the grid;
// until then it is kept here so URL writes don't drop it.
let pendingTile: string | null = null
export const getPendingTile = () => pendingTile
export const clearPendingTile = () => {
  pendingTile = null
}

// Restores everything except the map view (the map reads that itself at init).
export function applyUrlStateToStore(s: UrlState) {
  const st = useStore.getState()
  if (s.variable) st.setVariable(s.variable) // resets colormap/clim to the variable's defaults
  if (s.waterYearIndex !== undefined) st.setWaterYearIndex(s.waterYearIndex)
  if (s.colormap) st.setColormap(s.colormap)
  if (s.clim) st.setClim(s.clim)
  if (s.opacity !== undefined) st.setOpacity(s.opacity)
  if (s.basemap) st.setBasemap(s.basemap)
  if (s.globeProjection !== undefined) st.setGlobeProjection(s.globeProjection)
  if (s.showTiles) {
    st.setShowTiles(true)
    pendingTile = s.tile ?? null
  } else if (s.point) {
    // The map places the marker and runs the query once its layers are ready.
    st.setClickInfo({
      lng: s.point.lng,
      lat: s.point.lat,
      status: 'querying',
      values: { SAD_DOWY: null, SDD_DOWY: null, max_consec_snow_days: null },
    })
  }
}

let viewGetter: (() => MapView | null) | null = null
export const registerViewGetter = (fn: (() => MapView | null) | null) => {
  viewGetter = fn
}

function buildHash(): string {
  const p = ownParams()
  const st = useStore.getState()
  const set = (k: string, v: string | null) => (v === null ? p.delete(k) : p.set(k, v))

  const view = viewGetter?.()
  if (view) {
    const parts = [round(view.zoom, 2), round(view.lat, 4), round(view.lng, 4)]
    const bearing = Math.abs(view.bearing) >= 0.05 ? view.bearing : 0
    const pitch = Math.abs(view.pitch) >= 0.05 ? view.pitch : 0
    if (bearing || pitch) parts.push(round(bearing, 1), round(pitch, 1))
    set('map', parts.join('/'))
  } // before the map exists, keep whatever view the URL already had

  const cfg = VARIABLE_CONFIGS[st.variable]
  set('var', st.variable)
  set('wy', String(WATER_YEARS[st.waterYearIndex]))
  set('cmap', st.colormap !== cfg.colormap ? st.colormap : null)
  set('clim', st.clim[0] !== cfg.clim[0] || st.clim[1] !== cfg.clim[1] ? st.clim.join(',') : null)
  set('opacity', st.opacity !== 1 ? round(st.opacity, 2) : null)
  set('basemap', st.basemap !== 'dark' ? st.basemap : null)
  set('proj', st.globeProjection ? null : 'mercator')

  if (st.showTiles) {
    set('pt', null)
    set('grid', '1')
    set('tile', st.tileClickInfo?.tile ?? pendingTile)
  } else {
    pendingTile = null
    set('grid', null)
    set('tile', null)
    const ci = st.clickInfo
    set('pt', ci ? `${round(ci.lat, 5)},${round(ci.lng, 5)}` : null)
  }

  const ordered = new URLSearchParams()
  OWN_KEYS.forEach((k) => { const v = p.get(k); if (v !== null) ordered.set(k, v) })
  const qs = ordered.toString().replace(/%2C/g, ',').replace(/%2F/g, '/')
  return qs ? `#${qs}` : ''
}

// The query string minus our keys (they live in the fragment).
function foreignSearch(): string {
  const q = new URLSearchParams(window.location.search)
  if (!OWN_KEYS.some((k) => q.has(k))) return window.location.search
  OWN_KEYS.forEach((k) => q.delete(k))
  const s = q.toString()
  return s ? `?${s}` : ''
}

export function postStateToParent() {
  if (typeof window === 'undefined' || window.parent === window) return
  try {
    window.parent.postMessage(
      { type: 'egagli-map-state', search: window.location.search, hash: window.location.hash },
      SITE_ORIGIN
    )
  } catch {
    // never let embedding break the map
  }
}

// Returns true when the URL changed (and the parent page was told).
// The hash as we last wrote it. If the hash no longer matches, someone else
// changed it (typed a new view, followed a link) and the hashchange handler is
// about to reload into it: skip this write so the old view can't overwrite the
// new one. (Only one write is skipped, so this can never loop.)
let lastHash: string | null = null

export function reloadIntoNewHash() {
  suspendUrlWrites()
  window.location.reload()
}

export function writeUrlNow(): boolean {
  if (typeof window === 'undefined' || suspended) return false
  if (lastHash !== null && window.location.hash !== lastHash) {
    lastHash = window.location.hash
    return false
  }
  const search = foreignSearch()
  const hash = buildHash()
  if (search === window.location.search && hash === window.location.hash) {
    lastHash = hash
    return false
  }
  const url = window.location.pathname + search + hash
  try {
    // Keep history.state: the Next.js router stores its own state there.
    window.history.replaceState(window.history.state, '', url)
  } catch {
    return false
  } finally {
    lastHash = window.location.hash
  }
  postStateToParent()
  return true
}

let timer: ReturnType<typeof setTimeout> | undefined
let suspended = false
// Stop writing (e.g. while reloading into a hash someone typed), so a pending
// update can't overwrite the new hash with the old view.
export function suspendUrlWrites() {
  suspended = true
  clearTimeout(timer)
}
export function scheduleUrlUpdate(waitMs = 300) {
  if (typeof window === 'undefined' || suspended) return
  clearTimeout(timer)
  timer = setTimeout(() => writeUrlNow(), waitMs)
}

export const buildShareUrl = () =>
  SHARE_BASE + window.location.search + window.location.hash
