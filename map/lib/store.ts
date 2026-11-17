import { create } from 'zustand'
import type { LoadingState } from '@carbonplan/zarr-layer'
import type { SheetMode } from './sheet'

export type Variable = 'SAD_DOWY' | 'SDD_DOWY' | 'max_consec_snow_days'
export type Basemap = 'dark' | 'satellite' | 'topography'

export const VARIABLE_CONFIGS: Record<
  Variable,
  { clim: [number, number]; colormap: string; label: string; units: string }
> = {
  SAD_DOWY: {
    clim: [1, 366],
    colormap: 'purples',
    label: 'Snow Appearance Date',
    units: 'day of water year',
  },
  SDD_DOWY: {
    clim: [1, 366],
    colormap: 'reds',
    label: 'Snow Disappearance Date',
    units: 'day of water year',
  },
  max_consec_snow_days: {
    clim: [0, 366],
    colormap: 'blues',
    label: 'Max Consecutive Snow Days',
    units: 'days',
  },
}

export const COLORMAPS: string[] = [
  'rainbow', 'fire', 'earth', 'water', 'warm', 'cool',
  'reds', 'oranges', 'yellows', 'greens', 'teals', 'blues',
  'purples', 'pinks', 'greys', 'wind', 'heart', 'sinebow',
  'pinkgreen', 'redteal', 'orangeblue', 'yellowpurple',
]

export const WATER_YEARS = [
  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
] as const

export const TILES_STATUS_URL =
  process.env.NODE_ENV === 'production'
    ? '/MODIS_snow_phenology/tiles-status.geojson'
    : '/tiles-status.geojson'

// Injected at build time from MULTISCALE_PREFIX in the config file via NEXT_PUBLIC_ZARR_URL.
// Falls back to the v2 store so local `npm run dev` works without setting the env var.
export const ZARR_URL =
  process.env.NEXT_PUBLIC_ZARR_URL ??
  'https://uwcryo.blob.core.windows.net/snowmelt/modis_snow_phenology/modis_snow_phenology_multiscale_v2'

// GMBA v2.0 mountain-range index for the search box. Shared with the
// global_snowmelt_runoff_onset map, which builds it (prepare_gmba_overlay.py)
// and publishes it under an immutable, suffix-versioned name; reuse it by URL
// rather than vendoring a copy.
export const GMBA_INDEX_URL =
  process.env.NEXT_PUBLIC_GMBA_INDEX_URL ??
  'https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/gmba_v2_standard_300_index_2.json'

// One index entry.
export type GmbaInfo = {
  id: number
  name: string
  path: string // 'Eurasia > Europe > Alps > Western Alps', ending at this range
  countries: string
  feature: string
  level: number | null
  area_km2: number | null
  elev_low: number | null
  elev_high: number | null
  wikidata: string
  bbox: [number, number, number, number] // [w, s, e, n]; east may exceed 180
  center: [number, number]
}

let gmbaIndexPromise: Promise<GmbaInfo[] | null> | null = null

/** Fetch the GMBA index once (memoized) and put it in the store. */
export function loadGmbaIndex(): Promise<GmbaInfo[] | null> {
  if (!gmbaIndexPromise) {
    gmbaIndexPromise = fetch(GMBA_INDEX_URL)
      .then((r) => (r.ok ? r.json() : null))
      .then((entries: GmbaInfo[] | null) => {
        if (entries) useStore.getState().setGmbaIndex(entries)
        return entries
      })
      .catch((e) => {
        console.warn('GMBA index unavailable:', e)
        gmbaIndexPromise = null // allow a retry on the next request
        return null
      })
  }
  return gmbaIndexPromise
}

// A request from the search box for the map to move: fit a bbox, or fly to a
// point and query it. `seq` makes repeated identical requests distinct.
export type NavTarget =
  | { kind: 'bounds'; bbox: [number, number, number, number] }
  | { kind: 'point'; lat: number; lng: number }
export type NavRequest = NavTarget & { seq: number }

export type ClickInfo = {
  lng: number
  lat: number
  status: 'querying' | 'done'
  values: { SAD_DOWY: number | null; SDD_DOWY: number | null; max_consec_snow_days: number | null }
}

export type TileClickInfo = Record<string, unknown> & {
  tile: string
  processing_status: string
  land: boolean
}

interface AppState {
  variable: Variable
  waterYearIndex: number
  opacity: number
  clim: [number, number]
  colormap: string
  globeProjection: boolean
  loadingState: LoadingState
  sidebarWidth: number | null
  basemap: Basemap
  showTiles: boolean
  clickInfo: ClickInfo | null
  tileClickInfo: TileClickInfo | null
  activeWarning: { name: string; message: string } | null
  zoomLevel: number
  gmbaIndex: GmbaInfo[] | null
  navRequest: NavRequest | null
  // Phone/tablet sheet: its layout (null = desktop) and how much of the map it
  // covers once settled, which the map uses as camera padding.
  sheetMode: SheetMode | null
  sheetInset: { bottom: number; left: number }
  setVariable: (v: Variable) => void
  setWaterYearIndex: (i: number) => void
  setOpacity: (o: number) => void
  setClim: (c: [number, number]) => void
  setColormap: (c: string) => void
  setGlobeProjection: (g: boolean) => void
  setLoadingState: (s: LoadingState) => void
  setSidebarWidth: (w: number | null) => void
  setBasemap: (b: Basemap) => void
  setShowTiles: (v: boolean) => void
  setClickInfo: (info: ClickInfo | null) => void
  setTileClickInfo: (info: TileClickInfo | null) => void
  setActiveWarning: (w: { name: string; message: string } | null) => void
  setZoomLevel: (z: number) => void
  setGmbaIndex: (index: GmbaInfo[]) => void
  requestNav: (r: NavTarget) => void
  setSheet: (mode: SheetMode | null, inset: { bottom: number; left: number }) => void
}

export const useStore = create<AppState>((set) => ({
  variable: 'max_consec_snow_days',
  waterYearIndex: 0,
  opacity: 1,
  clim: VARIABLE_CONFIGS.max_consec_snow_days.clim,
  colormap: VARIABLE_CONFIGS.max_consec_snow_days.colormap,
  globeProjection: true,
  loadingState: { loading: false, metadata: false, chunks: false },
  sidebarWidth: null,
  basemap: 'dark',
  showTiles: false,
  clickInfo: null,
  tileClickInfo: null,
  activeWarning: null,
  zoomLevel: 2.4,
  gmbaIndex: null,
  navRequest: null,
  sheetMode: null,
  sheetInset: { bottom: 0, left: 0 },
  setVariable: (variable) =>
    set({
      variable,
      clim: VARIABLE_CONFIGS[variable].clim,
      colormap: VARIABLE_CONFIGS[variable].colormap,
    }),
  setWaterYearIndex: (waterYearIndex) => set({ waterYearIndex }),
  setOpacity: (opacity) => set({ opacity }),
  setClim: (clim) => set({ clim }),
  setColormap: (colormap) => set({ colormap }),
  setGlobeProjection: (globeProjection) => set({ globeProjection }),
  setLoadingState: (loadingState) => set({ loadingState }),
  setSidebarWidth: (sidebarWidth) => set({ sidebarWidth }),
  setBasemap: (basemap) => set({ basemap }),
  setShowTiles: (showTiles) => set({ showTiles }),
  setClickInfo: (clickInfo) => set({ clickInfo }),
  setTileClickInfo: (tileClickInfo) => set({ tileClickInfo }),
  setActiveWarning: (activeWarning) => set({ activeWarning }),
  setZoomLevel: (zoomLevel) => set({ zoomLevel }),
  setGmbaIndex: (gmbaIndex) => set({ gmbaIndex }),
  requestNav: (r) =>
    set((state) => ({ navRequest: { ...r, seq: (state.navRequest?.seq ?? 0) + 1 } })),
  setSheet: (sheetMode, sheetInset) => {
    const cur = useStore.getState()
    if (
      cur.sheetMode === sheetMode &&
      cur.sheetInset.bottom === sheetInset.bottom &&
      cur.sheetInset.left === sheetInset.left
    ) return
    set({ sheetMode, sheetInset })
  },
}))
