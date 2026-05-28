import { create } from 'zustand'
import type { LoadingState } from '@carbonplan/zarr-layer'

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

export const WATER_YEARS = [
  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024,
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
}))
