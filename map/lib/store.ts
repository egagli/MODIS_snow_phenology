import { create } from 'zustand'
import type { LoadingState } from '@carbonplan/zarr-layer'

export type Variable = 'SAD_DOWY' | 'SDD_DOWY' | 'max_consec_snow_days'

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

export const ZARR_URL =
  'https://uwcryo.blob.core.windows.net/snowmelt/modis_snow_phenology/modis_snow_phenology_multiscale_v1'

interface AppState {
  variable: Variable
  waterYearIndex: number
  opacity: number
  clim: [number, number]
  colormap: string
  globeProjection: boolean
  loadingState: LoadingState
  sidebarWidth: number | null
  setVariable: (v: Variable) => void
  setWaterYearIndex: (i: number) => void
  setOpacity: (o: number) => void
  setClim: (c: [number, number]) => void
  setColormap: (c: string) => void
  setGlobeProjection: (g: boolean) => void
  setLoadingState: (s: LoadingState) => void
  setSidebarWidth: (w: number | null) => void
}

export const useStore = create<AppState>((set) => ({
  variable: 'SAD_DOWY',
  waterYearIndex: 0,
  opacity: 1,
  clim: VARIABLE_CONFIGS.SAD_DOWY.clim,
  colormap: VARIABLE_CONFIGS.SAD_DOWY.colormap,
  globeProjection: true,
  loadingState: { loading: false, metadata: false, chunks: false },
  sidebarWidth: null,
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
}))
