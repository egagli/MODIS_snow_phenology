import React, { useEffect, useRef, useState } from 'react'
import { Box, Spinner } from 'theme-ui'
import { useThemedColormap, makeColormap } from '@carbonplan/colormaps'
import { ZarrLayer, ZarrLayerOptions } from '@carbonplan/zarr-layer'
import maplibregl from 'maplibre-gl'
import { layers, namedFlavor } from '@protomaps/basemaps'
import { Protocol } from 'pmtiles'
import {
  useStore,
  ZARR_URL,
} from '../lib/store'

const MODIS_SINUSOIDAL_PROJ4 =
  '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'

const backgroundColor = '#1b1e23'
const mapTheme = {
  ...namedFlavor('black'),
  background: backgroundColor,
  earth: backgroundColor,
  park_a: backgroundColor,
  park_b: backgroundColor,
  golf_course: backgroundColor,
  aerodrome: backgroundColor,
  industrial: backgroundColor,
  university: backgroundColor,
  school: backgroundColor,
  zoo: backgroundColor,
  farmland: backgroundColor,
  wood_a: backgroundColor,
  wood_b: backgroundColor,
  residential: backgroundColor,
  protected_area: backgroundColor,
  scrub_a: backgroundColor,
  scrub_b: backgroundColor,
  landcover: {
    barren: backgroundColor,
    farmland: backgroundColor,
    forest: backgroundColor,
    glacier: backgroundColor,
    grassland: backgroundColor,
    scrub: backgroundColor,
    urban_area: backgroundColor,
  },
  regular: 'Relative Pro Book',
  bold: 'Relative Pro Book',
  italic: 'Relative Pro Book',
}

export const Map = () => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const zarrLayerRef = useRef<InstanceType<typeof ZarrLayer> | null>(null)
  const [map, setMap] = useState<maplibregl.Map | null>(null)
  const [isMapLoaded, setIsMapLoaded] = useState(false)

  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const opacity = useStore((s) => s.opacity)
  const clim = useStore((s) => s.clim)
  const colormap = useStore((s) => s.colormap)
  const globeProjection = useStore((s) => s.globeProjection)
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const loadingState = useStore((s) => s.loadingState)
  const setLoadingState = useStore((s) => s.setLoadingState)

  const colormapArray = useThemedColormap(colormap, { format: 'hex' })

  useEffect(() => {
    if (!mapContainer.current) return

    const protocol = new Protocol()
    maplibregl.addProtocol('pmtiles', protocol.tile)

    const newMap = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        projection: { type: 'globe' } as any,
        version: 8,
        glyphs:
          'https://carbonplan-maps.s3.us-west-2.amazonaws.com/basemaps/fonts/{fontstack}/{range}.pbf',
        sources: {
          protomaps: {
            type: 'vector',
            url: 'pmtiles://https://carbonplan-maps.s3.us-west-2.amazonaws.com/basemaps/pmtiles/global.pmtiles',
            attribution:
              '<a href="https://overturemaps.org/">Overture Maps</a>, <a href="https://protomaps.com">Protomaps</a>, © <a href="https://openstreetmap.org">OpenStreetMap</a>',
          },
        },
        layers: layers('protomaps', mapTheme as any, { lang: 'en' }),
      },
      center: [0, 20],
      zoom: window.innerWidth < 640 ? 1.2 : 2.4,
    })

    newMap.on('load', () => {
      setMap(newMap)
      setIsMapLoaded(true)
    })

    return () => {
      try {
        newMap.remove()
      } catch {}
      setMap(null)
      setIsMapLoaded(false)
    }
  }, [])

  useEffect(() => {
    if (!map || !isMapLoaded) return
    ;(map as any).setProjection(
      globeProjection ? { type: 'globe' } : { type: 'mercator' }
    )
  }, [map, isMapLoaded, globeProjection])

  // Recreate ZarrLayer when the variable changes (different data array)
  useEffect(() => {
    if (!map || !isMapLoaded) return
    let cancelled = false

    if (zarrLayerRef.current) {
      try {
        if (map.getLayer('zarr-layer')) map.removeLayer('zarr-layer')
      } catch {}
      zarrLayerRef.current = null
    }

    const createLayer = () => {
      if (cancelled) return

      const state = useStore.getState()
      const options: ZarrLayerOptions = {
        id: 'zarr-layer',
        source: ZARR_URL,
        variable: state.variable,
        clim: state.clim,
        colormap: makeColormap(state.colormap, { format: 'hex' }),
        opacity: state.opacity,
        selector: { water_year: { selected: state.waterYearIndex, type: 'index' } },
        zarrVersion: 3,
        fillValue: -32768,
        proj4: MODIS_SINUSOIDAL_PROJ4,
        // Bounds in MODIS sinusoidal meters: ±πR × ±πR/2 where R=6371007.181
        bounds: [-20015087, -10007544, 20015087, 10007544],
        latIsAscending: false,
        onLoadingStateChange: setLoadingState,
      }

      if (cancelled) return

      const layer = new ZarrLayer(options)
      let beforeId: string | undefined
      try {
        beforeId = 'landuse_pedestrian'
        if (!map.getLayer(beforeId)) beforeId = undefined
      } catch {
        beforeId = undefined
      }
      map.addLayer(layer, beforeId)
      zarrLayerRef.current = layer
    }

    try {
      createLayer()
    } catch (err) {
      console.error('ZarrLayer creation failed:', err)
    }

    return () => {
      cancelled = true
      if (zarrLayerRef.current) {
        try {
          if (map.getLayer('zarr-layer')) map.removeLayer('zarr-layer')
        } catch {}
        zarrLayerRef.current = null
      }
    }
  }, [map, isMapLoaded, variable, setLoadingState])

  // Update layer properties without recreating (water year, opacity, clim, colormap)
  useEffect(() => {
    const layer = zarrLayerRef.current
    if (!layer || !map || !isMapLoaded) return

    layer.setOpacity(opacity)
    layer.setClim(clim)
    layer.setColormap(colormapArray)
    layer.setSelector({ water_year: { selected: waterYearIndex, type: 'index' } })
  }, [map, isMapLoaded, opacity, clim, colormapArray, waterYearIndex])

  useEffect(() => {
    if (map) map.resize()
  }, [map, sidebarWidth])

  return (
    <>
      <Box
        ref={mapContainer}
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          bottom: ['50vh', '50vh', 0],
          left: sidebarWidth ?? 0,
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          top: ['56px', '56px', '8px'],
          left: (sidebarWidth ?? 0) + 10,
          pointerEvents: 'none',
        }}
      >
        {loadingState.loading && <Spinner size={40} />}
      </Box>
    </>
  )
}
