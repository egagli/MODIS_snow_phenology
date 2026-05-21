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
  type TileClickInfo,
} from '../lib/store'

const MODIS_SINUSOIDAL_PROJ4 =
  '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'

const TILES_STATUS_URL =
  process.env.NODE_ENV === 'production'
    ? '/MODIS_snow_phenology/tiles-status.geojson'
    : '/tiles-status.geojson'

const ACCENT = '#1dbd8f'

const TILE_COLORS = {
  processed:   '#22c55e',
  unprocessed: '#ef4444',
  ocean:       '#3b82f6',
  unknown:     '#94a3b8',
}

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

let pmtilesRegistered = false

// IDs to skip when toggling basemap fill/background visibility for satellite mode
const OWN_LAYER_IDS = new Set([
  'zarr-layer', 'esri-imagery',
  'tiles-fill', 'tiles-outline', 'tiles-highlight', 'tiles-highlight-outline',
])

function setBasemapFillVisibility(map: maplibregl.Map, visible: boolean) {
  const vis = visible ? 'visible' : 'none'
  map.getStyle()?.layers.forEach((layer) => {
    if (OWN_LAYER_IDS.has(layer.id)) return
    if (layer.type === 'fill' || layer.type === 'background') {
      try { map.setLayoutProperty(layer.id, 'visibility', vis) } catch {}
    }
  })
}

export const Map = () => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const zarrLayerRef = useRef<InstanceType<typeof ZarrLayer> | null>(null)
  const markerRef = useRef<maplibregl.Marker | null>(null)

  // useState (not useRef) so that setting it triggers re-renders and dependent effects
  const [isMapLoaded, setIsMapLoaded] = useState(false)

  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const opacity = useStore((s) => s.opacity)
  const clim = useStore((s) => s.clim)
  const colormap = useStore((s) => s.colormap)
  const globeProjection = useStore((s) => s.globeProjection)
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const loadingState = useStore((s) => s.loadingState)
  const showSatellite = useStore((s) => s.showSatellite)
  const showTiles = useStore((s) => s.showTiles)
  const setLoadingState = useStore((s) => s.setLoadingState)
  const setClickInfo = useStore((s) => s.setClickInfo)
  const setTileClickInfo = useStore((s) => s.setTileClickInfo)

  const colormapArray = useThemedColormap(colormap, { format: 'hex' })

  // Map initialization — runs once
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return

    if (!pmtilesRegistered) {
      const protocol = new Protocol()
      maplibregl.addProtocol('pmtiles', protocol.tile)
      pmtilesRegistered = true
    }

    const pmLayers = layers('protomaps', mapTheme as any, { lang: 'en' })
    const satLayer = {
      id: 'esri-imagery',
      type: 'raster' as const,
      source: 'esri-imagery',
      layout: { visibility: 'none' as const },
    }
    const styleLayers = [pmLayers[0], satLayer, ...pmLayers.slice(1)]

    const map = new maplibregl.Map({
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
          'esri-imagery': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            maxzoom: 19,
            attribution:
              'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
          },
        },
        layers: styleLayers,
      },
      center: [0, 20],
      zoom: window.innerWidth < 640 ? 1.2 : 2.4,
    })

    mapRef.current = map

    map.on('load', () => {
      // Tile processing status overlay
      const emptyFC = { type: 'FeatureCollection' as const, features: [] }
      map.addSource('tiles-status', { type: 'geojson', data: TILES_STATUS_URL })
      map.addSource('tiles-highlight-source', { type: 'geojson', data: emptyFC })

      const fillColor: maplibregl.ExpressionSpecification = [
        'case',
        ['==', ['get', 'status'], 'processed'],   TILE_COLORS.processed,
        ['==', ['get', 'status'], 'unprocessed'],  TILE_COLORS.unprocessed,
        ['==', ['get', 'status'], 'ocean'],        TILE_COLORS.ocean,
        ['==', ['get', 'land'], false],            TILE_COLORS.ocean,
        TILE_COLORS.unknown,
      ]

      const beforeLabel = (() => {
        try { if (map.getLayer('address_label')) return 'address_label' } catch {}
        try { if (map.getLayer('landuse_pedestrian')) return 'landuse_pedestrian' } catch {}
        return undefined
      })()

      map.addLayer(
        { id: 'tiles-fill', type: 'fill', source: 'tiles-status',
          paint: { 'fill-color': fillColor, 'fill-opacity': 0.25 },
          layout: { visibility: 'none' } } as any, beforeLabel
      )
      map.addLayer(
        { id: 'tiles-outline', type: 'line', source: 'tiles-status',
          paint: { 'line-color': '#64748b', 'line-width': 0.5 },
          layout: { visibility: 'none' } } as any, beforeLabel
      )
      map.addLayer(
        { id: 'tiles-highlight', type: 'fill', source: 'tiles-highlight-source',
          paint: { 'fill-color': '#ffffff', 'fill-opacity': 0.25 } } as any, beforeLabel
      )
      map.addLayer(
        { id: 'tiles-highlight-outline', type: 'line', source: 'tiles-highlight-source',
          paint: { 'line-color': '#ffffff', 'line-width': 3 } } as any, beforeLabel
      )

      const tileClickHandler = (e: maplibregl.MapLayerMouseEvent) => {
        const feature = e.features?.[0]
        if (!feature) return
        const props = feature.properties as Record<string, unknown>
        const status = (props.status as string) ?? (props.land === false ? 'ocean' : 'unknown')
        setTileClickInfo({ ...props, status } as TileClickInfo)
        const hlSrc = map.getSource('tiles-highlight-source') as maplibregl.GeoJSONSource | undefined
        hlSrc?.setData({
          type: 'FeatureCollection',
          features: [{ type: 'Feature', geometry: feature.geometry, properties: {} }],
        })
      }
      const cursorOn  = () => { map.getCanvas().style.cursor = 'pointer' }
      const cursorOff = () => { map.getCanvas().style.cursor = '' }

      map.on('click', 'tiles-fill', tileClickHandler)
      map.on('mouseenter', 'tiles-fill', cursorOn)
      map.on('mouseleave', 'tiles-fill', cursorOff)

      // Signal that the map is ready — this triggers all dependent useEffects
      setIsMapLoaded(true)
    })

    return () => {
      markerRef.current?.remove()
      markerRef.current = null
      map.remove()
      mapRef.current = null
      setIsMapLoaded(false)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Projection toggle
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    ;(mapRef.current as any).setProjection(
      globeProjection ? { type: 'globe' } : { type: 'mercator' }
    )
  }, [globeProjection, isMapLoaded])

  // Satellite toggle — hide all basemap fill/background layers so they don't mask satellite
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    map.setLayoutProperty('esri-imagery', 'visibility', showSatellite ? 'visible' : 'none')
    setBasemapFillVisibility(map, !showSatellite)
  }, [showSatellite, isMapLoaded])

  // Tiles overlay toggle
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    const v = showTiles ? 'visible' : 'none'
    map.setLayoutProperty('tiles-fill', 'visibility', v)
    map.setLayoutProperty('tiles-outline', 'visibility', v)
    if (!showTiles) {
      setTileClickInfo(null)
      const hlSrc = map.getSource('tiles-highlight-source') as maplibregl.GeoJSONSource | undefined
      hlSrc?.setData({ type: 'FeatureCollection', features: [] })
    }
  }, [showTiles, isMapLoaded, setTileClickInfo])

  // Recreate ZarrLayer when variable changes or map first loads
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    let cancelled = false

    if (zarrLayerRef.current) {
      try { if (map.getLayer('zarr-layer')) map.removeLayer('zarr-layer') } catch {}
      zarrLayerRef.current = null
    }
    markerRef.current?.remove()
    markerRef.current = null

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
      bounds: [-20015087, -10007544, 20015087, 10007544],
      latIsAscending: false,
      onLoadingStateChange: setLoadingState,
    }

    let beforeId: string | undefined
    try {
      beforeId = 'landuse_pedestrian'
      if (!map.getLayer(beforeId)) beforeId = undefined
    } catch { beforeId = undefined }

    const layer = new ZarrLayer(options)
    if (cancelled) return
    map.addLayer(layer, beforeId)
    zarrLayerRef.current = layer

    const clickHandler = async (event: maplibregl.MapMouseEvent) => {
      const currentLayer = zarrLayerRef.current
      if (!currentLayer) return
      const { lng, lat } = event.lngLat
      markerRef.current?.remove()
      markerRef.current = new maplibregl.Marker({ color: ACCENT })
        .setLngLat([lng, lat])
        .addTo(map)
      const currentVariable = useStore.getState().variable
      setClickInfo({ lng, lat, status: 'querying', value: null })
      const result = await currentLayer.queryData({ type: 'Point', coordinates: [lng, lat] })
      const vals = result?.[currentVariable]
      const raw = Array.isArray(vals) ? vals[0] : null
      setClickInfo({
        lng, lat, status: 'done',
        value: typeof raw === 'number' && !isNaN(raw) ? raw : null,
      })
    }

    map.on('click', clickHandler)

    return () => {
      cancelled = true
      map.off('click', clickHandler)
      try { if (map.getLayer('zarr-layer')) map.removeLayer('zarr-layer') } catch {}
      zarrLayerRef.current = null
    }
  }, [variable, isMapLoaded, setLoadingState, setClickInfo])

  // Live updates — no layer recreation
  useEffect(() => {
    const layer = zarrLayerRef.current
    if (!layer) return
    layer.setOpacity(opacity)
    layer.setClim(clim)
    layer.setColormap(colormapArray)
    layer.setSelector({ water_year: { selected: waterYearIndex, type: 'index' } })
  }, [opacity, clim, colormapArray, waterYearIndex])

  // Resize map when sidebar width changes
  useEffect(() => {
    if (mapRef.current) mapRef.current.resize()
  }, [sidebarWidth])

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
