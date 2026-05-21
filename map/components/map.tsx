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
  VARIABLE_CONFIGS,
  type Variable,
  type ClickInfo,
  type TileClickInfo,
} from '../lib/store'

const MODIS_SINUSOIDAL_PROJ4 =
  '+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs'

const TILES_STATUS_URL =
  process.env.NODE_ENV === 'production'
    ? '/MODIS_snow_phenology/tiles-status.geojson'
    : '/tiles-status.geojson'

const ACCENT = '#1dbd8f'
const FILL_VALUE = -32768
const ALL_VARIABLES = ['SAD_DOWY', 'SDD_DOWY', 'max_consec_snow_days'] as const satisfies readonly Variable[]

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

function isValidValue(raw: unknown): raw is number {
  return typeof raw === 'number' && !isNaN(raw) && raw !== FILL_VALUE && raw >= 0
}

export const Map = () => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const zarrLayersRef = useRef<Partial<Record<Variable, InstanceType<typeof ZarrLayer>>>>({})
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const lastClickRef = useRef<{ lng: number; lat: number } | null>(null)

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
      attributionControl: false,
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

    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left')
    mapRef.current = map

    map.on('load', () => {
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
          paint: { 'fill-color': fillColor, 'fill-opacity': 0.1 },
          layout: { visibility: 'none' } } as any, beforeLabel
      )
      map.addLayer(
        { id: 'tiles-outline', type: 'line', source: 'tiles-status',
          paint: { 'line-color': '#64748b', 'line-width': 0.5 },
          layout: { visibility: 'none' } } as any, beforeLabel
      )
      map.addLayer(
        { id: 'tiles-highlight', type: 'fill', source: 'tiles-highlight-source',
          paint: { 'fill-color': '#ffffff', 'fill-opacity': 0.15 } } as any, beforeLabel
      )
      map.addLayer(
        { id: 'tiles-highlight-outline', type: 'line', source: 'tiles-highlight-source',
          paint: { 'line-color': '#ffffff', 'line-width': 2 } } as any, beforeLabel
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

  // Satellite toggle — hide all basemap fill/background layers to avoid masking satellite
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

  // Query all three variables at the currently clicked point
  const requeryAllVariables = (cancelled: { val: boolean }) => {
    const coords = lastClickRef.current
    if (!coords) return
    const EMPTY = { SAD_DOWY: null, SDD_DOWY: null, max_consec_snow_days: null }
    setClickInfo({ lng: coords.lng, lat: coords.lat, status: 'querying', values: EMPTY })
    Promise.all(
      ALL_VARIABLES.map(async (varName) => {
        const layer = zarrLayersRef.current[varName]
        if (!layer) return [varName, null] as [Variable, number | null]
        const result = await layer.queryData({ type: 'Point', coordinates: [coords.lng, coords.lat] })
        const vals = result?.[varName]
        const raw = Array.isArray(vals) ? vals[0] : null
        return [varName, isValidValue(raw) ? raw : null] as [Variable, number | null]
      })
    ).then((entries) => {
      if (cancelled.val) return
      const values = Object.fromEntries(entries) as ClickInfo['values']
      setClickInfo({ lng: coords.lng, lat: coords.lat, status: 'done', values })
    })
  }

  // Create all three ZarrLayers once when map loads; attach single click handler
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    const cancelled = { val: false }
    const state = useStore.getState()

    ALL_VARIABLES.forEach((varName) => {
      const isActive = varName === state.variable
      const vCfg = VARIABLE_CONFIGS[varName]
      // x!=x detects NaN portably; <-100 catches raw fill -32768 if NaN conversion is skipped
      const customFrag = `
        if (${varName} != ${varName} || ${varName} < -100.0) { discard; }
        float rescaled = (${varName} - clim.x) / (clim.y - clim.x);
        vec4 c = texture(colormap, vec2(rescaled, 0.5));
        fragColor = vec4(c.rgb, opacity);
        fragColor.rgb *= fragColor.a;
      `
      const options: ZarrLayerOptions = {
        id: `zarr-${varName}`,
        source: ZARR_URL,
        variable: varName,
        clim: isActive ? state.clim : vCfg.clim,
        colormap: makeColormap(isActive ? state.colormap : vCfg.colormap, { format: 'hex' }),
        opacity: isActive ? state.opacity : 0,
        selector: { water_year: { selected: state.waterYearIndex, type: 'index' } },
        zarrVersion: 3,
        fillValue: FILL_VALUE,
        proj4: MODIS_SINUSOIDAL_PROJ4,
        bounds: [-20015087, -10007544, 20015087, 10007544],
        latIsAscending: false,
        onLoadingStateChange: (ls) => {
          if (useStore.getState().variable === varName) setLoadingState(ls)
        },
        customFrag,
      }
      let beforeId: string | undefined
      try { if (map.getLayer('address_label')) beforeId = 'address_label' } catch {}
      if (!cancelled.val) {
        const layer = new ZarrLayer(options)
        map.addLayer(layer, beforeId)
        zarrLayersRef.current[varName] = layer
      }
    })

    const clickHandler = (event: maplibregl.MapMouseEvent) => {
      const { lng, lat } = event.lngLat
      lastClickRef.current = { lng, lat }
      markerRef.current?.remove()
      markerRef.current = new maplibregl.Marker({ color: ACCENT }).setLngLat([lng, lat]).addTo(map)
      requeryAllVariables({ val: false })
    }
    map.on('click', clickHandler)

    return () => {
      cancelled.val = true
      map.off('click', clickHandler)
      ALL_VARIABLES.forEach((varName) => {
        try { if (map.getLayer(`zarr-${varName}`)) map.removeLayer(`zarr-${varName}`) } catch {}
      })
      zarrLayersRef.current = {}
    }
  }, [isMapLoaded, setLoadingState, setClickInfo]) // eslint-disable-line react-hooks/exhaustive-deps

  // Switch which layer is visible when variable changes; re-query selected point
  useEffect(() => {
    if (!isMapLoaded) return
    ALL_VARIABLES.forEach((v) => {
      zarrLayersRef.current[v]?.setOpacity(v === variable ? opacity : 0)
    })
    if (lastClickRef.current) requeryAllVariables({ val: false })
  }, [variable, isMapLoaded]) // eslint-disable-line react-hooks/exhaustive-deps

  // Opacity + clim + colormap updates for the active layer
  useEffect(() => {
    const layer = zarrLayersRef.current[variable]
    if (!layer) return
    layer.setOpacity(opacity)
    layer.setClim(clim)
    layer.setColormap(colormapArray)
  }, [variable, opacity, clim, colormapArray])

  // Water year selector update on all layers + re-query selected point
  useEffect(() => {
    ALL_VARIABLES.forEach((v) => {
      zarrLayersRef.current[v]?.setSelector({ water_year: { selected: waterYearIndex, type: 'index' } })
    })
    if (lastClickRef.current) requeryAllVariables({ val: false })
  }, [waterYearIndex]) // eslint-disable-line react-hooks/exhaustive-deps

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
          left: 0,
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
