import React, { CSSProperties } from 'react'
import { useStore, WATER_YEARS, type Variable } from '../lib/store'

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const TEXT = '#d0d0d0'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'
const CARD_WIDTH = 300

const TILE_STATUS_META: Record<string, { label: string; color: string }> = {
  processed:   { label: 'Processed',   color: '#22c55e' },
  unprocessed: { label: 'Unprocessed', color: '#ef4444' },
  ocean:       { label: 'Ocean',       color: '#3b82f6' },
  unknown:     { label: 'Unknown',     color: '#94a3b8' },
}

const cardStyle: CSSProperties = {
  position: 'absolute',
  background: BG,
  border: `1px solid ${BORDER}`,
  backdropFilter: 'blur(6px)',
  borderRadius: 8,
  color: TEXT,
  fontSize: 12,
  zIndex: 10,
  padding: '14px 16px',
  width: CARD_WIDTH,
}

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  color: DIM,
  marginBottom: 6,
  fontWeight: 600,
}

function chip(active: boolean): CSSProperties {
  return {
    flex: 1,
    padding: '5px 0',
    borderRadius: 4,
    border: `1px solid ${active ? ACCENT : BORDER}`,
    cursor: 'pointer',
    fontSize: 12,
    textAlign: 'center' as const,
    background: active ? ACCENT : 'transparent',
    color: active ? '#fff' : TEXT,
    transition: 'background 0.15s, color 0.15s',
  }
}

function formatPropValue(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') return isNaN(v) ? '—' : Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  return String(v)
}

function propLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

// DOWY → calendar date for NH water year (Oct 1 of waterYear-1 is day 1)
function dowyToDate(dowy: number, waterYear: number): string {
  const start = new Date(waterYear - 1, 9, 1) // Oct 1 of previous calendar year
  const target = new Date(start.getTime() + (Math.round(dowy) - 1) * 86400000)
  return target.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function formatValue(value: number | null, variable: Variable, waterYear: number): string {
  if (value === null) return 'No data'
  if (variable === 'max_consec_snow_days') {
    return `${Math.round(value)} days`
  }
  const dowy = Math.round(value)
  return `${dowy} (${dowyToDate(dowy, waterYear)})`
}

// ---------------------------------------------------------------------------
// TopRightCard — Basemap + Projection
// ---------------------------------------------------------------------------

const TopRightCard = ({ right, top }: { right: number; top: number }) => {
  const globeProjection = useStore((s) => s.globeProjection)
  const showSatellite = useStore((s) => s.showSatellite)
  const setGlobeProjection = useStore((s) => s.setGlobeProjection)
  const setShowSatellite = useStore((s) => s.setShowSatellite)

  return (
    <div style={{ ...cardStyle, top, right }}>
      <div style={sectionLabelStyle}>Basemap</div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        {[{ label: 'Dark', value: false }, { label: 'Satellite', value: true }].map((opt) => (
          <button key={String(opt.value)} onClick={() => setShowSatellite(opt.value)} style={chip(showSatellite === opt.value)}>
            {opt.label}
          </button>
        ))}
      </div>
      <div style={sectionLabelStyle}>Projection</div>
      <div style={{ display: 'flex', gap: 6 }}>
        {[{ label: 'Globe', value: true }, { label: 'Mercator', value: false }].map((opt) => (
          <button key={String(opt.value)} onClick={() => setGlobeProjection(opt.value)} style={chip(globeProjection === opt.value)}>
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// TileInspectorCard — Tile overlay toggle + legend + click info
// ---------------------------------------------------------------------------

const SKIP_KEYS = new Set(['geometry', 'type', 'tile', 'status'])

const TileInspectorCard = ({ right, top }: { right: number; top: number }) => {
  const showTiles = useStore((s) => s.showTiles)
  const tileClickInfo = useStore((s) => s.tileClickInfo)
  const setShowTiles = useStore((s) => s.setShowTiles)
  const setTileClickInfo = useStore((s) => s.setTileClickInfo)

  return (
    <div style={{ ...cardStyle, top, right, maxHeight: 'calc(100vh - 260px)', overflowY: 'auto' }}>
      <button
        onClick={() => setShowTiles(!showTiles)}
        style={{ ...chip(showTiles), width: '100%', marginBottom: showTiles ? 12 : 0 }}
      >
        Processing grid
      </button>

      {showTiles && (
        <>
          <div style={sectionLabelStyle}>Status</div>
          {Object.entries(TILE_STATUS_META).map(([key, { label, color }]) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5 }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />
              <span>{label}</span>
            </div>
          ))}

          {tileClickInfo && (
            <>
              <div style={{ borderTop: `1px solid ${BORDER}`, margin: '12px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div style={{ fontFamily: 'monospace', fontSize: 16, fontWeight: 700, color: '#fff' }}>
                  {tileClickInfo.tile as string}
                </div>
                <button
                  onClick={() => setTileClickInfo(null)}
                  style={{ background: 'none', border: 'none', color: DIM, cursor: 'pointer', fontSize: 16, lineHeight: 1, padding: 0 }}
                >
                  ×
                </button>
              </div>
              <div style={{
                display: 'inline-block', padding: '2px 8px', borderRadius: 4, marginBottom: 10,
                background: TILE_STATUS_META[tileClickInfo.status]?.color ?? TILE_STATUS_META.unknown.color,
                color: '#fff', fontSize: 11, fontWeight: 600,
              }}>
                {TILE_STATUS_META[tileClickInfo.status]?.label ?? tileClickInfo.status}
              </div>
              {Object.entries(tileClickInfo)
                .filter(([k]) => !SKIP_KEYS.has(k))
                .map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: DIM }}>{propLabel(k)}</span>
                    <span style={{ fontFamily: 'monospace' }}>{formatPropValue(v)}</span>
                  </div>
                ))}
            </>
          )}
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// PointInspectorCard — always visible at bottom-right; shows placeholder until clicked
// ---------------------------------------------------------------------------

const PointInspectorCard = ({ right, bottom }: { right: number; bottom: number }) => {
  const clickInfo = useStore((s) => s.clickInfo)
  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const waterYear = WATER_YEARS[waterYearIndex]

  const varLabel = variable === 'SAD_DOWY' ? 'SAD DOWY'
    : variable === 'SDD_DOWY' ? 'SDD DOWY'
    : 'Max consec snow days'

  return (
    <div style={{ ...cardStyle, bottom, right }}>
      <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase' as const, color: TEXT, marginBottom: 10 }}>Point Query</div>
      {clickInfo ? (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 7 }}>
            <span style={{ color: DIM, fontSize: 14 }}>Latitude</span>
            <span style={{ fontFamily: 'monospace', fontSize: 15 }}>{clickInfo.lat.toFixed(4)}°</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 7 }}>
            <span style={{ color: DIM, fontSize: 14 }}>Longitude</span>
            <span style={{ fontFamily: 'monospace', fontSize: 15 }}>{clickInfo.lng.toFixed(4)}°</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 7 }}>
            <span style={{ color: DIM, fontSize: 14 }}>Water Year</span>
            <span style={{ fontFamily: 'monospace', fontSize: 15 }}>{waterYear}</span>
          </div>
          <div style={{ borderTop: `1px solid ${BORDER}`, margin: '10px 0' }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
            <span style={{ color: DIM, fontSize: 14, flexShrink: 0 }}>{varLabel}</span>
            <span style={{ fontFamily: 'monospace', fontSize: 18, fontWeight: 700, textAlign: 'right',
              color: clickInfo.status === 'querying' ? DIM
                : clickInfo.value === null ? DIM
                : ACCENT }}>
              {clickInfo.status === 'querying'
                ? 'Querying…'
                : formatValue(clickInfo.value, variable, waterYear)}
            </span>
          </div>
        </>
      ) : (
        <div style={{ color: DIM, fontStyle: 'italic', fontSize: 14 }}>Click the map to query</div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// FloatingCards — always rendered (desktop only via parent Box in index.tsx)
// ---------------------------------------------------------------------------

export const FloatingCards = () => {
  const CARD_RIGHT = 16
  const TOP_RIGHT_TOP = 16
  const TOP_CARD_HEIGHT = 118
  const TILE_CARD_TOP = TOP_RIGHT_TOP + TOP_CARD_HEIGHT + 8

  return (
    <>
      <TopRightCard right={CARD_RIGHT} top={TOP_RIGHT_TOP} />
      <TileInspectorCard right={CARD_RIGHT} top={TILE_CARD_TOP} />
      <PointInspectorCard right={CARD_RIGHT} bottom={16} />
    </>
  )
}
