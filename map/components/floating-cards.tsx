import React, { CSSProperties } from 'react'
import { useStore } from '../lib/store'

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const TEXT = '#d0d0d0'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'

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
}

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  color: DIM,
  marginBottom: 6,
  fontWeight: 600,
}

const chipBase: CSSProperties = {
  flex: 1,
  padding: '5px 0',
  borderRadius: 4,
  border: `1px solid ${BORDER}`,
  cursor: 'pointer',
  fontSize: 12,
  textAlign: 'center',
  transition: 'background 0.15s, color 0.15s',
}

const chipActive: CSSProperties = {
  ...chipBase,
  background: ACCENT,
  color: '#fff',
  borderColor: ACCENT,
}

const chipInactive: CSSProperties = {
  ...chipBase,
  background: 'transparent',
  color: TEXT,
}

function chip(active: boolean): CSSProperties {
  return active ? chipActive : chipInactive
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

// ---------------------------------------------------------------------------
// TopRightCard — Basemap + Projection
// ---------------------------------------------------------------------------

const TopRightCard = ({ right, top }: { right: number; top: number }) => {
  const globeProjection = useStore((s) => s.globeProjection)
  const showSatellite = useStore((s) => s.showSatellite)
  const setGlobeProjection = useStore((s) => s.setGlobeProjection)
  const setShowSatellite = useStore((s) => s.setShowSatellite)

  return (
    <div style={{ ...cardStyle, top, right, width: 220 }}>
      <div style={sectionLabelStyle}>Basemap</div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        {[{ label: 'Dark', value: false }, { label: 'Satellite', value: true }].map((opt) => (
          <button
            key={String(opt.value)}
            onClick={() => setShowSatellite(opt.value)}
            style={chip(showSatellite === opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <div style={sectionLabelStyle}>Projection</div>
      <div style={{ display: 'flex', gap: 6 }}>
        {[{ label: 'Globe', value: true }, { label: 'Mercator', value: false }].map((opt) => (
          <button
            key={String(opt.value)}
            onClick={() => setGlobeProjection(opt.value)}
            style={chip(globeProjection === opt.value)}
          >
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

const TileInspectorCard = ({ right, top }: { right: number; top: number }) => {
  const showTiles = useStore((s) => s.showTiles)
  const tileClickInfo = useStore((s) => s.tileClickInfo)
  const setShowTiles = useStore((s) => s.setShowTiles)
  const setTileClickInfo = useStore((s) => s.setTileClickInfo)

  const SKIP_KEYS = new Set(['geometry', 'type'])

  return (
    <div style={{ ...cardStyle, top, right, width: 220, maxHeight: 'calc(100vh - 260px)', overflowY: 'auto' }}>
      <button
        onClick={() => setShowTiles(!showTiles)}
        style={{
          ...chip(showTiles),
          width: '100%',
          marginBottom: showTiles ? 12 : 0,
        }}
      >
        Processing grid
      </button>

      {showTiles && (
        <>
          <div style={sectionLabelStyle}>Status</div>
          {Object.entries(TILE_STATUS_META).map(([key, { label, color }]) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5 }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />
              <span style={{ color: TEXT, fontSize: 12 }}>{label}</span>
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
                  style={{ background: 'none', border: 'none', color: DIM, cursor: 'pointer', fontSize: 16, lineHeight: 1 }}
                >
                  ×
                </button>
              </div>
              <div style={{
                display: 'inline-block',
                padding: '2px 8px',
                borderRadius: 4,
                background: TILE_STATUS_META[tileClickInfo.status]?.color ?? TILE_STATUS_META.unknown.color,
                color: '#fff',
                fontSize: 11,
                fontWeight: 600,
                marginBottom: 10,
              }}>
                {TILE_STATUS_META[tileClickInfo.status]?.label ?? tileClickInfo.status}
              </div>
              <div>
                {Object.entries(tileClickInfo)
                  .filter(([k]) => !SKIP_KEYS.has(k) && k !== 'tile' && k !== 'status')
                  .map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ color: DIM }}>{propLabel(k)}</span>
                      <span style={{ color: TEXT, fontFamily: 'monospace' }}>{formatPropValue(v)}</span>
                    </div>
                  ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// PointInspectorCard — value at clicked map location
// ---------------------------------------------------------------------------

const PointInspectorCard = ({ right, top }: { right: number; top: number }) => {
  const clickInfo = useStore((s) => s.clickInfo)
  const variable = useStore((s) => s.variable)

  if (!clickInfo) return null

  return (
    <div style={{ ...cardStyle, top, right, width: 220 }}>
      <div style={sectionLabelStyle}>Point query</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ color: DIM }}>Latitude</span>
        <span style={{ fontFamily: 'monospace' }}>{clickInfo.lat.toFixed(4)}°</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
        <span style={{ color: DIM }}>Longitude</span>
        <span style={{ fontFamily: 'monospace' }}>{clickInfo.lng.toFixed(4)}°</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ color: DIM }}>{propLabel(variable)}</span>
        <span style={{ fontFamily: 'monospace', color: ACCENT }}>
          {clickInfo.status === 'querying'
            ? 'Querying…'
            : clickInfo.value !== null
              ? `${Math.round(clickInfo.value)} DOY`
              : '—'}
        </span>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// FloatingCards — root export, desktop-only (hidden on mobile)
// ---------------------------------------------------------------------------

export const FloatingCards = ({ sidebarWidth }: { sidebarWidth: number }) => {
  const TOP_RIGHT = 16
  const CARD_RIGHT = 16

  const topCardHeight = 118  // approximate height of TopRightCard
  const tileCardTop = TOP_RIGHT + topCardHeight + 8

  const showTiles = useStore((s) => s.showTiles)
  const tileClickInfo = useStore((s) => s.tileClickInfo)
  const clickInfo = useStore((s) => s.clickInfo)

  // Estimate tile card height to position point card below it
  const baseTileCardHeight = 130
  const tileInfoExtra = tileClickInfo ? 120 : 0
  const tileCardHeight = showTiles ? baseTileCardHeight + tileInfoExtra : 52
  const pointCardTop = tileCardTop + tileCardHeight + 8

  if (!clickInfo && !showTiles) {
    return (
      <div style={{ display: 'none', ['@media (min-width: 640px)' as any]: { display: 'block' } }}>
        <TopRightCard right={CARD_RIGHT} top={TOP_RIGHT} />
        <TileInspectorCard right={CARD_RIGHT} top={tileCardTop} />
      </div>
    )
  }

  return (
    <>
      <TopRightCard right={CARD_RIGHT} top={TOP_RIGHT} />
      <TileInspectorCard right={CARD_RIGHT} top={tileCardTop} />
      <PointInspectorCard right={CARD_RIGHT} top={pointCardTop} />
    </>
  )
}
