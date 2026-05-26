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
const CARD_WIDTH = 340

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

// DOWY → calendar date, hemisphere-aware.
// NH: water year starts Oct 1 of (waterYear-1)  e.g. WY2015 = 2014-10-01
// SH: water year starts Apr 1 of waterYear       e.g. WY2015 = 2015-04-01
function dowyToDate(dowy: number, waterYear: number, southernHemisphere: boolean): string {
  const start = southernHemisphere
    ? new Date(waterYear, 3, 1)       // Apr 1 of the water year
    : new Date(waterYear - 1, 9, 1)   // Oct 1 of the previous calendar year
  const target = new Date(start.getTime() + (Math.round(dowy) - 1) * 86400000)
  return target.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function formatValue(
  value: number | null,
  variable: Variable,
  waterYear: number,
  southernHemisphere: boolean,
): string {
  if (value === null) return 'No data'
  if (variable === 'max_consec_snow_days') {
    return `${Math.round(value)} days`
  }
  const dowy = Math.round(value)
  return `${dowy} (${dowyToDate(dowy, waterYear, southernHemisphere)})`
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

// Columns that encode per-water-year stats as "{year}_valid_pixels" / "{year}_input_obs".
// These are rendered as a compact table instead of individual key-value rows.
const WY_STAT_RE = /^(\d{4})_(valid_pixels|input_obs)$/

const TOTAL_PIXELS = 2400 * 2400 // pixels per MODIS tile

/** Pull per-WY stats out of tileClickInfo props and return sorted rows. */
function extractWyStats(props: Record<string, unknown>): {
  year: number; inputObs: number | null; coverage: number | null
}[] {
  const byYear: Record<number, { inputObs: number | null; coverage: number | null }> = {}
  for (const [k, v] of Object.entries(props)) {
    const m = WY_STAT_RE.exec(k)
    if (!m) continue
    const year = parseInt(m[1])
    if (!byYear[year]) byYear[year] = { inputObs: null, coverage: null }
    const num = typeof v === 'number' && !isNaN(v) ? v : null
    if (m[2] === 'input_obs') byYear[year].inputObs = num
    if (m[2] === 'valid_pixels') byYear[year].coverage = num !== null ? (100 * num) / TOTAL_PIXELS : null
  }
  return Object.entries(byYear)
    .map(([yr, s]) => ({ year: parseInt(yr), ...s }))
    .sort((a, b) => a.year - b.year)
}

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
              {/* Scalar properties (skip per-WY stat columns — rendered below) */}
              {Object.entries(tileClickInfo)
                .filter(([k, v]) => !SKIP_KEYS.has(k) && !WY_STAT_RE.test(k) && v !== null && v !== undefined)
                .map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ color: DIM }}>{propLabel(k)}</span>
                    <span style={{ fontFamily: 'monospace' }}>{formatPropValue(v)}</span>
                  </div>
                ))}

              {/* Per-water-year stats — compact table */}
              {(() => {
                const rows = extractWyStats(tileClickInfo as Record<string, unknown>)
                if (rows.length === 0) return null
                return (
                  <div style={{ marginTop: 8 }}>
                    <div style={sectionLabelStyle}>Per-Year Stats</div>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: 'monospace' }}>
                      <thead>
                        <tr style={{ color: DIM }}>
                          <th style={{ textAlign: 'left',  paddingBottom: 3, fontWeight: 400 }}>WY</th>
                          <th style={{ textAlign: 'right', paddingBottom: 3, fontWeight: 400 }}>Scenes</th>
                          <th style={{ textAlign: 'right', paddingBottom: 3, fontWeight: 400 }}>Snow %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map(({ year, inputObs, coverage }) => (
                          <tr key={year}>
                            <td style={{ paddingBottom: 1 }}>{year}</td>
                            <td style={{ textAlign: 'right', paddingBottom: 1 }}>{inputObs ?? '—'}</td>
                            <td style={{ textAlign: 'right', paddingBottom: 1 }}>
                              {coverage !== null ? coverage.toFixed(1) + '%' : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              })()}
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

const QUERY_ROWS: { key: Variable; label: string }[] = [
  { key: 'SAD_DOWY',             label: 'Appearance' },
  { key: 'SDD_DOWY',             label: 'Disappearance' },
  { key: 'max_consec_snow_days', label: 'Duration' },
]

const PointInspectorCard = ({ right, bottom }: { right: number; bottom: number }) => {
  const clickInfo = useStore((s) => s.clickInfo)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const waterYear = WATER_YEARS[waterYearIndex]

  // Southern hemisphere pixels have lat < 0; their water year starts Apr 1 of the
  // label year rather than Oct 1 of the previous year.
  const southernHemisphere = (clickInfo?.lat ?? 0) < 0

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
          {clickInfo.status === 'querying' ? (
            <div style={{ color: DIM, fontSize: 14 }}>Querying…</div>
          ) : (
            QUERY_ROWS.map(({ key, label }) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
                <span style={{ color: DIM, fontSize: 14, flexShrink: 0 }}>{label}</span>
                <span style={{ fontFamily: 'monospace', fontSize: 16, fontWeight: 700, textAlign: 'right',
                  color: clickInfo.values[key] === null ? DIM : ACCENT }}>
                  {formatValue(clickInfo.values[key], key, waterYear, southernHemisphere)}
                </span>
              </div>
            ))
          )}
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
