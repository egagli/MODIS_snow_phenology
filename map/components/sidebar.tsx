import React, { ReactNode, useEffect, useRef, useState, CSSProperties } from 'react'
import { Box, Flex } from 'theme-ui'
import { Filter, Slider, Colorbar, Input } from '@carbonplan/components'
import { useThemedColormap } from '@carbonplan/colormaps'
import {
  useStore,
  VARIABLE_CONFIGS,
  WATER_YEARS,
  COLORMAPS,
  type Variable,
} from '../lib/store'
import { track, trackDebounced } from '../lib/analytics'
import { buildShareUrl } from '../lib/url-state'
import { SHEET, SHEET_MQ, useTone } from '../lib/sheet'

export const SIDEBAR_WIDTH = 380
const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const TEXT = '#d0d0d0'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.06em',
  color: DIM,
  fontWeight: 600,
  marginBottom: 8,
}

// "Share this view": copies a link to the map's page on egagli.github.io that
// reopens it with the current view (see lib/url-state.ts). If the clipboard is
// unavailable, the link is shown selected in a read-only field instead.
const ShareButton = () => {
  const { sheet } = useTone(DIM)
  const [status, setStatus] = useState<'idle' | 'copied' | 'fallback'>('idle')
  const [url, setUrl] = useState('')
  const fieldRef = useRef<HTMLInputElement>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout>>()

  useEffect(() => () => clearTimeout(timerRef.current), [])
  useEffect(() => {
    if (status === 'fallback') fieldRef.current?.select()
  }, [status, url])

  const share = () => {
    const link = buildShareUrl()
    setUrl(link)
    const fallback = () => {
      setStatus('fallback')
      track('map_share', { result: 'fallback' })
    }
    try {
      navigator.clipboard
        .writeText(link)
        .then(() => {
          setStatus('copied')
          track('map_share', { result: 'copied' })
          clearTimeout(timerRef.current)
          timerRef.current = setTimeout(() => setStatus('idle'), 2000)
        })
        .catch(fallback)
    } catch {
      fallback()
    }
  }

  return (
    <div style={{ marginTop: 10 }}>
      <button
        onClick={share}
        title='Copy a link that reopens the map with this view'
        style={{
          padding: sheet ? '0 16px' : '4px 10px',
          minHeight: sheet ? 44 : undefined,
          borderRadius: 4,
          border: `1px solid ${status === 'copied' ? ACCENT : BORDER}`,
          background: status === 'copied' ? ACCENT : 'transparent',
          color: status === 'copied' ? '#fff' : TEXT,
          cursor: 'pointer',
          fontSize: sheet ? 14 : 11,
          transition: 'background 0.15s, color 0.15s',
        }}
      >
        {status === 'copied' ? 'Link copied' : 'Share this view'}
      </button>
      {status === 'fallback' && (
        <input
          ref={fieldRef}
          readOnly
          value={url}
          aria-label='Link to this view'
          onFocus={(e) => e.currentTarget.select()}
          style={{
            display: 'block',
            width: '100%',
            marginTop: 6,
            background: '#1e2128',
            color: TEXT,
            border: `1px solid ${BORDER}`,
            borderRadius: 4,
            fontSize: sheet ? 16 : 11,
            padding: '4px 6px',
            fontFamily: 'monospace',
            boxSizing: 'border-box',
          }}
        />
      )}
    </div>
  )
}

const VARIABLE_LABELS: Record<Variable, string> = {
  SAD_DOWY: 'Appearance',
  SDD_DOWY: 'Disappearance',
  max_consec_snow_days: 'Duration',
}

/** The sidebar's sections. `extras` (the basemap/projection card) is placed
 *  before About in the sheet; the desktop card ignores it. */
const SidebarContent = ({ extras }: { extras?: ReactNode } = {}) => {
  const { sheet, dim, fs } = useTone(DIM)
  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const opacity = useStore((s) => s.opacity)
  const clim = useStore((s) => s.clim)
  const colormap = useStore((s) => s.colormap)
  const setVariable = useStore((s) => s.setVariable)
  const setWaterYearIndex = useStore((s) => s.setWaterYearIndex)
  const setOpacity = useStore((s) => s.setOpacity)
  const setClim = useStore((s) => s.setClim)
  const setColormap = useStore((s) => s.setColormap)

  const themedColormap = useThemedColormap(colormap)
  const [climInputs, setClimInputs] = useState<[string, string]>([
    String(clim[0]),
    String(clim[1]),
  ])

  useEffect(() => {
    setClimInputs([String(clim[0]), String(clim[1])])
  }, [clim])

  const commitClim = (index: 0 | 1, value?: string) => {
    const val = parseFloat(value ?? climInputs[index])
    if (Number.isFinite(val)) {
      setClim(index === 0 ? [val, clim[1]] : [clim[0], val])
    } else {
      setClimInputs([String(clim[0]), String(clim[1])])
    }
  }

  const handleClimInput = (index: 0 | 1, newValue: string) => {
    const newNum = parseFloat(newValue)
    const isArrow = Number.isFinite(newNum) && Math.abs(newNum - clim[index]) <= 1.01
    if (isArrow) {
      commitClim(index, newValue)
    } else {
      setClimInputs(index === 0 ? [newValue, climInputs[1]] : [climInputs[0], newValue])
    }
  }

  const label: CSSProperties = { ...sectionLabelStyle, fontSize: fs(10), color: dim }
  const inputSx = sheet
    ? { fontSize: '16px', minHeight: '44px', width: '88px', textAlign: 'center' as const }
    : {}

  const about = (
    <div key='about' style={sheet ? { borderTop: `1px solid ${BORDER}`, paddingTop: 16 } : undefined}>
      {sheet ? (
        <div style={label}>ABOUT</div>
      ) : (
        <Box
          as='h1'
          sx={{
            fontSize: [2, 2, 3, 3],
            fontFamily: 'heading',
            letterSpacing: 'heading',
            lineHeight: 'heading',
            color: 'primary',
            m: 0,
            mb: '4px',
          }}
        >
          MODIS snow phenology
        </Box>
      )}
      <div style={{ fontSize: sheet ? 13 : 11, color: dim, lineHeight: 1.6, marginBottom: 4 }}>
        A global, 500 meter resolution snow phenology dataset derived from MODIS MOD10A2 8-day maximum snow extent covering water years 2015 to 2025. Each pixel stores three annual metrics: snow appearance date [DOWY], snow disappearance date [DOWY], and longest continuous snow period [days]. Cloud gaps are filled using a bidirectional temporal interpolation method similar to Wrzesien et al. (2019), and polar-night periods are corrected before cloud-filling to prevent false no-snow classifications at high latitudes. For more information, please check out the github repo linked below.
      </div>
      <a
        href='https://github.com/egagli/MODIS_snow_phenology'
        target='_blank'
        rel='noopener noreferrer'
        style={{
          fontSize: sheet ? 14 : 11,
          color: ACCENT,
          textDecoration: 'none',
          ...(sheet && { display: 'flex', alignItems: 'center', minHeight: 44 }),
        }}
      >
        egagli/MODIS_snow_phenology ↗
      </a>
      <ShareButton />
    </div>
  )

  const variablePicker = (
    <div key='variable'>
      <div style={label}>variable</div>
      <Box
        sx={
          sheet
            ? { '& button': { fontSize: '13px', letterSpacing: '0.03em', minHeight: '44px', mr: '8px' } }
            : undefined
        }
      >
        <Filter
          values={Object.fromEntries(
            (Object.keys(VARIABLE_LABELS) as Variable[]).map((v) => [
              VARIABLE_LABELS[v],
              v === variable,
            ])
          )}
          setValues={(obj: Record<string, boolean>) => {
            const entry = (Object.entries(VARIABLE_LABELS) as [Variable, string][]).find(
              ([, label]) => obj[label]
            )
            if (entry && entry[0] !== variable) {
              setVariable(entry[0])
              track('map_layer_change', { layer: entry[0] })
            }
          }}
        />
      </Box>
      <div style={{ fontSize: fs(11), color: dim, marginTop: 4 }}>
        {VARIABLE_CONFIGS[variable].label} [{VARIABLE_CONFIGS[variable].units}]
      </div>
    </div>
  )

  const pickYear = (i: number) => {
    setWaterYearIndex(i)
    trackDebounced('map_time_change', { water_year: WATER_YEARS[i] })
  }
  const waterYear = (
    <div key='year'>
      <div style={{ ...label, marginBottom: 4 }}>
        water year —{' '}
        <span style={{ color: TEXT, fontWeight: 700 }}>{WATER_YEARS[waterYearIndex]}</span>
      </div>
      <Slider
        min={0}
        max={WATER_YEARS.length - 1}
        step={1}
        value={waterYearIndex}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) => pickYear(parseInt(e.target.value))}
      />
      {sheet ? (
        // touch: a grid of year chips (the slider stays for scrubbing)
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 8, marginTop: 12 }}>
          {WATER_YEARS.map((year, i) => (
            <button
              key={year}
              onClick={() => pickYear(i)}
              aria-pressed={i === waterYearIndex}
              style={{
                minHeight: 44, borderRadius: 8, cursor: 'pointer', fontFamily: 'monospace', fontSize: 14,
                border: `1px solid ${i === waterYearIndex ? ACCENT : BORDER}`,
                background: i === waterYearIndex ? 'rgba(29,189,143,0.15)' : 'transparent',
                color: i === waterYearIndex ? ACCENT : TEXT,
                fontWeight: i === waterYearIndex ? 700 : 400,
              }}
            >
              {year}
            </button>
          ))}
        </div>
      ) : (
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          {WATER_YEARS.map((year, i) => (
            <button
              key={year}
              onClick={() => pickYear(i)}
              style={{
                background: 'none', border: 'none', padding: 0, margin: 0,
                fontSize: 9, lineHeight: 1, cursor: 'pointer', fontFamily: 'monospace',
                color: i === waterYearIndex ? ACCENT : DIM,
              }}
            >
              {year}
            </button>
          ))}
        </div>
      )}
    </div>
  )

  // In the sheet the header shows the colorbar, so the colormap and range
  // sections don't repeat it.
  const colormapPicker = (
    <div key='colormap'>
      <div style={label}>colormap</div>
      <select
        value={colormap}
        onChange={(e) => setColormap(e.target.value)}
        style={{
          width: '100%',
          background: '#1e2128',
          color: TEXT,
          border: `1px solid ${BORDER}`,
          borderRadius: 4,
          fontSize: sheet ? 16 : 12,
          minHeight: sheet ? 44 : undefined,
          padding: '5px 8px',
          cursor: 'pointer',
          fontFamily: 'monospace',
          marginBottom: sheet ? 0 : 8,
        }}
      >
        {COLORMAPS.map((c) => (
          <option key={c} value={c}>{c}</option>
        ))}
      </select>
      {!sheet && <Colorbar width='100%' colormap={themedColormap} horizontal />}
    </div>
  )

  const range = (
    <div key='range'>
      <div style={label}>range</div>
      <Flex sx={{ gap: 2, alignItems: 'center' }}>
        <Input
          size='xs'
          type='number'
          value={climInputs[0]}
          aria-label='Colour range minimum'
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(0, e.target.value)}
          onBlur={() => commitClim(0)}
          onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(0) }}
          sx={{ width: `${Math.max(2, climInputs[0].length + 2)}ch`, ...inputSx }}
        />
        <Box sx={{ flex: 1 }}>
          {!sheet && <Colorbar width='100%' colormap={themedColormap} horizontal />}
        </Box>
        <Input
          size='xs'
          type='number'
          value={climInputs[1]}
          aria-label='Colour range maximum'
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(1, e.target.value)}
          onBlur={() => commitClim(1)}
          onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(1) }}
          sx={{ width: `${Math.max(2, climInputs[1].length + 2)}ch`, ...inputSx }}
        />
      </Flex>
    </div>
  )

  const opacityPicker = (
    <div key='opacity'>
      <div style={label}>opacity</div>
      <Slider
        min={0}
        max={1}
        step={0.01}
        value={opacity}
        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
          setOpacity(parseFloat(e.target.value))
        }
      />
    </div>
  )

  if (sheet) {
    // Sheet order: controls first, then the extras (basemap/projection), About last.
    return (
      <>
        {variablePicker}
        {waterYear}
        {range}
        {colormapPicker}
        {opacityPicker}
        {extras}
        {about}
      </>
    )
  }

  return (
    <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 16, color: TEXT, fontSize: 13 }}>
      {about}
      <div style={{ borderTop: `1px solid ${BORDER}` }} />
      {variablePicker}
      {waterYear}
      {colormapPicker}
      {range}
      {opacityPicker}
    </div>
  )
}

/** Sheet header subtitle: active variable and water year. */
export const SheetSubtitle = () => {
  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  return <>{VARIABLE_CONFIGS[variable].label} · WY {WATER_YEARS[waterYearIndex]}</>
}

/** Sheet header legend: the active colormap with its range and units. */
export const CompactLegend = () => {
  const variable = useStore((s) => s.variable)
  const clim = useStore((s) => s.clim)
  const colormap = useStore((s) => s.colormap)
  const themedColormap = useThemedColormap(colormap)
  const units = VARIABLE_CONFIGS[variable].units === 'days' ? 'days' : 'DOWY'
  const num: CSSProperties = { font: '12px/12px monospace', color: SHEET.text, whiteSpace: 'nowrap', flexShrink: 0 }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={num}>{clim[0]}</span>
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Colorbar width='100%' height='12px' colormap={themedColormap} horizontal />
      </Box>
      <span style={num}>{clim[1]} {units}</span>
    </div>
  )
}

export { SidebarContent }

export const Sidebar = () => {
  const setSidebarWidth = useStore((s) => s.setSidebarWidth)

  useEffect(() => {
    setSidebarWidth(SIDEBAR_WIDTH + 32)
    return () => setSidebarWidth(0)
  }, [setSidebarWidth])

  // Desktop: floating card (overlays map, no left offset on map). Phones and
  // tablets get SidebarContent inside the MobileSheet instead (pages/index.tsx).
  return (
    <Box
      sx={{
        display: 'block',
        [SHEET_MQ]: { display: 'none' },
        position: 'absolute',
        top: 16,
        left: 16,
        width: SIDEBAR_WIDTH,
        maxHeight: 'calc(100vh - 32px)',
        bg: BG,
        border: `1px solid ${BORDER}`,
        borderRadius: 8,
        backdropFilter: 'blur(6px)',
        overflowY: 'auto',
        zIndex: 10,
      }}
    >
      <SidebarContent />
    </Box>
  )
}
