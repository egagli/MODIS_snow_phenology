import React, { useEffect, useState, CSSProperties } from 'react'
import { Box } from 'theme-ui'
import { Filter, Slider, Colorbar, Input } from '@carbonplan/components'
import { useThemedColormap } from '@carbonplan/colormaps'
import {
  useStore,
  VARIABLE_CONFIGS,
  WATER_YEARS,
  type Variable,
} from '../lib/store'

const COLORMAPS = [
  'rainbow', 'fire', 'earth', 'water', 'warm', 'cool',
  'reds', 'oranges', 'yellows', 'greens', 'teals', 'blues',
  'purples', 'pinks', 'greys', 'wind', 'heart', 'sinebow',
  'pinkgreen', 'redteal', 'orangeblue', 'yellowpurple',
]

const SIDEBAR_WIDTH = 380
const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'

const dividerStyle: CSSProperties = {
  borderTop: `1px solid ${BORDER}`,
  margin: '12px 0',
}

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  color: DIM,
  fontWeight: 600,
  marginBottom: 8,
}

const rowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  marginBottom: 8,
}

const labelStyle: CSSProperties = {
  fontSize: 12,
  color: DIM,
  minWidth: 64,
  flexShrink: 0,
}

const VARIABLE_LABELS: Record<Variable, string> = {
  SAD_DOWY: 'Appearance',
  SDD_DOWY: 'Disappearance',
  max_consec_snow_days: 'Duration',
}

const SidebarContent = () => {
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

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 4, fontSize: 18, fontFamily: 'heading', fontWeight: 600, color: '#e2e8f0', letterSpacing: '0.01em' }}>
        MODIS Snow Phenology
      </div>
      <div style={{ fontSize: 12, color: DIM, marginBottom: 4, lineHeight: 1.5 }}>
        Global snow appearance, disappearance, and duration from MODIS MOD10A2 (2015–2024).
      </div>
      <div style={{ fontSize: 12, marginBottom: 12 }}>
        <a
          href='https://github.com/egagli/MODIS_snow_phenology'
          target='_blank'
          rel='noopener noreferrer'
          style={{ color: ACCENT, textDecoration: 'none' }}
        >
          egagli/MODIS_snow_phenology ↗
        </a>
      </div>

      <div style={dividerStyle} />

      {/* Variable */}
      <div style={sectionLabelStyle}>Variable</div>
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
          if (entry) setVariable(entry[0])
        }}
      />
      <div style={{ fontSize: 11, color: DIM, marginTop: 4, marginBottom: 0 }}>
        {VARIABLE_CONFIGS[variable].label} ({VARIABLE_CONFIGS[variable].units})
      </div>

      <div style={dividerStyle} />

      {/* Water Year */}
      <div style={sectionLabelStyle}>Water Year</div>
      <div style={{ ...rowStyle, marginBottom: 4 }}>
        <span style={{ ...labelStyle, minWidth: 'auto' }}>Year</span>
        <span style={{ fontFamily: 'monospace', fontSize: 16, color: '#e2e8f0', fontWeight: 600 }}>
          {WATER_YEARS[waterYearIndex]}
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 11, color: DIM }}>{WATER_YEARS[0]}</span>
        <div style={{ flex: 1 }}>
          <Slider
            min={0}
            max={WATER_YEARS.length - 1}
            step={1}
            value={waterYearIndex}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setWaterYearIndex(parseInt(e.target.value))
            }
          />
        </div>
        <span style={{ fontSize: 11, color: DIM }}>{WATER_YEARS[WATER_YEARS.length - 1]}</span>
      </div>

      <div style={dividerStyle} />

      {/* Display */}
      <div style={sectionLabelStyle}>Display</div>

      <div style={{ ...rowStyle, alignItems: 'flex-start' }}>
        <span style={labelStyle}>Colormap</span>
        <div style={{ flex: 1 }}>
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value)}
            style={{
              width: '100%',
              background: '#1b1e23',
              color: '#d0d0d0',
              border: `1px solid ${BORDER}`,
              borderRadius: 4,
              fontSize: 12,
              padding: '3px 6px',
              cursor: 'pointer',
              fontFamily: 'monospace',
              marginBottom: 4,
            }}
          >
            {COLORMAPS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <Colorbar width='100%' colormap={themedColormap} horizontal />
        </div>
      </div>

      <div style={rowStyle}>
        <span style={labelStyle}>Range</span>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Input
            size='xs'
            type='number'
            value={climInputs[0]}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(0, e.target.value)}
            onBlur={() => commitClim(0)}
            onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(0) }}
            sx={{ width: `${Math.max(2, climInputs[0].length + 2)}ch` }}
          />
          <div style={{ flex: 1 }}>
            <Colorbar width='100%' colormap={themedColormap} horizontal />
          </div>
          <Input
            size='xs'
            type='number'
            value={climInputs[1]}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(1, e.target.value)}
            onBlur={() => commitClim(1)}
            onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(1) }}
            sx={{ width: `${Math.max(2, climInputs[1].length + 2)}ch` }}
          />
        </div>
      </div>

      <div style={rowStyle}>
        <span style={labelStyle}>Opacity</span>
        <div style={{ flex: 1 }}>
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
      </div>
    </div>
  )
}

export const Sidebar = () => {
  const setSidebarWidth = useStore((s) => s.setSidebarWidth)

  useEffect(() => {
    setSidebarWidth(SIDEBAR_WIDTH + 32)
    return () => setSidebarWidth(0)
  }, [setSidebarWidth])

  return (
    <>
      {/* Desktop: compact floating card */}
      <Box
        sx={{
          display: ['none', 'none', 'block'],
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
          p: '16px',
        }}
      >
        <SidebarContent />
      </Box>

      {/* Mobile: bottom panel */}
      <Box
        sx={{
          display: ['block', 'block', 'none'],
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '50vh',
          bg: 'background',
          overflowY: 'auto',
          zIndex: 1000,
          px: [4, 5],
          py: [3],
          borderTop: '1px solid',
          borderColor: 'muted',
        }}
      >
        <SidebarContent />
      </Box>
    </>
  )
}
