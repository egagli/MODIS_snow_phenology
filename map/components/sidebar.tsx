import React, { useEffect, useState, CSSProperties } from 'react'
import { Box, Flex } from 'theme-ui'
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

export const SIDEBAR_WIDTH = 380
const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const TEXT = '#d0d0d0'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  color: DIM,
  fontWeight: 600,
  marginBottom: 8,
}

const dividerStyle: CSSProperties = {
  borderTop: `1px solid ${BORDER}`,
  margin: '14px 0',
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
    <>
      {/* Header */}
      <div style={{ padding: '20px 20px 14px', borderBottom: `1px solid ${BORDER}` }}>
        <Box
          as='h1'
          sx={{
            fontSize: [2, 2, 3, 3],
            fontFamily: 'heading',
            letterSpacing: 'heading',
            lineHeight: 'heading',
            color: 'primary',
            m: 0,
            mb: '6px',
          }}
        >
          MODIS Snow Phenology
        </Box>
        <div style={{ fontSize: 11, color: DIM, lineHeight: 1.6, marginBottom: 4 }}>
          Global snow appearance, disappearance, and duration derived from MODIS MOD10A2 (2015–2024).
        </div>
        <a
          href='https://github.com/egagli/MODIS_snow_phenology'
          target='_blank'
          rel='noopener noreferrer'
          style={{ fontSize: 11, color: ACCENT, textDecoration: 'none' }}
        >
          egagli/MODIS_snow_phenology ↗
        </a>
      </div>

      {/* Controls */}
      <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 18 }}>

        {/* Variable */}
        <div>
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
          <div style={{ fontSize: 11, color: DIM, marginTop: 4 }}>
            {VARIABLE_CONFIGS[variable].label} ({VARIABLE_CONFIGS[variable].units})
          </div>
        </div>

        {/* Water Year */}
        <div>
          <div style={{ ...sectionLabelStyle, marginBottom: 4 }}>
            Water Year —{' '}
            <span style={{ color: TEXT, fontWeight: 700 }}>{WATER_YEARS[waterYearIndex]}</span>
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
        </div>

        {/* Colormap */}
        <div>
          <div style={sectionLabelStyle}>Colormap</div>
          <select
            value={colormap}
            onChange={(e) => setColormap(e.target.value)}
            style={{
              width: '100%',
              background: '#1e2128',
              color: TEXT,
              border: `1px solid ${BORDER}`,
              borderRadius: 4,
              fontSize: 12,
              padding: '5px 8px',
              cursor: 'pointer',
              fontFamily: 'monospace',
              marginBottom: 8,
            }}
          >
            {COLORMAPS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <Colorbar width='100%' colormap={themedColormap} horizontal />
        </div>

        {/* Range */}
        <div>
          <div style={sectionLabelStyle}>Range</div>
          <Flex sx={{ gap: 2, alignItems: 'center' }}>
            <Input
              size='xs'
              type='number'
              value={climInputs[0]}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(0, e.target.value)}
              onBlur={() => commitClim(0)}
              onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(0) }}
              sx={{ width: `${Math.max(2, climInputs[0].length + 2)}ch` }}
            />
            <Box sx={{ flex: 1 }}>
              <Colorbar width='100%' colormap={themedColormap} horizontal />
            </Box>
            <Input
              size='xs'
              type='number'
              value={climInputs[1]}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(1, e.target.value)}
              onBlur={() => commitClim(1)}
              onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(1) }}
              sx={{ width: `${Math.max(2, climInputs[1].length + 2)}ch` }}
            />
          </Flex>
        </div>

        {/* Opacity */}
        <div>
          <div style={sectionLabelStyle}>Opacity</div>
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
    </>
  )
}

export const Sidebar = () => {
  const setSidebarWidth = useStore((s) => s.setSidebarWidth)

  useEffect(() => {
    setSidebarWidth(SIDEBAR_WIDTH)
    return () => setSidebarWidth(0)
  }, [setSidebarWidth])

  return (
    <>
      {/* Desktop: full-height flush left panel (overlays map, same as demo) */}
      <Box
        sx={{
          display: ['none', 'none', 'flex'],
          flexDirection: 'column',
          position: 'absolute',
          top: 0,
          left: 0,
          width: SIDEBAR_WIDTH,
          height: '100%',
          bg: BG,
          borderRight: `1px solid ${BORDER}`,
          backdropFilter: 'blur(6px)',
          overflowY: 'auto',
          zIndex: 10,
          color: TEXT,
          fontSize: '13px',
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
