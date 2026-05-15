import React, { useRef, useEffect, useState } from 'react'
import { Box, Flex } from 'theme-ui'
import { Sidebar as CarbonSidebar, SidebarDivider } from '@carbonplan/layouts'
import { Filter, Slider, Row, Column, Colorbar, Input } from '@carbonplan/components'
import { useThemedColormap } from '@carbonplan/colormaps'
import {
  useStore,
  VARIABLE_CONFIGS,
  WATER_YEARS,
  type Variable,
} from '../lib/store'

const headingSx = {
  fontFamily: 'heading',
  letterSpacing: 'smallcaps',
  textTransform: 'uppercase' as const,
  fontSize: [2, 2, 3, 3],
}

const subheadingSx = {
  fontFamily: 'mono',
  letterSpacing: 'mono',
  textTransform: 'uppercase' as const,
  fontSize: [1, 1, 2, 2],
  color: 'secondary',
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
  const globeProjection = useStore((s) => s.globeProjection)
  const setVariable = useStore((s) => s.setVariable)
  const setWaterYearIndex = useStore((s) => s.setWaterYearIndex)
  const setOpacity = useStore((s) => s.setOpacity)
  const setClim = useStore((s) => s.setClim)
  const setGlobeProjection = useStore((s) => s.setGlobeProjection)

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
      setClimInputs(
        index === 0 ? [newValue, climInputs[1]] : [climInputs[0], newValue]
      )
    }
  }

  return (
    <>
      <Box
        as='h1'
        sx={{
          fontSize: [3],
          fontFamily: 'heading',
          letterSpacing: 'heading',
          lineHeight: 'heading',
          mb: 1,
        }}
      >
        MODIS Snow Phenology
      </Box>
      <Box sx={{ color: 'secondary', fontSize: 1, mb: 2 }}>
        Global snow appearance, disappearance, and duration derived from MODIS
        MOD10A2 (2015–2024)
      </Box>

      <SidebarDivider sx={{ my: 3 }} />

      <Box sx={headingSx}>Variable</Box>

      <Row columns={[4, 4, 4, 4]} sx={{ mt: 2, alignItems: 'baseline' }}>
        <Column start={1} width={4}>
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
        </Column>
      </Row>

      <Box sx={{ color: 'secondary', fontSize: 1, mt: 1 }}>
        {VARIABLE_CONFIGS[variable].label} ({VARIABLE_CONFIGS[variable].units})
      </Box>

      <SidebarDivider sx={{ my: 3 }} />

      <Box sx={headingSx}>Water Year</Box>

      <Row columns={[4, 4, 4, 4]} sx={{ mt: 2, alignItems: 'center' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Year</Box>
        </Column>
        <Column start={2} width={3}>
          <Box
            sx={{
              fontFamily: 'mono',
              fontSize: [2, 2, 3, 3],
              color: 'primary',
            }}
          >
            {WATER_YEARS[waterYearIndex]}
          </Box>
        </Column>
      </Row>

      <Row columns={[4, 4, 4, 4]} sx={{ mt: 1 }}>
        <Column start={1} width={4}>
          <Flex sx={{ gap: 1, alignItems: 'center' }}>
            <Box sx={{ color: 'secondary', fontSize: 1 }}>
              {WATER_YEARS[0]}
            </Box>
            <Box sx={{ flex: 1 }}>
              <Slider
                min={0}
                max={WATER_YEARS.length - 1}
                step={1}
                value={waterYearIndex}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setWaterYearIndex(parseInt(e.target.value))
                }
              />
            </Box>
            <Box sx={{ color: 'secondary', fontSize: 1 }}>
              {WATER_YEARS[WATER_YEARS.length - 1]}
            </Box>
          </Flex>
        </Column>
      </Row>

      <SidebarDivider sx={{ my: 3 }} />

      <Box sx={headingSx}>Display</Box>

      <Row columns={[4, 4, 4, 4]} sx={{ mt: 2, alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Range</Box>
        </Column>
        <Column start={2} width={3}>
          <Flex sx={{ gap: 2, alignItems: 'center' }}>
            <Input
              size='xs'
              type='number'
              value={climInputs[0]}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleClimInput(0, e.target.value)
              }
              onBlur={() => commitClim(0)}
              onKeyDown={(e: React.KeyboardEvent) => {
                if (e.key === 'Enter') commitClim(0)
              }}
              sx={{ width: `${Math.max(2, climInputs[0].length + 2)}ch` }}
            />
            <Box sx={{ flex: 1 }}>
              <Colorbar width='100%' colormap={themedColormap} horizontal />
            </Box>
            <Input
              size='xs'
              type='number'
              value={climInputs[1]}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                handleClimInput(1, e.target.value)
              }
              onBlur={() => commitClim(1)}
              onKeyDown={(e: React.KeyboardEvent) => {
                if (e.key === 'Enter') commitClim(1)
              }}
              sx={{ width: `${Math.max(2, climInputs[1].length + 2)}ch` }}
            />
          </Flex>
        </Column>
      </Row>

      <Row columns={[4, 4, 4, 4]} sx={{ mt: 2, alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Opacity</Box>
        </Column>
        <Column start={2} width={3}>
          <Slider
            min={0}
            max={1}
            step={0.01}
            value={opacity}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
              setOpacity(parseFloat(e.target.value))
            }
          />
        </Column>
      </Row>

      <SidebarDivider sx={{ my: 3 }} />

      <Box sx={headingSx}>Map</Box>

      <Row columns={[4, 4, 4, 4]} sx={{ mt: 2, alignItems: 'baseline' }}>
        <Column start={1} width={1}>
          <Box sx={subheadingSx}>Projection</Box>
        </Column>
        <Column start={2} width={3}>
          <Filter
            values={{ globe: globeProjection, mercator: !globeProjection }}
            setValues={(obj: Record<string, boolean>) => {
              if (obj.globe) setGlobeProjection(true)
              if (obj.mercator) setGlobeProjection(false)
            }}
          />
        </Column>
      </Row>
    </>
  )
}

export const Sidebar = () => {
  const sidebarRef = useRef<HTMLDivElement>(null)
  const setSidebarWidth = useStore((s) => s.setSidebarWidth)

  useEffect(() => {
    const updateWidth = () => {
      if (sidebarRef.current) {
        const width =
          sidebarRef.current.parentElement?.parentElement?.offsetWidth ?? 0
        setSidebarWidth(width)
      }
    }
    updateWidth()
    window.addEventListener('resize', updateWidth)
    return () => {
      window.removeEventListener('resize', updateWidth)
      setSidebarWidth(0)
    }
  }, [setSidebarWidth])

  return (
    <>
      {/* Desktop sidebar */}
      <Box sx={{ display: ['none', 'none', 'block'] }}>
        <CarbonSidebar expanded={true} side='left' width={4}>
          <div ref={sidebarRef}>
            <SidebarContent />
          </div>
        </CarbonSidebar>
      </Box>

      {/* Mobile bottom panel */}
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
