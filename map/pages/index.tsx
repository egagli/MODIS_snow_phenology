import React, { useEffect } from 'react'
import { Box } from 'theme-ui'
import { Map } from '../components/map'
import { Sidebar, SidebarContent, SheetSubtitle, CompactLegend } from '../components/sidebar'
import {
  FloatingCards,
  CombinedInspectorCard,
  OpeningNotice,
  TopRightCard,
  WarningToast,
} from '../components/floating-cards'
import { MobileSheet } from '../components/mobile-sheet'
import { SearchBox } from '../components/search'
import { useStore } from '../lib/store'
import { SHEET_MQ } from '../lib/sheet'
import {
  readUrlState,
  applyUrlStateToStore,
  scheduleUrlUpdate,
  writeUrlNow,
  postStateToParent,
  reloadIntoNewHash,
} from '../lib/url-state'

export default function Home() {
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  // a new point query or processing-grid tile opens the phone/tablet sheet
  const queryKey = useStore((s) =>
    s.clickInfo ? `${s.clickInfo.lat},${s.clickInfo.lng}` : s.tileClickInfo ? String(s.tileClickInfo.tile) : ''
  )

  // Restore the view from the URL, then keep the URL in step with the store.
  // Runs before the map mounts (it waits for sidebarWidth), so the map and its
  // layers are created with the restored settings.
  useEffect(() => {
    applyUrlStateToStore(readUrlState())
    if (!writeUrlNow()) postStateToParent()
    const unsubscribe = useStore.subscribe(() => scheduleUrlUpdate())
    // replaceState never fires hashchange, so this is someone editing or
    // following a link to a different view: reload to restore it cleanly.
    window.addEventListener('hashchange', reloadIntoNewHash)
    return () => {
      unsubscribe()
      window.removeEventListener('hashchange', reloadIntoNewHash)
    }
  }, [])

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
        overflow: 'hidden',
        // the sheet slides off the bottom with `transform`; `clip` (unlike
        // `hidden`) also stops the browser from scrolling this box to it
        [SHEET_MQ]: { overflow: 'clip' },
      }}
    >
      <Sidebar />
      {sidebarWidth !== null && (
        <>
          <Map />
          <SearchBox />
          {/* Desktop: floating cards. Phones/tablets: one bottom sheet. */}
          <Box sx={{ display: 'block', [SHEET_MQ]: { display: 'none' } }}>
            <FloatingCards />
          </Box>
          <MobileSheet
            title='MODIS snow phenology'
            subtitle={<SheetSubtitle />}
            legend={<CompactLegend />}
            queryKey={queryKey}
          >
            <CombinedInspectorCard />
            <OpeningNotice variant='inline' />
            <SidebarContent extras={<TopRightCard />} />
          </MobileSheet>
          <Box sx={{ display: 'none', [SHEET_MQ]: { display: 'block' } }}>
            <WarningToast variant='banner' />
          </Box>
        </>
      )}
    </Box>
  )
}
