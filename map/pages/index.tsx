import React, { useEffect } from 'react'
import { Box } from 'theme-ui'
import { Map } from '../components/map'
import { Sidebar } from '../components/sidebar'
import { FloatingCards } from '../components/floating-cards'
import { useStore } from '../lib/store'
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
      }}
    >
      <Sidebar />
      {sidebarWidth !== null && (
        <>
          <Map />
          <Box sx={{ display: ['none', 'none', 'block'] }}>
            <FloatingCards />
          </Box>
        </>
      )}
    </Box>
  )
}
