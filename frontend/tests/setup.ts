import { config } from '@vue/test-utils'
import { vi } from 'vitest'

// Mock navigator.mediaDevices globally
Object.defineProperty(global.navigator, 'mediaDevices', {
  writable: true,
  value: {
    getUserMedia: vi.fn(() => Promise.resolve({})),
  },
})

config.global.mocks = {
  $t: (msg: string) => msg,
}
