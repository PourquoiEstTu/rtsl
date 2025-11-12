import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config'

export default mergeConfig(viteConfig, defineConfig({
  test: {
    environment: 'jsdom',
    exclude: ['packages/template/*'],
    include: ['src/**/*.spec.ts'],
    globals: true,
    setupFiles: './tests/setup.ts',
    coverage: {
      reporter: ['text', 'html']
    }
  },
}))