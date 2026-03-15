import { VitePWA } from 'vite-plugin-pwa';
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import type { PluginOption } from 'vite'
import mkcert from 'vite-plugin-mkcert'
import tailwindcss from '@tailwindcss/vite'
import path from 'path';
import { viteStaticCopy } from 'vite-plugin-static-copy';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    mkcert(),
    tailwindcss(),
    viteStaticCopy({
      targets: [
        {
          src: 'src/extension/manifest.json',
          dest: ''
        },
        {
          src: 'src/extension/background/service-worker.ts',
          dest: 'background'
        }
      ]
    }),
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: false,

      pwaAssets: {
        disabled: false,
        config: true,
      },

      manifest: {
        name: 'rtsl-frontend',
        short_name: 'rtsl-frontend',
        description: 'Frontend for rtsl, real-time (American) sign language translation',
        theme_color: '#ffffff',
      },

      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,ico}'],
        cleanupOutdatedCaches: true,
        clientsClaim: true,
      },

      devOptions: {
        enabled: false,
        navigateFallback: 'index.html',
        suppressWarnings: true,
        type: 'module',
      }
    }),
  ] as PluginOption[],
  server: {
    host: '0.0.0.0',
    port: 5173,
  },
  resolve: {
    alias: [
      { find: "@", replacement: path.resolve(__dirname, "./src") }
    ]
  },
  build: {
    rollupOptions: {
      input: {
        index: path.resolve(__dirname, 'index.html'),
        popup: path.resolve(__dirname, 'src/extension/popup.html'),
        translator: path.resolve(__dirname, 'src/extension/translator.html'),
        serviceWorker: path.resolve(__dirname, 'src/extension/background/service-worker.ts'),
      }
    }
  }
})