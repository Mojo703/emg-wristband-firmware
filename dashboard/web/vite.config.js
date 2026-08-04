import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

// `pnpm run dev` serves on 5173 and proxies the WebSocket and the backend's HTTP
// routes to axum. `pnpm run build` emits static assets to dist/, which the backend
// serves itself.
export default defineConfig({
  plugins: [tailwindcss(), svelte()],
  resolve: {
    alias: {
      $lib: path.resolve('./src/lib'),
    },
  },
  server: {
    proxy: {
      '/ws': { target: 'ws://localhost:8090', ws: true },
      // Track audio, map import and track deletion all live under this prefix.
      '/collection': { target: 'http://localhost:8090' },
    },
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 600 },
});
