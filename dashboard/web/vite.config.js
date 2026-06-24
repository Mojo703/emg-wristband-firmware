import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// `npm run dev` serves on 5173 and proxies the WebSocket to the axum backend.
// `npm run build` emits static assets to dist/, which the backend serves itself.
export default defineConfig({
  plugins: [svelte()],
  server: {
    proxy: {
      '/ws': { target: 'ws://localhost:8090', ws: true },
    },
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 600 },
});
