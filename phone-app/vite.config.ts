import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import path from 'node:path';

// `pnpm run dev` serves on localhost, which browsers treat as a secure context,
// so Web Bluetooth works there without a certificate. A phone needs HTTPS, so
// the phone path is `pnpm run build` and a static host (see README.md).
export default defineConfig({
  plugins: [svelte()],
  resolve: {
    alias: {
      $lib: path.resolve('./src/lib'),
    },
  },
  build: { outDir: 'dist' },
});
