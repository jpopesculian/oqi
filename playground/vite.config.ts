import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/oqi/',
  plugins: [react()],
  // esbuild pre-bundling would relocate oqi-js and break its
  // `new URL('oqi_js_bg.wasm', import.meta.url)` wasm reference.
  optimizeDeps: { exclude: ['oqi-js'] },
  // The oqi-js dependency is a symlink to ../js/pkg, outside this root.
  server: { fs: { allow: ['..'] } },
  worker: { format: 'es' },
});
