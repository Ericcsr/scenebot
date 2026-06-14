import { defineConfig } from 'vite';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Production build for the full-browser scenebot demo (no backend).
// Differs from the default `vite.config.js` in two ways:
//   - input is index-fullbrowser.html (not index.html which is data-mode="spawn")
//   - publicDir points at public/scenebot specifically — the rest of public/
//     (parkour ONNX files, dev-mode symlinks like meshes/) MUST NOT ship.
// publicDir contents are copied flat into dist-desktop/, so to keep the prod
// fetch paths under "scenebot/" we mirror that layout in dist via the build.
// We can't directly tell vite "copy public/scenebot/ into dist/scenebot/"
// without using a plugin, so we set publicDir: false here and copy the
// scenebot subtree post-build (handled by the npm script).
export default defineConfig({
  base: './',
  publicDir: false,
  build: {
    target: 'esnext',
    outDir: 'dist-desktop',
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(__dirname, 'index-fullbrowser.html'),
    },
  },
});
