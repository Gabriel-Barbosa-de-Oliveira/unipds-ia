import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { resolveBasePath } from "./src/lib/base-path.ts";

/**
 * Caminho base: `/opspilot/` no local (spec 015, FR-025); o Pages define `OPSPILOT_WEB_BASE`
 * (`/<repo>/opspilot/`, spec 016 research.md item 6). Valor inválido faz o build falhar.
 */
export default defineConfig({
  base: resolveBasePath(process.env.OPSPILOT_WEB_BASE),
  plugins: [react()],
  server: { port: 5173 },
  preview: { port: 4173 },
});
