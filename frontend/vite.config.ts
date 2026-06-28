import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Two entry points (paths are relative to the Vite project root):
//   /          -> index.html, redirects to the 3D landing in public/chaturvyuha-site/
//   /app/      -> app/index.html, the React upload/results demo (src/main.tsx)
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: {
    rollupOptions: {
      input: {
        main: "index.html",
        app: "app/index.html",
      },
    },
  },
});
