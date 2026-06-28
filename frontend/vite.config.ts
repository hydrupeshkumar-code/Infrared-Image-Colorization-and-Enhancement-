import { defineConfig } from "vite";

// The landing is a static, self-contained site in public/chaturvyuha-site/.
// index.html just redirects to it; Vite only needs to serve / copy public/.
export default defineConfig({
  server: { port: 5173 },
});
