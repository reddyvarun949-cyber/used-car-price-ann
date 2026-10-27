import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server runs on http://localhost:5173 (the origin the Flask API allows).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
