import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Turbopack is the default bundler in Next.js 16.
  // It handles maplibre-gl's ESM worker imports (new URL pattern) natively —
  // no custom webpack rules needed.
  turbopack: {},
};

export default nextConfig;
