import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: {
    buildActivity: false,
    buildActivityPosition: "bottom-right"
  },
  experimental: {
    disableStaticRouteIndicator: true
  }
};

export default nextConfig;
