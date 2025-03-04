/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: {
    buildActivity: false,
    buildActivityPosition: "bottom-right"
  },
  experimental: {
    disableStaticRouteIndicator: true,
    staticRouteIndicator: false
  }
};

module.exports = nextConfig; 