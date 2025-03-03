/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: {
    buildActivity: false,
    buildActivityPosition: "bottom-right"
  },
  experimental: {
    disableStaticRouteIndicator: true
  },
  
  // Add webpack configuration for Monaco Editor
  webpack: (config, { isServer }) => {
    // Prevent Monaco Editor from loading during server-side rendering
    if (isServer) {
      // Add external dependencies that should not be bundled
      config.externals = [...(config.externals || []), { 'monaco-editor': 'commonjs monaco-editor' }];
    }
    
    return config;
  }
};

module.exports = nextConfig; 