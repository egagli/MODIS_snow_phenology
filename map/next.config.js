const path = require('path')

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/MODIS_snow_phenology',
  assetPrefix: '/MODIS_snow_phenology',
  reactStrictMode: true,
  // Compile zarr-layer TypeScript source directly via SWC so local dev/build
  // works without a separate `npm run build` step in the zarr-layer repo.
  // The webpack alias bypasses the package's exports field (which points to
  // the non-existent dist/) and sends imports straight to the TypeScript source.
  transpilePackages: ['@carbonplan/zarr-layer'],
  webpack: (config) => {
    config.resolve.alias['@carbonplan/zarr-layer'] = path.resolve(
      __dirname,
      '../../zarr-layer/src/index.ts'
    )
    return config
  },
}

module.exports = nextConfig
