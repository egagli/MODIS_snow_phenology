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
    // Alias zarrita and its storage sub-package to zarr-layer's copies so map.tsx
    // can import zarrita directly without adding it as a separate map dependency.
    const zarritaRoot = path.resolve(__dirname, '../../zarr-layer/node_modules/zarrita/dist/src')
    const zarritaStorageRoot = path.resolve(__dirname, '../../zarr-layer/node_modules/@zarrita/storage/dist/src')
    config.resolve.alias['zarrita'] = path.join(zarritaRoot, 'index.js')
    config.resolve.alias['@zarrita/storage/fetch'] = path.join(zarritaStorageRoot, 'fetch.js')
    return config
  },
}

module.exports = nextConfig
