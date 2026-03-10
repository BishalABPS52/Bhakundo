/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['i.ibb.co'],
  },
  async rewrites() {
    return [
      {
        source: '/api/predict',
        destination: 'http://localhost:8000/api/predict',
      },
      {
        source: '/api/teams',
        destination: 'http://localhost:8000/api/teams',
      },
      {
        source: '/api/team-stats/:team',
        destination: 'http://localhost:8000/api/team-stats/:team',
      },
      {
        source: '/api/model-info',
        destination: 'http://localhost:8000/api/model-info',
      },
    ];
  },
}

module.exports = nextConfig
