/**
 * The proxy the development server uses.
 *
 * The browser talks to the API through the development server, so the
 * page and the API are on the same origin and the stream behaves the
 * same as it would behind a proxy in production.
 *
 * The address of the API is read from `API_URL`, so a machine where
 * the API is not on the default port needs nothing but the variable:
 *
 *     API_URL=http://127.0.0.1:8010 npm start
 */
const target = process.env['API_URL'] ?? 'http://127.0.0.1:8000';

const routes = ['/chat', '/memory', '/health'];

module.exports = Object.fromEntries(
  routes.map((route) => [
    route,
    { target, secure: false, changeOrigin: true },
  ]),
);
