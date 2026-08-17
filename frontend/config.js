/**
 * Frontend configuration.
 *
 * Set this to your Render service URL when the dashboard is hosted separately
 * (GitHub Pages) from the API (Render). Include the scheme, no trailing slash:
 *
 *   window.NUMBER_PLATE_API_URL = "https://number-plate-ai.onrender.com";
 *
 * Leave it empty for local development. The dashboard then talks to
 * http://127.0.0.1:8000, which is where `uvicorn backend.main:app` listens.
 *
 * You can also override it per visit with a query string, which is handy for
 * testing a second backend without editing this file:
 *
 *   https://<user>.github.io/<repo>/?api=https://other-service.onrender.com
 */
window.NUMBER_PLATE_API_URL = "";
