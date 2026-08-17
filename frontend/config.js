/**
 * Frontend configuration.
 *
 * When this dashboard is served by the same FastAPI process (the default on
 * Render), leave NUMBER_PLATE_API_URL empty — the browser will use the same
 * origin automatically.
 *
 * Override per visit with a query string for testing a different backend:
 *   https://your-app.onrender.com/?api=http://localhost:8000
 */
window.NUMBER_PLATE_API_URL = "";
