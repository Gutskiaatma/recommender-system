// Small fetch wrapper. In development Vite proxies /api to FastAPI (see vite.config.js).
// For a separately hosted backend set VITE_API_URL, e.g. VITE_API_URL=http://localhost:8000
const BASE = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError("Cannot reach the API. Is the backend running on port 8000?", 0);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join("; ")
      : detail || `Request failed (${response.status})`;
    throw new ApiError(message, response.status);
  }
  return body;
}

export const api = {
  health: () => request("/api/health"),
  examples: () => request("/api/examples"),
  aspects: () => request("/api/aspects"),
  stats: () => request("/api/stats"),
  recommend: (query, method = "hybrid", topK = 10) =>
    request("/api/recommend", {
      method: "POST",
      body: JSON.stringify({ query, method, top_k: topK }),
    }),
  compare: (productIds, query) =>
    request("/api/compare", {
      method: "POST",
      body: JSON.stringify({ product_ids: productIds, query: query || null }),
    }),
  product: (id) => request(`/api/products/${encodeURIComponent(id)}`),
  searchProducts: (search, limit = 10) =>
    request(`/api/products?search=${encodeURIComponent(search)}&limit=${limit}`),
};
