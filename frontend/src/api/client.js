const rawBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_BASE_URL = String(rawBaseUrl).replace(/\/+$/, '')

function buildUrl(path) {
  const normalizedPath = String(path).startsWith('/') ? path : `/${path}`
  return `${API_BASE_URL}${normalizedPath}`
}

async function request(path, options = {}) {
  const controller = new AbortController()
  const timeoutMs = Number(import.meta.env.VITE_API_TIMEOUT_MS || 45000)
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const res = await fetch(buildUrl(path), {
      cache: 'no-store',
      ...options,
      signal: controller.signal,
    })
    return handleResponse(res)
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw new Error('Request timed out while contacting backend.')
    }
    throw err
  } finally {
    clearTimeout(timeoutId)
  }
}

async function handleResponse(res) {
  if (!res.ok) {
    let message = `Request failed with status ${res.status}`
    try {
      const data = await res.json()
      if (data?.detail) message = Array.isArray(data.detail) ? data.detail.map((d) => d.msg || d).join(', ') : data.detail
    } catch {
      // ignore parse errors, fall back to generic message
    }
    const error = new Error(message)
    error.status = res.status
    throw error
  }
  return res.json()
}

export async function uploadCsv(file) {
  const formData = new FormData()
  formData.append('file', file)

  return request('/upload', {
    method: 'POST',
    body: formData,
  })
}

export async function fetchHealth() {
  return request('/health')
}

export async function fetchMetrics() {
  return request('/metrics')
}

export async function fetchHistory() {
  return request('/history')
}

export async function fetchHistoryReport(runId) {
  return request(`/history/${encodeURIComponent(runId)}`)
}
