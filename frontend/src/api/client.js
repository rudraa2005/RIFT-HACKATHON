const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

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

  const res = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

  return handleResponse(res)
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/health`)
  return handleResponse(res)
}

export async function fetchMetrics() {
  const res = await fetch(`${API_BASE_URL}/metrics`)
  return handleResponse(res)
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE_URL}/history`)
  return handleResponse(res)
}

export async function fetchHistoryReport(runId) {
  const res = await fetch(`${API_BASE_URL}/history/${encodeURIComponent(runId)}`)
  return handleResponse(res)
}

