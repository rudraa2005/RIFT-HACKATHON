import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { uploadCsv, fetchHealth, fetchMetrics, fetchHistory, fetchHistoryReport } from '../api/client'

const AnalysisContext = createContext(null)

export function AnalysisProvider({ children }) {
  const [analysis, setAnalysis] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [history, setHistory] = useState([])
  const [health, setHealth] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState(null)

  const refreshHealth = useCallback(async () => {
    try {
      const data = await fetchHealth()
      setHealth(data)
    } catch {
      // best-effort only
    }
  }, [])

  const refreshMetrics = useCallback(async () => {
    try {
      const data = await fetchMetrics()
      setMetrics(data)
    } catch {
      // best-effort only
    }
  }, [])

  const refreshHistory = useCallback(async () => {
    try {
      const data = await fetchHistory()
      setHistory(Array.isArray(data?.items) ? data.items : [])
    } catch {
      // best-effort only
    }
  }, [])

  const getHistoryReport = useCallback(async (runId) => {
    const data = await fetchHistoryReport(runId)
    return data?.report || null
  }, [])

  useEffect(() => {
    // warm up health on first load so UI can reflect backend status
    refreshHealth()
  }, [refreshHealth])

  const uploadAndAnalyze = useCallback(
    async (file) => {
      setIsUploading(true)
      setError(null)
      try {
        const result = await uploadCsv(file)
        setAnalysis(result)
        // backend also records metrics based on result["summary"]
        await refreshMetrics()
        await refreshHistory()
        return result
      } catch (err) {
        setError(err)
        throw err
      } finally {
        setIsUploading(false)
      }
    },
    [refreshMetrics, refreshHistory]
  )

  const value = {
    analysis,
    metrics,
    history,
    health,
    isUploading,
    error,
    uploadAndAnalyze,
    refreshHealth,
    refreshMetrics,
    refreshHistory,
    getHistoryReport,
  }

  return <AnalysisContext.Provider value={value}>{children}</AnalysisContext.Provider>
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext)
  if (!ctx) {
    throw new Error('useAnalysis must be used within an AnalysisProvider')
  }
  return ctx
}

