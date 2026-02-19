import { createContext, useContext, useState, useCallback, useEffect } from 'react'
import { uploadCsv, fetchHealth, fetchMetrics } from '../api/client'

const AnalysisContext = createContext(null)

export function AnalysisProvider({ children }) {
  const [analysis, setAnalysis] = useState(null)
  const [metrics, setMetrics] = useState(null)
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
        return result
      } catch (err) {
        setError(err)
        throw err
      } finally {
        setIsUploading(false)
      }
    },
    [refreshMetrics]
  )

  const value = {
    analysis,
    metrics,
    health,
    isUploading,
    error,
    uploadAndAnalyze,
    refreshHealth,
    refreshMetrics,
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

