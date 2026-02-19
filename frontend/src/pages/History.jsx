import { useState, useEffect } from 'react'
import Navbar from '../components/Navbar'
import { useAnalysis } from '../context/AnalysisContext'

function formatBytes(bytes) {
  const n = Number(bytes)
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = n
  let idx = 0
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024
    idx += 1
  }
  return `${size.toFixed(size >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`
}

function formatDate(iso) {
  const d = new Date(iso)
  if (!Number.isFinite(d.getTime())) return 'N/A'
  return d.toLocaleString([], {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export default function History() {
  const [selectedReport, setSelectedReport] = useState(null)
  const [loadingId, setLoadingId] = useState(null)
  const { history, refreshHistory, getHistoryReport } = useAnalysis()

  useEffect(() => {
    refreshHistory()
  }, [refreshHistory])

  const downloadJSON = (reportPayload, filename = 'analysis_report.json') => {
    const dataStr = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(reportPayload, null, 2))}`
    const link = document.createElement('a')
    link.setAttribute('href', dataStr)
    link.setAttribute('download', filename.replace('.csv', '.json'))
    document.body.appendChild(link)
    link.click()
    link.remove()
  }

  const handleView = async (item) => {
    try {
      setLoadingId(item.id)
      const report = await getHistoryReport(item.id)
      if (!report) return
      setSelectedReport({
        filename: item.filename,
        report,
      })
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <div className="bg-background-dark text-slate-100 font-display min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 bg-background-dark py-12 px-8 lg:px-16 mt-24">
        <div className="max-w-7xl mx-auto">
          <header className="flex flex-col md:flex-row md:items-end justify-between gap-8 mb-16 pb-8 border-b border-white/5">
            <div className="space-y-4">
              <div className="flex items-center gap-3 text-primary text-sm font-semibold uppercase tracking-wider">
                <span className="material-symbols-outlined text-[20px]">history</span>
                <span>Upload Archive</span>
              </div>
              <h2 className="text-5xl font-bold tracking-tight text-white">History</h2>
              <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
                Archive of previously analyzed datasets from backend storage.
              </p>
            </div>
          </header>

          <div className="w-full bg-[rgb(20,20,20)] border border-[rgb(36,36,36)] rounded-xl overflow-hidden shadow-2xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="text-xs uppercase text-slate-500 border-b border-[rgb(36,36,36)] bg-[rgb(16,16,16)]">
                    <th className="py-5 px-6 font-semibold">Filename</th>
                    <th className="py-5 px-6 font-semibold">Date Uploaded</th>
                    <th className="py-5 px-6 font-semibold">Records</th>
                    <th className="py-5 px-6 font-semibold">Avg Risk Score</th>
                    <th className="py-5 px-6 font-semibold">Status</th>
                    <th className="py-5 px-6 font-semibold">Size</th>
                    <th className="py-5 px-6 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  {history.length === 0 && (
                    <tr>
                      <td colSpan={7} className="py-10 px-6 text-center text-slate-400">
                        No history yet. Upload and analyze a CSV to create records.
                      </td>
                    </tr>
                  )}

                  {history.map((item) => (
                    <tr key={item.id} className="border-b border-[rgb(36,36,36)] hover:bg-white/[0.02] transition-colors group">
                      <td className="py-5 px-6">
                        <div className="flex items-center gap-3">
                          <span className="material-symbols-outlined text-slate-500">description</span>
                          <span className="font-mono text-slate-200">{item.filename}</span>
                        </div>
                      </td>
                      <td className="py-5 px-6 text-slate-400">{formatDate(item.uploaded_at)}</td>
                      <td className="py-5 px-6 font-mono text-slate-400">{Number(item.records || 0).toLocaleString()}</td>
                      <td className="py-5 px-6">
                        <div className="flex items-center gap-2">
                          <span className={`font-bold ${item.risk_score > 75 ? 'text-red-400' : (item.risk_score > 40 ? 'text-yellow-400' : 'text-green-400')}`}>
                            {Math.round(Number(item.risk_score || 0))}
                          </span>
                          <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${item.risk_score > 75 ? 'bg-red-400' : (item.risk_score > 40 ? 'bg-yellow-400' : 'bg-green-400')}`}
                              style={{ width: `${Math.min(100, Math.max(0, Number(item.risk_score || 0)))}%` }}
                            ></div>
                          </div>
                        </div>
                      </td>
                      <td className="py-5 px-6">
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-bold uppercase tracking-wider">
                          <span className="w-1.5 h-1.5 rounded-full bg-primary"></span>
                          {item.status || 'Analyzed'}
                        </span>
                      </td>
                      <td className="py-5 px-6 text-slate-400 font-mono">{formatBytes(item.file_size_bytes)}</td>
                      <td className="py-5 px-6 text-right">
                        <div className="flex items-center justify-end gap-3 text-slate-400">
                          <button
                            onClick={() => handleView(item)}
                            disabled={loadingId === item.id}
                            className="flex items-center gap-2 hover:text-white transition-colors text-xs font-medium uppercase tracking-wide disabled:opacity-50"
                          >
                            <span className="material-symbols-outlined text-[18px]">visibility</span>
                            {loadingId === item.id ? 'Loading' : 'View'}
                          </button>
                          <div className="w-px h-4 bg-white/10"></div>
                          <button
                            onClick={async () => {
                              const report = await getHistoryReport(item.id)
                              if (report) downloadJSON(report, item.filename)
                            }}
                            className="flex items-center gap-2 hover:text-primary transition-colors text-xs font-medium uppercase tracking-wide"
                          >
                            <span className="material-symbols-outlined text-[18px]">download</span>
                            JSON
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {selectedReport && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-fade-in">
            <div className="bg-[rgb(18,18,18)] border border-white/10 rounded-2xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl">
              <div className="flex items-center justify-between p-6 border-b border-white/5">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-primary/10 rounded-lg text-primary">
                    <span className="material-symbols-outlined text-xl">data_object</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-white leading-tight">Analysis Report</h3>
                    <p className="text-xs text-slate-500 font-mono mt-0.5">{selectedReport.filename}</p>
                  </div>
                </div>
                <button onClick={() => setSelectedReport(null)} className="text-slate-500 hover:text-white transition-colors">
                  <span className="material-symbols-outlined text-2xl">close</span>
                </button>
              </div>
              <div className="p-6 overflow-auto custom-scrollbar flex-1">
                <pre className="text-xs md:text-sm font-mono text-slate-300 bg-[#0a0a0a] p-4 rounded-xl border border-white/5 overflow-x-auto">
                  {JSON.stringify(selectedReport.report, null, 2)}
                </pre>
              </div>
              <div className="p-4 border-t border-white/5 flex justify-end gap-3 bg-[rgb(20,20,20)] rounded-b-2xl">
                <button onClick={() => setSelectedReport(null)} className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white transition-colors">Close</button>
                <button
                  onClick={() => downloadJSON(selectedReport.report, selectedReport.filename)}
                  className="flex items-center gap-2 px-4 py-2 bg-accent-blue hover:bg-blue-500 text-white text-sm font-bold rounded-lg transition-colors"
                >
                  <span className="material-symbols-outlined text-[18px]">download</span>
                  Download JSON
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
