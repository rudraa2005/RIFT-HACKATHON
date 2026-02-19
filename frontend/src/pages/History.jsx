import { useState, useEffect } from 'react'
import Navbar from '../components/Navbar'
import { useAnalysis } from '../context/AnalysisContext'

export default function History() {
    const [selectedReport, setSelectedReport] = useState(null)
    const { metrics, refreshMetrics } = useAnalysis()

    useEffect(() => {
        if (!metrics) {
            refreshMetrics()
        }
    }, [metrics, refreshMetrics])

    const historyData = [
        { id: 1, filename: 'transaction_batch_2023_10_24.csv', date: 'Oct 24, 2023', records: 12450, riskScore: 84, status: 'Analyzed', size: '4.2 MB' },
        { id: 2, filename: 'swift_transfers_v2.csv', date: 'Oct 22, 2023', records: 8320, riskScore: 45, status: 'Analyzed', size: '2.8 MB' },
        { id: 3, filename: 'crypto_bridge_logs.csv', date: 'Oct 20, 2023', records: 15600, riskScore: 92, status: 'Analyzed', size: '5.1 MB' },
        { id: 4, filename: 'merchant_settlements_q3.csv', date: 'Oct 18, 2023', records: 45000, riskScore: 12, status: 'Analyzed', size: '12.4 MB' },
        { id: 5, filename: 'kyc_fails_oct.csv', date: 'Oct 15, 2023', records: 320, riskScore: 78, status: 'Analyzed', size: '0.4 MB' },
    ]

    const downloadJSON = (report) => {
        // Generate content if it's the raw item
        const content = report.metadata ? report : {
            reportId: `RPT-${report.id}-${new Date(report.date).getTime()}`,
            metadata: {
                filename: report.filename,
                processedAt: new Date(report.date).toISOString(),
                duration: "14.2s",
                version: "v2.1.0"
            },
            summary: {
                totalRecords: report.records,
                riskScore: report.riskScore,
                flaggedCount: Math.floor(report.records * (report.riskScore / 500)),
                riskLevel: report.riskScore > 75 ? "CRITICAL" : report.riskScore > 40 ? "HIGH" : "LOW"
            },
            insights: [
                report.riskScore > 80 ? "High frequency burst detected in node cluster A-4" : "Normal traffic pattern observed",
                report.riskScore > 50 ? "Suspicious cyclic transaction detected (length: 4)" : "No cyclic anomalies found",
                "Geographic conformity verify: PASSED"
            ]
        }

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(content, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", content.metadata.filename.replace('.csv', '.json'));
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
    }

    const handleViewConfirm = (report) => {
        // Determine mock content based on risk score to make it look realistic
        const mockContent = {
            reportId: `RPT-${report.id}-${new Date(report.date).getTime()}`,
            metadata: {
                filename: report.filename,
                processedAt: new Date(report.date).toISOString(),
                duration: "14.2s",
                version: "v2.1.0"
            },
            summary: {
                totalRecords: report.records,
                riskScore: report.riskScore,
                flaggedCount: Math.floor(report.records * (report.riskScore / 500)),
                riskLevel: report.riskScore > 75 ? "CRITICAL" : report.riskScore > 40 ? "HIGH" : "LOW"
            },
            insights: [
                report.riskScore > 80 ? "High frequency burst detected in node cluster A-4" : "Normal traffic pattern observed",
                report.riskScore > 50 ? "Suspicious cyclic transaction detected (length: 4)" : "No cyclic anomalies found",
                "Geographic conformity verify: PASSED"
            ]
        }

        setSelectedReport(mockContent)
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
                                Archive of previously analyzed datasets. Download JSON reports or review past analysis metrics.
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
                                        <th className="py-5 px-6 font-semibold text-right">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="text-sm">
                                    {historyData.map((item, idx) => (
                                        <tr key={item.id} className="border-b border-[rgb(36,36,36)] hover:bg-white/[0.02] transition-colors group">
                                            <td className="py-5 px-6">
                                                <div className="flex items-center gap-3">
                                                    <span className="material-symbols-outlined text-slate-500">description</span>
                                                    <span className="font-mono text-slate-200">{item.filename}</span>
                                                </div>
                                            </td>
                                            <td className="py-5 px-6 text-slate-400">{item.date}</td>
                                            <td className="py-5 px-6 font-mono text-slate-400">{item.records.toLocaleString()}</td>
                                            <td className="py-5 px-6">
                                                <div className="flex items-center gap-2">
                                                    <span className={`font-bold ${item.riskScore > 75 ? 'text-red-400' : (item.riskScore > 40 ? 'text-yellow-400' : 'text-green-400')}`}>
                                                        {item.riskScore}
                                                    </span>
                                                    <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                                        <div className={`h-full rounded-full ${item.riskScore > 75 ? 'bg-red-400' : (item.riskScore > 40 ? 'bg-yellow-400' : 'bg-green-400')}`} style={{ width: `${item.riskScore}%` }}></div>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="py-5 px-6">
                                                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-bold uppercase tracking-wider">
                                                    <span className="w-1.5 h-1.5 rounded-full bg-primary"></span>
                                                    {metrics?.status === 'ready' && metrics?.total_runs
                                                        ? `Analyzed (run #${metrics.total_runs})`
                                                        : item.status}
                                                </span>
                                            </td>
                                            <td className="py-5 px-6 text-right">
                                                <div className="flex items-center justify-end gap-3 text-slate-400">
                                                    <button onClick={() => handleViewConfirm(item)} className="flex items-center gap-2 hover:text-white transition-colors text-xs font-medium uppercase tracking-wide">
                                                        <span className="material-symbols-outlined text-[18px]">visibility</span>
                                                        View
                                                    </button>
                                                    <div className="w-px h-4 bg-white/10"></div>
                                                    <button onClick={() => downloadJSON(item)} className="flex items-center gap-2 hover:text-primary transition-colors text-xs font-medium uppercase tracking-wide">
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

                {/* JSON Viewer Modal */}
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
                                        <p className="text-xs text-slate-500 font-mono mt-0.5">{selectedReport.metadata.filename}</p>
                                    </div>
                                </div>
                                <button onClick={() => setSelectedReport(null)} className="text-slate-500 hover:text-white transition-colors">
                                    <span className="material-symbols-outlined text-2xl">close</span>
                                </button>
                            </div>
                            <div className="p-6 overflow-auto custom-scrollbar flex-1">
                                <pre className="text-xs md:text-sm font-mono text-slate-300 bg-[#0a0a0a] p-4 rounded-xl border border-white/5 overflow-x-auto">
                                    {JSON.stringify(selectedReport, null, 2)}
                                </pre>
                            </div>
                            <div className="p-4 border-t border-white/5 flex justify-end gap-3 bg-[rgb(20,20,20)] rounded-b-2xl">
                                <button onClick={() => setSelectedReport(null)} className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white transition-colors">Close</button>
                                <button onClick={() => downloadJSON(selectedReport)} className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary-hover text-white text-sm font-bold rounded-lg transition-colors">
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
