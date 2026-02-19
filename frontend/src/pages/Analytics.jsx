import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import Navbar from '../components/Navbar'
import { useAnalysis } from '../context/AnalysisContext'

export default function Analytics() {
  const [timeRange, setTimeRange] = useState('72h')
  const { analysis, metrics, refreshMetrics } = useAnalysis()

  useEffect(() => {
    if (!metrics) {
      refreshMetrics()
    }
  }, [metrics, refreshMetrics])

  /* ─── Dynamic Data Extraction ─── */
  const connMetrics = analysis?.summary?.network_connectivity || {}

  const sccSeries = useMemo(() => {
    return connMetrics.scc_distribution || [45, 15, 35, 75, 95, 55, 25, 10, 60, 40, 80, 20, 45, 60, 30, 70, 25, 50, 90, 35]
  }, [connMetrics])

  const burstSeries = useMemo(() => {
    return connMetrics.burst_activity || [30, 50, 40, 70, 50, 80, 45, 60, 40, 50, 30, 70, 60, 40, 50, 30, 20, 40, 50, 30]
  }, [connMetrics])

  const depthSeries = useMemo(() => {
    return connMetrics.depth_distribution || [30, 50, 40, 70, 50, 80, 45, 60]
  }, [connMetrics])

  // Dynamically build SVG path for trendline
  const trendlinePath = useMemo(() => {
    if (!burstSeries.length) return ''
    const points = burstSeries.map((val, idx) => {
      const x = (idx / (burstSeries.length - 1)) * 1000
      const y = 200 - (val / 100) * 150 - 20
      return `${x},${y}`
    })
    return `M${points.join(' L')}`
  }, [burstSeries])

  const trendlineArea = useMemo(() => {
    if (!trendlinePath) return ''
    return `${trendlinePath} V200 H0 Z`
  }, [trendlinePath])

  return (
    <div className="bg-background-dark text-text-muted font-display min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 bg-background-dark py-6 px-6 lg:px-8 mt-24 relative">
        {/* Edge Gradients */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-accent-blue via-accent-purple to-accent-red opacity-50"></div>

        <div className="max-w-[1600px] mx-auto">
          <header className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-6">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-accent-blue text-xs font-bold uppercase tracking-widest font-body">
                <span className="material-symbols-outlined text-[18px]">analytics</span>
                <span>System Analytics</span>
              </div>
              <h2 className="text-5xl font-display font-medium tracking-tight text-white">Network Intelligence</h2>
            </div>
          </header>

          {/* Bento Grid Layout - Ultra Dense */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 auto-rows-[minmax(140px,auto)]">

            {/* SCC Distribution - Large Card (Top Left) */}
            <div className="md:col-span-2 xl:row-span-2 p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-blue/30 transition-colors group flex flex-col relative overflow-hidden">
              <div className="absolute top-0 right-0 p-6 opacity-10 pointer-events-none">
                <span className="material-symbols-outlined text-9xl text-accent-blue">hub</span>
              </div>
              <div className="flex justify-between items-start mb-6 relative z-10">
                <div>
                  <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest mb-1 font-body">SCC Strength Distribution</h4>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-medium text-white tracking-tight font-display">
                      {connMetrics.connected_components_count || '0.84'}
                    </span>
                    <span className="text-xl text-text-muted font-medium font-display">Clusters</span>
                  </div>
                </div>
                <span className="text-accent-blue bg-accent-blue/10 px-2.5 py-1 rounded text-[11px] font-bold flex items-center gap-1 border border-accent-blue/20 font-technical">
                  <span className="material-symbols-outlined text-[14px]">sensors</span> Real-time
                </span>
              </div>

              <div className="flex-1 flex items-end gap-1.5 px-1 pb-1 relative z-10">
                {sccSeries.map((height, idx) => (
                  <div key={idx} className="w-full bg-white/5 hover:bg-white/10 transition-colors rounded-t-sm relative overflow-hidden group-hover:shadow-[0_0_15px_rgba(255,255,255,0.2)]" style={{ height: `${height}%`, opacity: 0.5 + (height / 200) }}>
                    <div className="absolute bottom-0 left-0 w-full bg-white/90 transition-all duration-500" style={{ height: height > 50 ? '100%' : '0%' }}></div>
                  </div>
                ))}
              </div>
            </div>

            {/* Network Throughput - Square Card */}
            <div className="p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-purple/30 transition-colors flex flex-col justify-between">
              <div className="flex justify-between items-start mb-1">
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest font-body">Analysis Volume</h4>
                <span className="material-symbols-outlined text-white/50 text-[20px]">speed</span>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-[11px] mb-1.5 font-technical">
                    <span className="text-text-muted">Total Accounts</span>
                    <span className="text-white font-bold">
                      {analysis?.summary?.total_accounts_analyzed?.toLocaleString() || '0'}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full bg-white/90 rounded-full w-[75%] shadow-[0_0_10px_rgba(255,255,255,0.3)]"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-[11px] mb-1.5 font-technical">
                    <span className="text-text-muted">Process Time</span>
                    <span className="text-white font-bold">
                      {analysis?.summary?.processing_time_seconds
                        ? `${analysis.summary.processing_time_seconds.toFixed(2)} s`
                        : '0.00 s'}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full bg-white/20 rounded-full w-[65%]"></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Cascade Depth - Square Card */}
            <div className="p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-red/30 transition-colors flex flex-col">
              <div className="flex justify-between items-start">
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest mb-1 font-body">Cascade Complexity</h4>
                <span className="material-symbols-outlined text-white/50 text-[20px]">layers</span>
              </div>
              <div className="mt-auto">
                <span className="text-4xl font-medium text-white tracking-tight font-display">
                  {connMetrics.avg_cascade_depth || '0.0'}
                  <span className="text-sm text-text-muted font-medium ml-1 font-body">Avg Depth</span>
                </span>
                <div className="h-16 flex items-end justify-between gap-1.5 mt-3">
                  {depthSeries.map((height, idx) => (
                    <div key={idx} className="w-full bg-gradient-to-t from-white/30 to-white/80 rounded-t-sm" style={{ height: `${height}%` }}></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Component Summary - Wide Card (Middle Row Right) */}
            <div className="md:col-span-2 xl:col-span-2 p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-purple/30 transition-colors flex flex-col md:flex-row gap-8 items-center">
              <div className="flex-1 w-full text-center md:text-left">
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest mb-4 font-body">Graph Topology</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <span className="text-white font-technical text-2xl font-bold">{connMetrics.largest_component_size || '0'}</span>
                    <p className="text-[9px] text-text-muted uppercase font-bold tracking-widest mt-1">Largest SCC</p>
                  </div>
                  <div>
                    <span className="text-white font-technical text-2xl font-bold">{connMetrics.is_single_network ? 'Yes' : 'No'}</span>
                    <p className="text-[9px] text-text-muted uppercase font-bold tracking-widest mt-1">Single Network</p>
                  </div>
                </div>
              </div>
              <div className="md:border-l md:border-white/5 md:pl-8 text-center min-w-[120px]">
                <span className="text-4xl font-bold text-white font-technical">
                  {analysis?.summary?.suspicious_accounts_flagged || '0'}
                </span>
                <p className="text-[11px] text-text-muted font-bold uppercase mt-1 font-body">Flagged Entities</p>
              </div>
            </div>

            {/* 72H Burst Activity - Full Width Card (Bottom Row) */}
            <div className="md:col-span-2 xl:col-span-4 p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-red/30 transition-colors flex flex-col justify-between h-[280px]">
              <div className="flex justify-between items-center mb-4">
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest font-body">Transactional Velocity (Last 20 Samples)</h4>
                <div className="flex gap-4">
                  <div className="flex items-center gap-2">
                    <div className="size-2 rounded-full bg-accent-red animate-pulse"></div>
                    <span className="text-[11px] text-text-muted uppercase font-bold font-body">Live Context</span>
                  </div>
                </div>
              </div>
              <div className="relative flex-1 w-full mt-2">
                <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 1000 200">
                  <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ffffff" stopOpacity="0.25" />
                      <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <line stroke="rgba(255,255,255,0.05)" strokeWidth="1" x1="0" x2="1000" y1="50" y2="50"></line>
                  <line stroke="rgba(255,255,255,0.05)" strokeWidth="1" x1="0" x2="1000" y1="100" y2="100"></line>
                  <line stroke="rgba(255,255,255,0.05)" strokeWidth="1" x1="0" x2="1000" y1="150" y2="150"></line>

                  <path d={trendlineArea} fill="url(#chartGradient)" stroke="none"></path>
                  <path d={trendlinePath} fill="none" stroke="#ffffff" strokeLinejoin="round" strokeWidth="2" vectorEffect="non-scaling-stroke"></path>
                </svg>
              </div>
            </div>

          </div>
        </div>
      </main>
    </div>
  )
}
