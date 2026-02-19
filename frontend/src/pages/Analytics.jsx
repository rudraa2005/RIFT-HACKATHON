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

  // Simple time-range dependent series to make charts respond to user selection.
  const burstSeries = useMemo(() => {
    switch (timeRange) {
      case '24h':
        return [20, 35, 50, 65, 80, 60, 40, 30, 50, 45, 70, 30, 40, 55, 35, 60, 30, 45, 70, 40]
      case '7d':
        return [15, 25, 45, 55, 75, 65, 35, 25, 55, 35, 65, 25, 35, 50, 40, 65, 35, 55, 80, 50]
      case '30d':
        return [10, 20, 30, 40, 50, 45, 30, 20, 35, 25, 45, 20, 25, 35, 30, 45, 25, 35, 55, 35]
      default:
        return [45, 15, 35, 75, 95, 55, 25, 10, 60, 40, 80, 20, 45, 60, 30, 70, 25, 50, 90, 35]
    }
  }, [timeRange])

  const latencyBuckets = useMemo(() => {
    if (timeRange === '24h') {
      return [
        { label: '< 50ms', value: 72, color: 'bg-white' },
        { label: '50-200ms', value: 18, color: 'bg-white/60' },
        { label: '200-500ms', value: 7, color: 'bg-white/30' },
        { label: '> 500ms', value: 3, color: 'bg-white/10' },
      ]
    }
    if (timeRange === '7d') {
      return [
        { label: '< 50ms', value: 58, color: 'bg-white' },
        { label: '50-200ms', value: 26, color: 'bg-white/60' },
        { label: '200-500ms', value: 11, color: 'bg-white/30' },
        { label: '> 500ms', value: 5, color: 'bg-white/10' },
      ]
    }
    if (timeRange === '30d') {
      return [
        { label: '< 50ms', value: 48, color: 'bg-white' },
        { label: '50-200ms', value: 30, color: 'bg-white/60' },
        { label: '200-500ms', value: 14, color: 'bg-white/30' },
        { label: '> 500ms', value: 8, color: 'bg-white/10' },
      ]
    }
    return [
      { label: '< 50ms', value: 64, color: 'bg-white' },
      { label: '50-200ms', value: 22, color: 'bg-white/60' },
      { label: '200-500ms', value: 10, color: 'bg-white/30' },
      { label: '> 500ms', value: 4, color: 'bg-white/10' },
    ]
  }, [timeRange])

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
              <div className="flex items-center gap-3">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="px-4 py-2 bg-card-dark text-text-muted border border-white/5 rounded-lg hover:bg-white/5 hover:text-white transition-colors text-xs font-medium focus:outline-none focus:ring-1 focus:ring-accent-blue/50 cursor-pointer font-body"
              >
                <option value="24h">Last 24 Hours</option>
                <option value="72h">Last 72 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <Link to="/history" className="flex items-center gap-2 px-4 py-2 bg-card-dark text-text-muted border border-white/5 rounded-lg hover:bg-white/5 hover:text-white transition-colors text-xs font-medium group font-body">
                <span className="material-symbols-outlined text-[16px] text-text-muted group-hover:text-accent-blue transition-colors">history</span>
                History
              </Link>
            </div>
          </header>

          {/* Bento Grid Layout - Ultra Dense */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 auto-rows-[minmax(140px,auto)]">

            {/* Scale-Free Nature - Large Card (Top Left) */}
            <div className="md:col-span-2 xl:row-span-2 p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-blue/30 transition-colors group flex flex-col relative overflow-hidden">
              <div className="absolute top-0 right-0 p-6 opacity-10 pointer-events-none">
                <span className="material-symbols-outlined text-9xl text-accent-blue">hub</span>
              </div>
              <div className="flex justify-between items-start mb-6 relative z-10">
                <div>
                  <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest mb-1 font-body">SCC Strength Distribution</h4>
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-medium text-white tracking-tight font-display">0.84</span>
                    <span className="text-xl text-text-muted font-medium font-display">σ</span>
                  </div>
                </div>
                <span className="text-accent-blue bg-accent-blue/10 px-2.5 py-1 rounded text-[11px] font-bold flex items-center gap-1 border border-accent-blue/20 font-technical">
                  <span className="material-symbols-outlined text-[14px]">trending_up</span> 2.1%
                </span>
              </div>

              <div className="flex-1 flex items-end gap-1.5 px-1 pb-1 relative z-10">
                {burstSeries.map((height, idx) => (
                  <div key={idx} className="w-full bg-white/5 hover:bg-white/10 transition-colors rounded-t-sm relative overflow-hidden group-hover:shadow-[0_0_15px_rgba(255,255,255,0.2)]" style={{ height: `${height}%`, opacity: 0.5 + (height / 200) }}>
                    <div className="absolute bottom-0 left-0 w-full bg-white/90 transition-all duration-500" style={{ height: height > 50 ? '100%' : '0%' }}></div>
                  </div>
                ))}
              </div>
            </div>

            {/* Network Throughput - Square Card */}
            <div className="p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-purple/30 transition-colors flex flex-col justify-between">
              <div className="flex justify-between items-start mb-1">
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest font-body">Network Throughput</h4>
                <span className="material-symbols-outlined text-white/50 text-[20px]">speed</span>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-[11px] mb-1.5 font-technical">
                    <span className="text-text-muted">Ingress</span>
                    <span className="text-white font-bold">
                      {analysis?.summary?.total_accounts_analyzed
                        ? `${analysis.summary.total_accounts_analyzed.toLocaleString()} accts`
                        : '4.2 GB/s'}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full bg-white/90 rounded-full w-[75%] shadow-[0_0_10px_rgba(255,255,255,0.3)]"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-[11px] mb-1.5 font-technical">
                    <span className="text-text-muted">Egress</span>
                    <span className="text-white font-bold">
                      {metrics?.last_run?.processing_time_seconds
                        ? `${metrics.last_run.processing_time_seconds.toFixed(2)} s`
                        : '3.8 GB/s'}
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
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest mb-1 font-body">Cascade Depth</h4>
                <span className="material-symbols-outlined text-white/50 text-[20px]">layers</span>
              </div>
              <div className="mt-auto">
                <span className="text-4xl font-medium text-white tracking-tight font-display">12.4 <span className="text-sm text-text-muted font-medium ml-1 font-body">Avg Lvl</span></span>
                <div className="h-16 flex items-end justify-between gap-1.5 mt-3">
                  {[30, 50, 40, 70, 50, 80, 45, 60].map((height, idx) => (
                    <div key={idx} className="w-full bg-gradient-to-t from-white/30 to-white/80 rounded-t-sm" style={{ height: `${height}%` }}></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Latency Distribution - Wide Card (Middle Row Right) */}
            <div className="md:col-span-2 xl:col-span-2 p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-purple/30 transition-colors flex flex-col md:flex-row gap-8 items-center">
              <div className="flex-1 w-full">
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest mb-4 font-body">Latency Distribution</h4>
                <div className="space-y-3">
                  {latencyBuckets.map((item, idx) => (
                    <div key={idx} className="group">
                      <div className="flex justify-between text-[11px] font-bold text-text-muted uppercase mb-1.5 font-technical">
                        <span>{item.label}</span>
                        <span className="text-white">{item.value}%</span>
                      </div>
                      <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${item.color}`} style={{ width: `${item.value}%` }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="md:border-l md:border-white/5 md:pl-8 text-center min-w-[120px]">
                <span className="text-4xl font-bold text-white font-technical">42ms</span>
                <p className="text-[11px] text-text-muted font-bold uppercase mt-1 font-body">Global Avg</p>
              </div>
            </div>

            {/* 72H Burst Activity - Full Width Card (Bottom Row) */}
            <div className="md:col-span-2 xl:col-span-4 p-6 rounded-3xl bg-card-dark border border-white/5 hover:border-accent-red/30 transition-colors flex flex-col justify-between h-[280px]">
              <div className="flex justify-between items-center mb-4">
                <h4 className="text-text-muted text-[11px] font-bold uppercase tracking-widest font-body">72-Hour Burst Activity</h4>
                <div className="flex gap-4">
                  <div className="flex items-center gap-2">
                    <div className="size-2 rounded-full bg-accent-red animate-pulse"></div>
                    <span className="text-[11px] text-text-muted uppercase font-bold font-body">Live</span>
                  </div>
                  {metrics?.last_run?.processing_time_seconds && (
                    <div className="flex items-center gap-2 text-[11px] text-text-muted font-body">
                      <span className="material-symbols-outlined text-[16px] text-accent-blue">timer</span>
                      <span>
                        Last run:{" "}
                        <span className="text-white">
                          {metrics.last_run.processing_time_seconds.toFixed(2)}s
                        </span>
                      </span>
                    </div>
                  )}
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

                  <path d="M0,150 L50,145 L100,160 L150,130 L200,150 L250,90 L300,50 L350,70 L400,60 L450,80 L500,40 L550,60 L600,100 L650,90 L700,70 L750,110 L800,120 L850,90 L900,40 L950,60 L1000,70 V200 H0 Z" fill="url(#chartGradient)" stroke="none"></path>
                  <path d="M0,150 L50,145 L100,160 L150,130 L200,150 L250,90 L300,50 L350,70 L400,60 L450,80 L500,40 L550,60 L600,100 L650,90 L700,70 L750,110 L800,120 L850,90 L900,40 L950,60 L1000,70" fill="none" stroke="#ffffff" strokeLinejoin="round" strokeWidth="2" vectorEffect="non-scaling-stroke"></path>
                </svg>
              </div>
            </div>

          </div>
        </div>
      </main>
    </div>
  )
}
