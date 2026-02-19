import { useState, useRef, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Lenis from 'lenis'
import Navbar from '../components/Navbar'
import Footer from '../components/Footer'
import RotatingEarth from '../components/RotatingGlobe'
import { useAnalysis } from '../context/AnalysisContext'

/* ─── Animated counter ─── */
function AnimatedCounter({ target, suffix = '', duration = 2000 }) {
  const [count, setCount] = useState(0)
  const ref = useRef(null)
  const hasAnimated = useRef(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasAnimated.current) {
          hasAnimated.current = true
          const start = performance.now()
          const step = (now) => {
            const elapsed = now - start
            const progress = Math.min(elapsed / duration, 1)
            const eased = 1 - Math.pow(1 - progress, 3)
            setCount(Math.floor(eased * target))
            if (progress < 1) requestAnimationFrame(step)
          }
          requestAnimationFrame(step)
        }
      },
      { threshold: 0.3 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [target, duration])

  return <span ref={ref}>{count}{suffix}</span>
}

/* ─── Scroll reveal ─── */
function useScrollReveal() {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) { el.classList.add('revealed'); observer.unobserve(el) }
      },
      { threshold: 0.15 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return ref
}

export default function Home() {
  const { uploadAndAnalyze, isUploading, error, analysis, health } = useAnalysis()
  const navigate = useNavigate()
  const heroRef = useScrollReveal()
  const uploadRef = useScrollReveal()
  const statsRef = useScrollReveal()
  const modulesRef = useScrollReveal()
  const [selectedFileName, setSelectedFileName] = useState('')

  /* Lenis smooth scroll */
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.4,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smooth: true,
    })
    function raf(time) { lenis.raf(time); requestAnimationFrame(raf) }
    requestAnimationFrame(raf)
    return () => lenis.destroy()
  }, [])

  return (
    <div className="bg-background-dark text-white font-display min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 mt-14">
        {/* ═══════ Hero ═══════ */}
        <section ref={heroRef} className="scroll-reveal relative min-h-[88vh] flex items-center justify-center overflow-hidden">
          {/* Hero Image Background - Increased Visibility */}
          <div className="absolute inset-0 z-0">
            <img src="/hero_network.jpg" alt="Financial Network" className="w-full h-full object-cover opacity-60 mix-blend-screen" />
            <div className="absolute inset-0 bg-gradient-to-b from-background-dark via-transparent to-background-dark"></div>
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(0,0,0,0.6)_100%)]"></div>
          </div>

          <div className="text-center relative z-10 max-w-4xl mx-auto px-6">
            <div className="animate-fade-in-up inline-flex items-center gap-2 bg-white/5 border border-white/10 px-4 py-1.5 rounded-full mb-8 backdrop-blur-md">
              <span className="size-1.5 rounded-full bg-accent-red animate-pulse"></span>
              <span className="text-[11px] font-medium text-white tracking-wide font-body">Live Monitoring Active</span>
            </div>

            <h1 className="animate-fade-in-up delay-200 text-7xl md:text-9xl font-display font-medium leading-[1.0] tracking-tight mb-8 text-white drop-shadow-2xl">
              Financial<br />
              Network Intelligence
            </h1>

            <p className="animate-fade-in-up delay-400 text-xl text-neutral-300 max-w-xl mx-auto leading-relaxed mb-12 font-body drop-shadow-lg">
              Detect complex fraud patterns across financial networks with real-time graph analysis and machine learning.
            </p>

            <div className="animate-fade-in-up delay-500 flex items-center justify-center gap-5">
              <Link to="/network-graph" className="group relative px-8 py-3.5 bg-white text-black rounded-full font-semibold transition-transform hover:scale-105">
                <span className="relative z-10">Get Started</span>
              </Link>
              <Link to="/network-graph" className="px-8 py-3.5 rounded-full border border-white/20 hover:bg-white/10 transition-colors text-white font-medium backdrop-blur-sm">
                View Live Graph
              </Link>
            </div>
          </div>
        </section>

        {/* ═══════ Upload ═══════ */}
        <section ref={uploadRef} className="scroll-reveal relative py-32 px-8">
          {/* Edge gradients - Blue/Purple/Red */}
          <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-accent-blue via-accent-purple to-accent-red opacity-30"></div>

          <div className="absolute top-0 left-0 w-32 h-[500px] bg-accent-blue/5 blur-[100px]"></div>
          <div className="absolute bottom-0 right-0 w-32 h-[500px] bg-accent-purple/5 blur-[100px]"></div>

          <div className="max-w-3xl mx-auto text-center relative z-10">
            <h2 className="text-5xl font-display text-white mb-4 tracking-tight">Upload & Analyze</h2>
            <p className="text-base text-neutral-400 mb-4 font-body">Drop your transaction data to begin fraud detection analysis.</p>
            {health && (
              <p className="text-xs text-neutral-500 mb-6 font-body">
                Backend status: <span className={health.status === 'healthy' ? 'text-emerald-400' : 'text-amber-300'}>{health.status}</span>
              </p>
            )}

            <div className="card-glass rounded-3xl p-12 bg-card-dark border border-white/10 backdrop-blur-md shadow-2xl relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-accent-blue/5 via-transparent to-accent-purple/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700"></div>

              <div className="flex flex-col items-center gap-6 relative z-10">
                <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-500">
                  <span className="material-symbols-outlined text-white text-3xl">upload_file</span>
                </div>
                <div>
                  <p className="text-lg font-medium text-white mb-2 font-display">Upload Transaction Data</p>
                  <p className="text-sm text-neutral-500 font-body">
                    {selectedFileName || 'Drop your CSV file here or click to browse'}
                  </p>
                </div>
                <label className="inline-flex items-center gap-3 px-8 py-3 rounded-full text-sm font-semibold bg-white text-black hover:bg-neutral-200 transition-colors cursor-pointer">
                  <span>{isUploading ? 'Analyzing…' : 'Select CSV File'}</span>
                  <input
                    type="file"
                    accept=".csv"
                    className="hidden"
                    onChange={async (e) => {
                      const file = e.target.files?.[0]
                      if (!file) return
                      setSelectedFileName(file.name)
                      try {
                        const result = await uploadAndAnalyze(file)
                        if (result?.graph_data) {
                          navigate('/network-graph')
                        } else {
                          navigate('/reports')
                        }
                      } catch {
                        // error is surfaced below
                      } finally {
                        e.target.value = ''
                      }
                    }}
                  />
                </label>
                {error && (
                  <p className="text-xs text-red-400 font-body">
                    {error.message || 'Failed to analyze file. Please try again.'}
                  </p>
                )}
                {analysis && !error && !isUploading && (
                  <p className="text-xs text-neutral-400 font-body">
                    Last run: {analysis.summary?.suspicious_accounts_flagged ?? 0} suspicious accounts,{' '}
                    {analysis.summary?.fraud_rings_detected ?? 0} rings detected.
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ═══════ Stats ═══════ */}
        <section ref={statsRef} className="scroll-reveal relative py-20 px-8">
          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 relative z-10">
            {[
              { target: 2.4, suffix: 'B+', label: 'Nodes Analyzed', text: null },
              { target: 99.9, suffix: '%', label: 'Uptime', text: null },
              { target: 0, suffix: '', label: 'Real-time Detection', text: 'Real-time' },
            ].map((stat, i) => (
              <div key={i} className="text-center p-8 rounded-2xl border border-white/5 bg-card-dark hover:border-white/10 transition-all duration-300 relative group overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <p className="text-4xl font-normal text-white mb-2 font-display relative z-10">
                  {stat.text || <AnimatedCounter target={stat.target * 10} suffix={stat.suffix} />}
                </p>
                <p className="text-sm text-neutral-500 font-medium font-body relative z-10">{stat.label}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ═══════ Modules ═══════ */}
        <section ref={modulesRef} className="scroll-reveal relative py-32 px-8 pb-40">
          {/* Bottom Gradient Edge */}
          <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-accent-red via-accent-purple to-accent-blue opacity-30"></div>

          <div className="max-w-6xl mx-auto relative z-10">
            <h2 className="text-4xl font-display text-white mb-3 tracking-tight text-center">Core Intelligence Modules</h2>
            <p className="text-base text-neutral-400 text-center mb-16 font-body">Comprehensive tools for financial network analysis</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                { icon: 'grid_view', title: 'Global Dashboard', desc: 'Unified view of all monitored entities with real-time risk scoring.', to: '/network-graph', color: 'text-accent-blue' },
                { icon: 'hub', title: 'Fraud Ring Detection', desc: 'Automatically identify circular transaction patterns and clusters.', to: '/fraud-rings', color: 'text-accent-purple' },
                { icon: 'bar_chart', title: 'Advanced Analytics', desc: 'Predictive modeling using historical data to flag potential risks.', to: '/analytics', color: 'text-accent-red' },
              ].map((mod, i) => (
                <Link key={i} to={mod.to} className="group relative p-8 rounded-3xl bg-card-dark border border-white/5 hover:border-white/10 transition-all duration-500 overflow-hidden">
                  {/* Hover Gradient */}
                  <div className={`absolute -right-20 -top-20 w-64 h-64 bg-gradient-to-br ${i === 0 ? 'from-accent-blue/10' : i === 1 ? 'from-accent-purple/10' : 'from-accent-red/10'} to-transparent blur-[60px] opacity-0 group-hover:opacity-100 transition-opacity duration-500`}></div>

                  <div className="relative z-10">
                    <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-6 text-white group-hover:scale-110 transition-transform duration-500">
                      <span className={`material-symbols-outlined text-2xl ${mod.color}`}>{mod.icon}</span>
                    </div>
                    <h3 className="text-xl font-normal text-white mb-3 font-display tracking-wide">{mod.title}</h3>
                    <p className="text-sm text-neutral-400 leading-relaxed mb-6 font-body">{mod.desc}</p>
                    <span className="text-xs text-white opacity-60 group-hover:opacity-100 transition-opacity font-medium flex items-center gap-1">
                      Explore Module <span className="material-symbols-outlined text-sm">arrow_forward</span>
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  )
}
