import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const location = useLocation()
  const isActive = (path) => location.pathname === path

  return (
    <nav className="fixed top-4 left-1/2 -translate-x-1/2 w-[95%] max-w-7xl z-50 rounded-2xl border border-white/5 bg-[#121212cc] backdrop-blur-md shadow-2xl shadow-black/50">
      <div className="px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center shadow-lg shadow-accent-blue/20 bg-accent-blue">
            <span className="material-symbols-outlined text-white text-[18px]">hub</span>
          </div>
          <span className="font-bold text-base tracking-tight text-white font-display">GraphLens</span>
        </Link>

        <div className="flex items-center gap-1 p-1 bg-white/[0.03] rounded-xl border border-white/[0.02]">
          {[
            { to: '/', label: 'Home' },
            { to: '/network-graph', label: 'Network Graph' },
            { to: '/fraud-rings', label: 'Fraud Rings' },
            { to: '/reports', label: 'Reports' },
            { to: '/analytics', label: 'Analytics' },
          ].map(({ to, label }) => (
            <Link key={to} to={to}
              className={`px-4 py-2 rounded-lg text-xs font-bold transition-all duration-200 ${isActive(to)
                ? 'bg-accent-blue text-white shadow-lg shadow-accent-blue/25'
                : 'text-text-muted hover:text-white hover:bg-white/5'
                }`}>
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
