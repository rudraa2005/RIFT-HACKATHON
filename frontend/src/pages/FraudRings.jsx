import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import Footer from '../components/Footer'
import { useAnalysis } from '../context/AnalysisContext'

const fraudRingsData = [
  {
    id: 'FR-8842', type: 'Layering', entities: 18, score: 94, cohesion: 0.85, active: true,
    description: 'Detected multi-hop layering behavior. Multiple entities funneling large transactions through intermittent sleeper accounts before final extraction.',
    nodes: [
      { id: 'n1', label: 'US-88219-X', role: 'Primary Node', r: 16, color: '#e5e5e5', accId: 'ACCT-8821-B', txnId: 'TXN-4821-AB', amount: '$124,500.00', timestamp: 'Oct 24, 2023 14:30' },
      { id: 'n2', label: 'EU-00421-B', role: 'Intermediary', r: 12, color: '#a3a3a3', accId: 'ACCT-0042-E', txnId: 'TXN-9913-CD', amount: '$87,200.00', timestamp: 'Oct 24, 2023 12:15' },
      { id: 'n3', label: 'AS-99120-Q', role: 'Intermediary', r: 12, color: '#a3a3a3', accId: 'ACCT-9912-Q', txnId: 'TXN-3302-EF', amount: '$62,100.00', timestamp: 'Oct 23, 2023 09:45' },
      { id: 'n4', label: 'EU-77891-K', role: 'Mule', r: 8, color: '#525252', accId: 'ACCT-7789-K', txnId: 'TXN-5578-GH', amount: '$31,400.00', timestamp: 'Oct 23, 2023 16:20' },
      { id: 'n5', label: 'US-55432-M', role: 'Endpoint', r: 8, color: '#525252', accId: 'ACCT-5543-M', txnId: 'TXN-2201-IJ', amount: '$18,900.00', timestamp: 'Oct 22, 2023 11:05' },
    ],
    edges: [
      { from: 'n1', to: 'n2', suspicious: true },
      { from: 'n1', to: 'n3', suspicious: true },
      { from: 'n2', to: 'n4', suspicious: false },
      { from: 'n3', to: 'n5', suspicious: false },
      { from: 'n4', to: 'n5', suspicious: true },
    ],
    entityList: [
      { id: 'US-88219-X', role: 'Primary Node', score: 98.2 },
      { id: 'EU-00421-B', role: 'Intermediary', score: 87.5 },
      { id: 'AS-99120-Q', role: 'Intermediary', score: 76.1 },
    ]
  },
  {
    id: 'FR-9021', type: 'Cycle', entities: 24, score: 89, cohesion: 0.72, active: false,
    description: 'Circular transaction flow detected among 24 entities. Funds cycle through multiple jurisdictions before returning to origin accounts.',
    nodes: [
      { id: 'n1', label: 'CY-11294-A', role: 'Hub', r: 15, color: '#e5e5e5', accId: 'ACCT-1129-A', txnId: 'TXN-8834-KL', amount: '$245,000.00', timestamp: 'Oct 20, 2023 08:30' },
      { id: 'n2', label: 'UK-82910-D', role: 'Relay', r: 11, color: '#a3a3a3', accId: 'ACCT-8291-D', txnId: 'TXN-6621-MN', amount: '$98,400.00', timestamp: 'Oct 20, 2023 10:55' },
      { id: 'n3', label: 'SG-44291-R', role: 'Relay', r: 11, color: '#a3a3a3', accId: 'ACCT-4429-R', txnId: 'TXN-1192-OP', amount: '$72,300.00', timestamp: 'Oct 19, 2023 14:10' },
      { id: 'n4', label: 'CH-19920-V', role: 'Sink', r: 9, color: '#737373', accId: 'ACCT-1992-V', txnId: 'TXN-4410-QR', amount: '$55,800.00', timestamp: 'Oct 19, 2023 17:40' },
      { id: 'n5', label: 'HK-76420-L', role: 'Sink', r: 9, color: '#737373', accId: 'ACCT-7642-L', txnId: 'TXN-7788-ST', amount: '$41,200.00', timestamp: 'Oct 18, 2023 09:20' },
    ],
    edges: [
      { from: 'n1', to: 'n2', suspicious: true },
      { from: 'n1', to: 'n3', suspicious: true },
      { from: 'n2', to: 'n4', suspicious: false },
      { from: 'n3', to: 'n5', suspicious: false },
      { from: 'n4', to: 'n1', suspicious: true },
      { from: 'n5', to: 'n1', suspicious: true },
    ],
    entityList: [
      { id: 'CY-11294-A', role: 'Hub Node', score: 92.8 },
      { id: 'UK-82910-D', role: 'Relay', score: 81.3 },
      { id: 'SG-44291-R', role: 'Relay', score: 74.6 },
    ]
  },
  {
    id: 'FR-7731', type: 'Synthetic', entities: 42, score: 76, cohesion: 0.54, active: false,
    description: 'Cluster of synthetic identity accounts opened within 72-hour window. Shared PII fragments detected across 42 entities.',
    nodes: [
      { id: 'n1', label: 'SYN-00142', role: 'Generator', r: 14, color: '#a3a3a3', accId: 'ACCT-0014-S', txnId: 'TXN-3391-UV', amount: '$8,200.00', timestamp: 'Oct 15, 2023 03:20' },
      { id: 'n2', label: 'SYN-00143', role: 'Clone', r: 9, color: '#525252', accId: 'ACCT-0014-T', txnId: 'TXN-3392-WX', amount: '$5,100.00', timestamp: 'Oct 15, 2023 03:22' },
      { id: 'n3', label: 'SYN-00144', role: 'Clone', r: 9, color: '#525252', accId: 'ACCT-0014-U', txnId: 'TXN-3393-YZ', amount: '$4,800.00', timestamp: 'Oct 15, 2023 03:25' },
      { id: 'n4', label: 'SYN-00145', role: 'Clone', r: 8, color: '#525252', accId: 'ACCT-0014-V', txnId: 'TXN-3394-AB', amount: '$6,300.00', timestamp: 'Oct 15, 2023 03:28' },
    ],
    edges: [
      { from: 'n1', to: 'n2', suspicious: false },
      { from: 'n1', to: 'n3', suspicious: false },
      { from: 'n1', to: 'n4', suspicious: false },
      { from: 'n2', to: 'n3', suspicious: true },
    ],
    entityList: [
      { id: 'SYN-00142', role: 'Generator', score: 78.4 },
      { id: 'SYN-00143', role: 'Clone', score: 68.2 },
      { id: 'SYN-00144', role: 'Clone', score: 65.9 },
    ]
  },
  {
    id: 'FR-6540', type: 'Circular Flow', entities: 12, score: 92, cohesion: 0.91, active: false,
    description: 'Tight circular flow among 12 entities with near-identical transaction amounts. Highly coordinated timing suggests automated transfers.',
    nodes: [
      { id: 'n1', label: 'CF-81002-A', role: 'Origin', r: 15, color: '#e5e5e5', accId: 'ACCT-8100-A', txnId: 'TXN-9901-CD', amount: '$320,000.00', timestamp: 'Oct 21, 2023 06:00' },
      { id: 'n2', label: 'CF-81003-B', role: 'Relay', r: 11, color: '#a3a3a3', accId: 'ACCT-8100-B', txnId: 'TXN-9902-EF', amount: '$319,800.00', timestamp: 'Oct 21, 2023 06:02' },
      { id: 'n3', label: 'CF-81004-C', role: 'Relay', r: 11, color: '#a3a3a3', accId: 'ACCT-8100-C', txnId: 'TXN-9903-GH', amount: '$319,600.00', timestamp: 'Oct 21, 2023 06:04' },
      { id: 'n4', label: 'CF-81005-D', role: 'Return', r: 12, color: '#737373', accId: 'ACCT-8100-D', txnId: 'TXN-9904-IJ', amount: '$319,400.00', timestamp: 'Oct 21, 2023 06:06' },
    ],
    edges: [
      { from: 'n1', to: 'n2', suspicious: true },
      { from: 'n2', to: 'n4', suspicious: true },
      { from: 'n1', to: 'n3', suspicious: true },
      { from: 'n3', to: 'n4', suspicious: true },
      { from: 'n4', to: 'n1', suspicious: true },
    ],
    entityList: [
      { id: 'CF-81002-A', role: 'Origin', score: 96.1 },
      { id: 'CF-81005-D', role: 'Return Point', score: 91.3 },
      { id: 'CF-81003-B', role: 'Relay', score: 82.4 },
    ]
  },
  {
    id: 'FR-5192', type: 'Mule Account', entities: 8, score: 65, cohesion: 0.62, active: false,
    description: 'Money mule network with 8 accounts receiving structured deposits below reporting thresholds before rapid outbound transfers.',
    nodes: [
      { id: 'n1', label: 'ML-40210-F', role: 'Controller', r: 13, color: '#a3a3a3', accId: 'ACCT-4021-F', txnId: 'TXN-7701-KL', amount: '$9,800.00', timestamp: 'Oct 18, 2023 13:15' },
      { id: 'n2', label: 'ML-40211-G', role: 'Mule', r: 8, color: '#525252', accId: 'ACCT-4021-G', txnId: 'TXN-7702-MN', amount: '$4,900.00', timestamp: 'Oct 18, 2023 13:20' },
      { id: 'n3', label: 'ML-40212-H', role: 'Mule', r: 8, color: '#525252', accId: 'ACCT-4021-H', txnId: 'TXN-7703-OP', amount: '$4,800.00', timestamp: 'Oct 18, 2023 13:22' },
    ],
    edges: [
      { from: 'n1', to: 'n2', suspicious: false },
      { from: 'n1', to: 'n3', suspicious: false },
      { from: 'n2', to: 'n3', suspicious: true },
    ],
    entityList: [
      { id: 'ML-40210-F', role: 'Controller', score: 72.3 },
      { id: 'ML-40211-G', role: 'Mule', score: 58.1 },
    ]
  },
  {
    id: 'FR-4421', type: 'Bust-out', entities: 31, score: 88, cohesion: 0.78, active: false,
    description: 'Coordinated bust-out scheme across 31 credit accounts. Rapid credit utilization followed by default within 60-day window.',
    nodes: [
      { id: 'n1', label: 'BO-92101-P', role: 'Orchestrator', r: 15, color: '#e5e5e5', accId: 'ACCT-9210-P', txnId: 'TXN-1101-QR', amount: '$185,000.00', timestamp: 'Oct 10, 2023 09:00' },
      { id: 'n2', label: 'BO-92102-Q', role: 'Participant', r: 10, color: '#a3a3a3', accId: 'ACCT-9210-Q', txnId: 'TXN-1102-ST', amount: '$42,000.00', timestamp: 'Oct 10, 2023 09:15' },
      { id: 'n3', label: 'BO-92103-R', role: 'Participant', r: 10, color: '#a3a3a3', accId: 'ACCT-9210-R', txnId: 'TXN-1103-UV', amount: '$38,500.00', timestamp: 'Oct 10, 2023 09:30' },
      { id: 'n4', label: 'BO-92104-S', role: 'Cashout', r: 11, color: '#737373', accId: 'ACCT-9210-S', txnId: 'TXN-1104-WX', amount: '$95,200.00', timestamp: 'Oct 11, 2023 14:00' },
    ],
    edges: [
      { from: 'n1', to: 'n2', suspicious: true },
      { from: 'n1', to: 'n3', suspicious: true },
      { from: 'n2', to: 'n4', suspicious: true },
      { from: 'n3', to: 'n4', suspicious: true },
    ],
    entityList: [
      { id: 'BO-92101-P', role: 'Orchestrator', score: 94.7 },
      { id: 'BO-92104-S', role: 'Cashout', score: 88.2 },
      { id: 'BO-92102-Q', role: 'Participant', score: 71.9 },
    ]
  },
]

export { fraudRingsData }

/* ═══════ Force-simulated animated graph (Canvas) ═══════ */
function ForceGraph({ ring }) {
  const canvasRef = useRef(null)
  const animRef = useRef(null)
  const nodesRef = useRef([])
  const edgesRef = useRef([])
  const hoveredRef = useRef(null)
  const [hoveredNode, setHoveredNode] = useState(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const sizeRef = useRef({ w: 0, h: 0 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const w = canvas.offsetWidth
    const h = canvas.offsetHeight
    sizeRef.current = { w, h }
    canvas.width = w * 2
    canvas.height = h * 2
    const cx = w / 2, cy = h / 2
    const spread = Math.min(w, h) * 0.34

    nodesRef.current = ring.nodes.map((n, i) => {
      const angle = (i / ring.nodes.length) * Math.PI * 2 - Math.PI / 2
      return {
        ...n,
        x: cx + Math.cos(angle) * spread,
        y: cy + Math.sin(angle) * spread,
        vx: 0, vy: 0,
        targetX: cx + Math.cos(angle) * spread,
        targetY: cy + Math.sin(angle) * spread,
      }
    })
    edgesRef.current = ring.edges.map(e => ({ ...e, dashOffset: 0 }))
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
  }, [ring])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const getNode = (id) => nodesRef.current.find(n => n.id === id)

    function tick() {
      const { w, h } = sizeRef.current
      ctx.setTransform(2, 0, 0, 2, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const nodes = nodesRef.current
      const edges = edgesRef.current
      const hovered = hoveredRef.current

      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i]
        node.vx += (node.targetX - node.x) * 0.03
        node.vy += (node.targetY - node.y) * 0.03
        for (let j = i + 1; j < nodes.length; j++) {
          const other = nodes[j]
          const dx = node.x - other.x
          const dy = node.y - other.y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1
          const minDist = (node.r + other.r) * 2.5
          if (dist < minDist) {
            const force = (minDist - dist) * 0.01
            node.vx += (dx / dist) * force; node.vy += (dy / dist) * force
            other.vx -= (dx / dist) * force; other.vy -= (dy / dist) * force
          }
        }
        node.vx *= 0.88; node.vy *= 0.88
        node.x += node.vx; node.y += node.vy
        node.x = Math.max(node.r + 24, Math.min(w - node.r - 24, node.x))
        node.y = Math.max(node.r + 28, Math.min(h - node.r - 28, node.y))
      }

      // Edges
      for (const edge of edges) {
        const from = getNode(edge.from), to = getNode(edge.to)
        if (!from || !to) continue
        const isHL = hovered && (hovered.id === edge.from || hovered.id === edge.to)
        ctx.beginPath()
        ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y)
        ctx.strokeStyle = edge.suspicious
          ? (isHL ? '#ffffff' : 'rgba(255,255,255,0.7)')
          : (isHL ? '#a3a3a3' : 'rgba(140,140,140,0.35)')
        ctx.lineWidth = isHL ? 2.2 : 1.2
        if (edge.suspicious) {
          edge.dashOffset = (edge.dashOffset - 0.4) % 16
          ctx.setLineDash([3, 3]); ctx.lineDashOffset = edge.dashOffset
        } else { ctx.setLineDash([]) }
        ctx.stroke(); ctx.setLineDash([])
      }

      // Nodes
      for (const node of nodes) {
        const isHov = hovered?.id === node.id
        const baseR = node.r * 1.55
        const drawR = isHov ? baseR + 4 : baseR

        // Glow
        if (isHov) {
          ctx.beginPath()
          ctx.arc(node.x, node.y, drawR + 10, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(239, 68, 68, 0.1)'
          ctx.fill()
        }

        // Main circle - Minimal Dark Theme
        ctx.beginPath()
        ctx.arc(node.x, node.y, drawR, 0, Math.PI * 2)

        let nodeColor = '#232323'
        if (node.role?.toLowerCase().includes('primary') || node.role?.toLowerCase().includes('hub') || node.role?.toLowerCase().includes('origin')) {
          nodeColor = '#f5f5f5'
        } else if (node.role?.toLowerCase().includes('intermediary') || node.role?.toLowerCase().includes('relay')) {
          nodeColor = '#9ca3af'
        } else if (node.role?.toLowerCase().includes('mule') || node.role?.toLowerCase().includes('sink')) {
          nodeColor = '#525252'
        }

        if (isHov) {
          const g = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, drawR)
          g.addColorStop(0, nodeColor); g.addColorStop(1, '#000')
          ctx.fillStyle = g
        } else { ctx.fillStyle = nodeColor }

        ctx.fill()

        // Borders
        ctx.strokeStyle = isHov ? '#fff' : '#525252'
        ctx.lineWidth = isHov ? 3 : 2
        ctx.stroke()

        const shortLabel = node.accId || node.label
        ctx.fillStyle = isHov ? '#fff' : '#d1d5db'
        ctx.font = `600 ${isHov ? 8 : 7}px 'JetBrains Mono', monospace`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        const centerText = String(shortLabel).length > 7 ? `${String(shortLabel).slice(0, 7)}…` : shortLabel
        ctx.fillText(centerText, node.x, node.y)
      }

      animRef.current = requestAnimationFrame(tick)
    }

    animRef.current = requestAnimationFrame(tick)
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current) }
  }, [ring])

  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    let found = null
    for (const node of nodesRef.current) {
      const dx = mx - node.x, dy = my - node.y
      if (dx * dx + dy * dy <= (node.r + 6) * (node.r + 6)) { found = node; break }
    }
    hoveredRef.current = found
    if (found) {
      setHoveredNode(found)
      setTooltipPos({ x: found.x, y: found.y - found.r - 12 })
      canvas.style.cursor = 'pointer'
    } else { setHoveredNode(null); canvas.style.cursor = 'default' }
  }, [])

  return (
    <div className="relative w-full aspect-[4/3] rounded-xl bg-[#0a0a0a] border border-white/[0.06] overflow-hidden"
      style={{
        backgroundImage:
          'radial-gradient(circle at 50% 30%, rgba(255,255,255,0.06) 0%, rgba(0,0,0,0) 55%), radial-gradient(circle, rgba(100,100,100,0.12) 0.6px, transparent 0.6px)',
        backgroundSize: '100% 100%, 16px 16px',
      }}>
      <canvas ref={canvasRef} className="w-full h-full" onMouseMove={handleMouseMove}
        onMouseLeave={() => { hoveredRef.current = null; setHoveredNode(null) }} />
      {hoveredNode && (
        <div className="node-tooltip absolute z-50 px-4 py-3 min-w-[220px]"
          style={{ left: tooltipPos.x, top: tooltipPos.y, transform: 'translate(-50%, -100%)' }}>
          <div className="flex items-center gap-2 mb-2">
            <div className="size-1.5 rounded-full bg-neutral-400"></div>
            <span className="text-[10px] uppercase font-semibold tracking-widest text-neutral-500">{hoveredNode.role}</span>
          </div>
          <p className="text-xs font-semibold text-white mb-2.5 font-technical">{hoveredNode.label}</p>
          <div className="space-y-1.5 text-[11px]">
            <div className="flex justify-between gap-4"><span className="text-neutral-500">Account ID</span><span className="text-white font-technical">{hoveredNode.accId}</span></div>
            <div className="flex justify-between gap-4"><span className="text-neutral-500">Transaction ID</span><span className="text-white font-technical">{hoveredNode.txnId}</span></div>
            <div className="flex justify-between gap-4"><span className="text-neutral-500">Amount</span><span className="text-white font-semibold font-technical">{hoveredNode.amount}</span></div>
            <div className="flex justify-between gap-4"><span className="text-neutral-500">Timestamp</span><span className="text-neutral-300 font-technical">{hoveredNode.timestamp}</span></div>
          </div>
        </div>
      )}
    </div>
  )
}


export default function FraudRings() {
  const { analysis } = useAnalysis()
  const navigate = useNavigate()

  const hasAnalysis = Boolean(analysis)
  const suspiciousAccounts = analysis?.suspicious_accounts ?? []

  // Normalize backend rings so UI always reads a stable shape.
  const backendRings = (analysis?.fraud_rings ?? []).map((ring) => {
    const memberAccounts = Array.isArray(ring?.member_accounts)
      ? ring.member_accounts.map((a) => String(a))
      : []
    const riskScore = Number(ring?.risk_score)
    const densityScore = Number(ring?.density_score)
    return {
      ring_id: String(ring?.ring_id ?? 'RING_UNKNOWN'),
      pattern_type: String(ring?.pattern_type ?? 'unknown'),
      member_accounts: memberAccounts,
      risk_score: Number.isFinite(riskScore) ? riskScore : 0,
      density_score: Number.isFinite(densityScore) ? densityScore : null,
    }
  })
  const hasBackendData = backendRings.length > 0

  const uiRings = hasBackendData
    ? backendRings
      .slice()
      .sort((a, b) => b.risk_score - a.risk_score)
    : hasAnalysis
      ? []
      : fraudRingsData

  const [activeRingIdx, setActiveRingIdx] = useState(0)
  const [selectedAccount, setSelectedAccount] = useState(null)
  const activeRing = uiRings[activeRingIdx] || uiRings[0]
  const hasRings = uiRings.length > 0

  useEffect(() => {
    setActiveRingIdx(0)
    setSelectedAccount(null)
  }, [analysis])

  useEffect(() => {
    if (!activeRing) {
      setSelectedAccount(null)
      return
    }
    if (hasBackendData) {
      setSelectedAccount(activeRing.member_accounts?.[0] ?? null)
    } else {
      setSelectedAccount(activeRing.entityList?.[0]?.id ?? null)
    }
  }, [activeRingIdx, activeRing, hasBackendData])

  const accountScoreMap = suspiciousAccounts.reduce((acc, item) => {
    const key = String(item?.account_id ?? '')
    if (key) {
      const score = Number(item?.suspicion_score)
      acc[key] = Number.isFinite(score) ? score : 0
    }
    return acc
  }, {})

  const backendActiveMembers = hasBackendData && activeRing
    ? activeRing.member_accounts.map((accountId) => ({
      id: accountId,
      score: accountScoreMap[accountId] ?? 0,
    }))
      .sort((a, b) => b.score - a.score)
    : []

  const previewConnectedIds = hasRings
    ? (
      hasBackendData
        ? activeRing.member_accounts.map((id) => String(id))
        : activeRing.entityList.map((entity) => String(entity.id))
    )
    : []
  const orderedPreviewIds = selectedAccount
    ? [selectedAccount, ...previewConnectedIds.filter((id) => id !== selectedAccount)]
    : previewConnectedIds
  const previewNodeIds = orderedPreviewIds.slice(0, 12)
  const previewOverflow = Math.max(0, orderedPreviewIds.length - previewNodeIds.length)
  const previewRingGraph = useMemo(() => {
    if (!hasRings) return null

    if (!hasBackendData) return activeRing

    const memberIds = activeRing.member_accounts.map((id) => String(id)).slice(0, 12)
    const nodes = memberIds.map((accountId, idx) => ({
      id: `n${idx + 1}`,
      label: accountId,
      role: accountId === selectedAccount ? 'Selected Member' : 'Member Account',
      r: accountId === selectedAccount ? 14 : 11,
      color: accountId === selectedAccount ? '#e5e7eb' : '#9ca3af',
      accId: accountId,
      txnId: '',
      amount: '',
      timestamp: '',
    }))

    const edges = []
    for (let i = 0; i < nodes.length; i++) {
      const from = nodes[i].id
      const to = nodes[(i + 1) % nodes.length]?.id
      if (!to) continue
      edges.push({ from, to, suspicious: true })
    }

    return {
      id: activeRing.ring_id,
      nodes,
      edges,
    }
  }, [hasRings, hasBackendData, activeRing, selectedAccount])

  return (
    <div className="bg-background-dark font-display text-text-muted overflow-hidden h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 flex flex-col overflow-hidden relative mt-24">
        {/* Edge Gradients */}
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-accent-blue via-accent-purple to-accent-red opacity-50"></div>

        <header className="h-auto flex items-end justify-between px-8 pb-8 shrink-0">
          <div>
            <h2 className="text-5xl font-medium text-white tracking-tight leading-none font-display mb-2">Active Investigations</h2>
            <p className="text-text-muted text-sm max-w-xl font-body">
              {hasBackendData
                ? 'Fraud rings detected by the latest backend run.'
                : hasAnalysis
                  ? 'No fraud rings were returned by the latest backend run.'
                  : 'Sample investigations. Upload data on Home to see live rings.'}
            </p>
          </div>
        </header>

        <div className="flex flex-1 overflow-hidden px-8 pb-6 gap-6">
          {/* Table */}
          <section className="flex-1 flex flex-col min-w-0 bg-card-dark border border-white/5 rounded-3xl overflow-hidden shadow-2xl">
            <div className="overflow-y-auto flex-1">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-card-dark z-10">
                  <tr className="border-b border-white/[0.04]">
                    <th className="px-6 py-5 text-[11px] font-semibold text-text-muted uppercase tracking-widest w-32 font-body">Ring ID</th>
                    <th className="px-6 py-5 text-[11px] font-semibold text-text-muted uppercase tracking-widest font-body">Pattern</th>
                    <th className="px-6 py-5 text-[11px] font-semibold text-text-muted uppercase tracking-widest text-right font-body">Entities</th>
                    <th className="px-6 py-5 text-[11px] font-semibold text-text-muted uppercase tracking-widest w-48 font-body">Risk Score</th>
                    <th className="px-6 py-5 text-[11px] font-semibold text-text-muted uppercase tracking-widest font-body">Member Account IDs</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.03]">
                  {hasRings ? uiRings.map((ring, idx) => {
                    const isActive = idx === activeRingIdx
                    const ringId = hasBackendData ? ring.ring_id : ring.id
                    const pattern = hasBackendData ? ring.pattern_type : ring.type
                    const entities = hasBackendData ? ring.member_accounts.length : ring.entities
                    const riskScore = hasBackendData ? ring.risk_score : ring.score
                    return (
                      <tr key={idx} onClick={() => setActiveRingIdx(idx)}
                        className={`cursor-pointer transition-all duration-200 group border-l-2 ${isActive ? 'bg-white/[0.02] border-accent-purple' : 'border-transparent hover:bg-white/[0.015]'}`}>
                        <td className={`px-6 py-4 font-technical text-xs ${isActive ? 'text-accent-purple font-semibold' : 'text-text-muted group-hover:text-white'}`}>#{ringId}</td>
                        <td className="px-6 py-4">
                          <span className={`px-2.5 py-1 rounded-md text-[10px] font-medium border ${isActive ? 'bg-accent-purple/10 text-accent-purple border-accent-purple/20' : 'bg-white/5 text-text-muted border-white/10'}`}>
                            {pattern}
                          </span>
                        </td>
                        <td className={`px-6 py-4 text-right text-xs font-technical ${isActive ? 'text-white' : 'text-text-muted'}`}>{entities}</td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className="w-full h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                              <div className={`h-full rounded-full transition-all duration-500 ${isActive ? 'bg-accent-purple' : 'bg-white/20'}`} style={{ width: `${Math.min(100, riskScore)}%` }}></div>
                            </div>
                            <span className={`text-xs font-technical w-10 text-right ${isActive ? 'text-white' : 'text-text-muted'}`}>{Math.round(riskScore)}</span>
                          </div>
                        </td>
                        <td className={`px-6 py-4 text-xs font-technical truncate max-w-[300px] ${isActive ? 'text-white' : 'text-text-muted'}`}>
                          {hasBackendData ? ring.member_accounts.join(', ') : ring.entityList.map(e => e.id).join(', ')}
                        </td>
                      </tr>
                    )
                  }) : (
                    <tr>
                      <td colSpan={5} className="px-6 py-12 text-center text-sm text-text-muted font-body">
                        {hasAnalysis
                          ? 'No fraud rings detected for the uploaded CSV.'
                          : 'Upload a CSV on Home to replace sample ring data with backend output.'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="border-t border-white/[0.04] p-4 flex justify-between items-center text-xs text-text-muted font-body">
              <span>Showing {uiRings.length} active rings</span>
              <div className="flex gap-2">
                <button className="px-3 py-1.5 bg-white/[0.03] hover:bg-white/[0.06] rounded-lg transition-colors text-text-muted">Prev</button>
                <button className="px-3 py-1.5 bg-white/[0.03] hover:bg-white/[0.06] rounded-lg transition-colors text-white">Next</button>
              </div>
            </div>
          </section>

          {/* Aside */}
          <aside className="w-[420px] rounded-3xl bg-card-dark flex flex-col overflow-y-auto border border-white/5 shadow-2xl">
            <div className="p-6 border-b border-white/[0.04]">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <p className="text-[10px] uppercase tracking-widest font-semibold text-text-muted mb-1 font-body">Active Cluster</p>
                  {hasRings ? (
                    <h3 className="text-4xl font-normal text-white font-display">
                      #{hasBackendData ? activeRing.ring_id : activeRing.id}
                    </h3>
                  ) : (
                    <h3 className="text-2xl font-normal text-white/60 font-display">No Active Ring</h3>
                  )}
                </div>
                <div className="text-right">
                  <p className="text-[10px] font-semibold text-text-muted uppercase mb-1 font-body">Risk Score</p>
                  <div className="flex items-baseline gap-0.5 justify-end">
                    <span className="text-4xl font-medium text-accent-red font-display">
                      {hasRings ? Math.round(hasBackendData ? activeRing.risk_score : activeRing.score) : 0}
                    </span>
                    <span className="text-xs text-text-muted font-display">/100</span>
                  </div>
                </div>
              </div>

              <div className="bg-white/[0.02] rounded-xl p-4 border border-white/[0.04] mb-6">
                <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-2 font-body">Pattern Logic</p>
                <p className="text-xs text-neutral-300 leading-relaxed font-body">
                  {!hasRings
                    ? 'Upload and analyze a CSV to inspect detected ring members and risk patterns.'
                    : hasBackendData
                    ? `Primary pattern: ${activeRing.pattern_type}. Member accounts: ${activeRing.member_accounts.length}.`
                    : activeRing.description}
                </p>
              </div>

              {hasRings && previewRingGraph && (
                <ForceGraph
                  ring={previewRingGraph}
                  key={`${hasBackendData ? activeRing.ring_id : activeRing.id}-${selectedAccount || 'none'}`}
                />
              )}
            </div>

            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-[10px] font-semibold text-text-muted uppercase tracking-widest font-body">
                  Entities ({hasRings ? (hasBackendData ? activeRing.member_accounts.length : activeRing.entities) : 0})
                </h4>
              </div>
              <div className="space-y-2">
                {!hasRings ? (
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                    <p className="text-xs text-text-muted font-body">No entities to display.</p>
                  </div>
                ) : hasBackendData
                  ? backendActiveMembers.slice(0, 20).map((member, idx) => (
                    <button
                      type="button"
                      key={idx}
                      onClick={() => setSelectedAccount(member.id)}
                      className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all group text-left ${selectedAccount === member.id ? 'bg-accent-purple/10 border-accent-purple/30' : 'bg-white/[0.02] border-white/[0.04] hover:bg-white/[0.04]'}`}
                    >
                      <div>
                        <p className="text-xs font-medium text-neutral-300 group-hover:text-white transition-colors font-technical">{member.id}</p>
                        <p className="text-[9px] text-text-muted uppercase mt-0.5 font-body">
                          {selectedAccount === member.id ? 'Selected' : 'Member Account'}
                        </p>
                      </div>
                      <span className="text-xs font-technical font-medium text-accent-red">{Math.round(member.score)}</span>
                    </button>
                  ))
                  : activeRing.entityList.map((entity, idx) => (
                    <button
                      type="button"
                      key={idx}
                      onClick={() => setSelectedAccount(entity.id)}
                      className={`w-full flex items-center justify-between p-3 rounded-xl border transition-all cursor-pointer group text-left ${selectedAccount === entity.id ? 'bg-accent-purple/10 border-accent-purple/30' : 'bg-white/[0.02] border-white/[0.04] hover:bg-white/[0.04]'}`}
                    >
                      <div>
                        <p className="text-xs font-medium text-neutral-300 group-hover:text-white transition-colors font-technical">{entity.id}</p>
                        <p className="text-[9px] text-text-muted uppercase mt-0.5 font-body">
                          {selectedAccount === entity.id ? 'Selected' : entity.role}
                        </p>
                      </div>
                      <span className="text-xs font-technical font-medium text-accent-red">{entity.score}</span>
                    </button>
                  ))}
              </div>
            </div>

            {hasRings && (
              <div className="px-6 pb-6">
                <div className="rounded-xl bg-white/[0.02] border border-white/[0.04] p-4">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-[10px] font-semibold text-text-muted uppercase tracking-widest font-body">Ring Preview</p>
                    <button
                      type="button"
                      onClick={() =>
                        navigate(
                          `/network-graph?ring=${encodeURIComponent(
                            hasBackendData ? activeRing.ring_id : activeRing.id,
                          )}${selectedAccount ? `&account=${encodeURIComponent(selectedAccount)}` : ''}`,
                        )
                      }
                      className="px-3 py-1 rounded-md text-[10px] font-semibold bg-white text-black hover:bg-neutral-200 transition-colors font-body"
                    >
                      Preview
                    </button>
                  </div>

                  <p className="text-[11px] text-neutral-400 mb-3 font-body">
                    {selectedAccount
                      ? `Selected ${selectedAccount} with ${previewConnectedIds.length} connected nodes in this ring.`
                      : `Showing ${previewConnectedIds.length} connected nodes in this ring.`}
                  </p>

                  <div className="flex flex-wrap gap-1.5">
                    {previewNodeIds.map((id) => (
                      <span
                        key={id}
                        className={`px-2 py-1 rounded-md text-[10px] font-technical border ${id === selectedAccount
                          ? 'bg-accent-purple/15 border-accent-purple/30 text-white'
                          : 'bg-white/[0.03] border-white/[0.08] text-neutral-300'
                          }`}
                      >
                        {id}
                      </span>
                    ))}
                    {previewOverflow > 0 && (
                      <span className="px-2 py-1 rounded-md text-[10px] font-technical border bg-white/[0.03] border-white/[0.08] text-neutral-400">
                        +{previewOverflow} more
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div className="p-6 mt-auto border-t border-white/[0.04]">
              <button
                disabled={!hasRings}
                onClick={() =>
                  navigate(
                    `/network-graph?ring=${encodeURIComponent(
                      hasBackendData ? activeRing.ring_id : activeRing.id,
                    )}${selectedAccount ? `&account=${encodeURIComponent(selectedAccount)}` : ''}`,
                  )
                }
                className="w-full py-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 bg-white text-black hover:bg-neutral-200 transition-colors font-body disabled:bg-white/20 disabled:text-white/50 disabled:cursor-not-allowed">
                <span className="material-symbols-outlined text-lg">schema</span>
                Investigate Full Graph
              </button>
            </div>
          </aside>
        </div>
      </main>
    </div>
  )
}
