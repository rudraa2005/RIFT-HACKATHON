import { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'

const Home = lazy(() => import('./pages/Home'))
const NetworkGraph = lazy(() => import('./pages/NetworkGraph'))
const Analytics = lazy(() => import('./pages/Analytics'))
const FraudRings = lazy(() => import('./pages/FraudRings'))
const Reports = lazy(() => import('./pages/Reports'))
const History = lazy(() => import('./pages/History'))

function App() {
  return (
    <Router>
      <Suspense
        fallback={
          <div className="min-h-screen bg-background-dark text-white flex items-center justify-center font-body">
            Loading...
          </div>
        }
      >
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/network-graph" element={<NetworkGraph />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/fraud-rings" element={<FraudRings />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </Suspense>
    </Router>
  )
}

export default App
