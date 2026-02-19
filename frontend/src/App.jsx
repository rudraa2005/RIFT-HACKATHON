import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import NetworkGraph from './pages/NetworkGraph'
import Analytics from './pages/Analytics'
import FraudRings from './pages/FraudRings'
import Reports from './pages/Reports'
import History from './pages/History'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/network-graph" element={<NetworkGraph />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/fraud-rings" element={<FraudRings />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </Router>
  )
}

export default App
