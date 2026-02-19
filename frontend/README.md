# GraphLens - Financial Network Intelligence System

A React-based financial fraud detection and network intelligence dashboard with real-time monitoring, graph visualization, and analytics.

## Features

- **Home Dashboard**: Overview of the system with live monitoring, suspicious activity tracking, and core intelligence modules
- **Network Graph**: Interactive graph visualization showing entity relationships and fraud patterns
- **Analytics**: Deep structural and temporal data analysis with charts and metrics
- **Fraud Rings**: Active fraud ring investigations with detailed entity information
- **Reports**: Comprehensive reports on suspicious accounts with risk scoring

## Tech Stack

- React 18
- React Router DOM 6
- Tailwind CSS 3
- Vite
- Google Material Symbols Icons

## Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

4. Preview production build:
```bash
npm run preview
```

## Project Structure

```
├── src/
│   ├── components/
│   │   └── Navbar.jsx          # Unified navigation component
│   ├── pages/
│   │   ├── Home.jsx             # Main landing page (index.html)
│   │   ├── NetworkGraph.jsx    # Graph visualization (code.html)
│   │   ├── Analytics.jsx       # System analytics (code copy.html)
│   │   ├── FraudRings.jsx      # Fraud investigations (code copy 2.html)
│   │   └── Reports.jsx         # Suspicious accounts (code copy 4.html)
│   ├── App.jsx                 # Main app with routing
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── index.html
├── package.json
├── tailwind.config.js
├── vite.config.js
└── postcss.config.js
```

## Design System

### Colors
- Primary: `#FF9408` (Vivid Orange)
- Primary Dark: `#CA3F16` (Deep Red/Orange)
- Accent Red: `#95122C`
- Background: `#050505` (Rich Black)
- Surface: `#101010`

### Typography
- Display Font: Clash Display, Inter
- Body Font: Inter
- Technical Font: Inter (monospace)

### Key Features
- Glass morphism effects
- Gradient backgrounds
- Animated components
- Responsive design
- Dark theme optimized

## Navigation

The navigation bar is consistent across all pages with active state indicators:
- Product (Home)
- Analytics
- Network Graphs
- Fraud Rings
- Reports

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

Proprietary - GraphLens Inc. © 2024
