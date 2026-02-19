import { Link } from 'react-router-dom'

export default function Footer() {
    return (
        <footer className="bg-[rgb(20,20,20)] border-t border-[rgb(36,36,36)] mt-auto">
            <div className="max-w-7xl mx-auto px-8 py-10">
                <div className="grid grid-cols-4 gap-12 mb-10">
                    <div>
                        <div className="flex items-center gap-2 mb-4">
                            <div className="w-6 h-6 rounded bg-primary flex items-center justify-center">
                                <span className="material-symbols-outlined text-white text-[14px]">hub</span>
                            </div>
                            <span className="font-semibold text-white text-sm">GraphLens</span>
                        </div>
                        <p className="text-xs text-[rgb(107,107,107)] leading-relaxed">
                            Financial network intelligence platform for fraud detection and risk analysis.
                        </p>
                    </div>
                    <div>
                        <h4 className="text-[11px] font-semibold text-[rgb(107,107,107)] uppercase tracking-widest mb-4">Product</h4>
                        <ul className="space-y-2">
                            <li><Link to="/network-graph" className="text-xs text-[rgb(107,107,107)] hover:text-white transition-colors">Network Graph</Link></li>
                            <li><Link to="/fraud-rings" className="text-xs text-[rgb(107,107,107)] hover:text-white transition-colors">Fraud Rings</Link></li>
                            <li><Link to="/analytics" className="text-xs text-[rgb(107,107,107)] hover:text-white transition-colors">Analytics</Link></li>
                            <li><Link to="/reports" className="text-xs text-[rgb(107,107,107)] hover:text-white transition-colors">Reports</Link></li>
                            <li><Link to="/history" className="text-xs text-[rgb(107,107,107)] hover:text-white transition-colors">History</Link></li>
                        </ul>
                    </div>
                    <div>
                        <h4 className="text-[11px] font-semibold text-[rgb(107,107,107)] uppercase tracking-widest mb-4">Resources</h4>
                        <ul className="space-y-2">
                            <li><span className="text-xs text-[rgb(107,107,107)]">Documentation</span></li>
                            <li><span className="text-xs text-[rgb(107,107,107)]">API Reference</span></li>
                            <li><span className="text-xs text-[rgb(107,107,107)]">Changelog</span></li>
                        </ul>
                    </div>
                    <div>
                        <h4 className="text-[11px] font-semibold text-[rgb(107,107,107)] uppercase tracking-widest mb-4">Company</h4>
                        <ul className="space-y-2">
                            <li><span className="text-xs text-[rgb(107,107,107)]">About</span></li>
                            <li><span className="text-xs text-[rgb(107,107,107)]">Privacy</span></li>
                            <li><span className="text-xs text-[rgb(107,107,107)]">Terms</span></li>
                        </ul>
                    </div>
                </div>
                <div className="border-t border-[rgb(36,36,36)] pt-5 flex items-center justify-between">
                    <p className="text-[11px] text-[rgb(60,60,60)]">© 2024 GraphLens. All rights reserved.</p>
                    <span className="text-[11px] text-[rgb(60,60,60)]">v2.4.1</span>
                </div>
            </div>
        </footer>
    )
}
