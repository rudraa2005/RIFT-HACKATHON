/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "primary": "#ffffff", // White text
        "accent-blue": "#3b82f6", // Blue
        "accent-purple": "#8b5cf6", // Purple
        "accent-red": "#ef4444", // Red
        "surface": "#000000",
        "surface-highlight": "#0a0a0a",
        "border-subtle": "#27272a",
        "text-muted": "#a1a1aa", // Light grey
        "background-dark": "#000000",
        "card-dark": "#050505",
      },
      fontFamily: {
        "display": ["Instrument Serif", "serif"],
        "body": ["Instrument Sans", "sans-serif"],
        "technical": ["JetBrains Mono", "monospace"],
      },
      animation: {
        "spin-slow": "spin 15s linear infinite",
      },
    },
  },
  plugins: [],
}
