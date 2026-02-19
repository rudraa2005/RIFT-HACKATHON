import { useEffect, useRef } from "react"
import * as d3 from "d3"

/* Fixed-size auto-rotating wireframe globe — sits behind the hero title */
export default function RotatingEarth({ size = 420, className = "" }) {
    const canvasRef = useRef(null)

    useEffect(() => {
        if (!canvasRef.current) return
        const canvas = canvasRef.current
        const context = canvas.getContext("2d")
        if (!context) return

        const dpr = window.devicePixelRatio || 1
        canvas.width = size * dpr
        canvas.height = size * dpr
        canvas.style.width = `${size}px`
        canvas.style.height = `${size}px`
        context.scale(dpr, dpr)

        const radius = size / 2.3

        const projection = d3.geoOrthographic()
            .scale(radius)
            .translate([size / 2, size / 2])
            .clipAngle(90)

        const path = d3.geoPath().projection(projection).context(context)

        const pointInPolygon = (point, polygon) => {
            const [x, y] = point
            let inside = false
            for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
                const [xi, yi] = polygon[i], [xj, yj] = polygon[j]
                if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
            }
            return inside
        }

        const pointInFeature = (point, feature) => {
            const g = feature.geometry
            if (g.type === "Polygon") {
                if (!pointInPolygon(point, g.coordinates[0])) return false
                for (let i = 1; i < g.coordinates.length; i++) if (pointInPolygon(point, g.coordinates[i])) return false
                return true
            } else if (g.type === "MultiPolygon") {
                for (const poly of g.coordinates) {
                    if (pointInPolygon(point, poly[0])) {
                        let hole = false
                        for (let i = 1; i < poly.length; i++) if (pointInPolygon(point, poly[i])) { hole = true; break }
                        if (!hole) return true
                    }
                }
            }
            return false
        }

        const allDots = []
        let landFeatures = null

        const render = () => {
            context.clearRect(0, 0, size, size)

            // Globe circle
            context.beginPath()
            context.arc(size / 2, size / 2, projection.scale(), 0, 2 * Math.PI)
            context.fillStyle = "rgba(0, 0, 0, 0.15)"
            context.fill()
            context.strokeStyle = "rgba(26, 137, 23, 0.15)"
            context.lineWidth = 1.5
            context.stroke()

            if (landFeatures) {
                // Graticule
                const graticule = d3.geoGraticule()
                context.beginPath()
                path(graticule())
                context.strokeStyle = "rgba(26, 137, 23, 0.06)"
                context.lineWidth = 0.5
                context.stroke()

                // Land outlines
                context.beginPath()
                landFeatures.features.forEach((f) => path(f))
                context.strokeStyle = "rgba(26, 137, 23, 0.18)"
                context.lineWidth = 0.8
                context.stroke()

                // Dots
                allDots.forEach((dot) => {
                    const p = projection([dot[0], dot[1]])
                    if (p && p[0] >= 0 && p[0] <= size && p[1] >= 0 && p[1] <= size) {
                        context.beginPath()
                        context.arc(p[0], p[1], 0.7, 0, 2 * Math.PI)
                        context.fillStyle = "rgba(26, 137, 23, 0.3)"
                        context.fill()
                    }
                })
            }
        }

        // Load world data
        fetch("https://raw.githubusercontent.com/martynafford/natural-earth-geojson/refs/heads/master/110m/physical/ne_110m_land.json")
            .then(r => r.json())
            .then(data => {
                landFeatures = data
                data.features.forEach(feature => {
                    const bounds = d3.geoBounds(feature)
                    const [[minLng, minLat], [maxLng, maxLat]] = bounds
                    const step = 16 * 0.08
                    for (let lng = minLng; lng <= maxLng; lng += step) {
                        for (let lat = minLat; lat <= maxLat; lat += step) {
                            if (pointInFeature([lng, lat], feature)) allDots.push([lng, lat])
                        }
                    }
                })
                render()
            })
            .catch(() => { })

        // Auto-rotation only — no drag/zoom
        const rotation = [0, -15]
        const timer = d3.timer(() => {
            rotation[0] += 0.25
            projection.rotate(rotation)
            render()
        })

        return () => timer.stop()
    }, [size])

    return (
        <canvas
            ref={canvasRef}
            className={className}
            style={{ width: size, height: size, pointerEvents: 'none' }}
        />
    )
}
