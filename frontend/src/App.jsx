import { useState } from 'react'

// Locations and attack types
const LOCATIONS = [
    'Bijapur', 'Sukma', 'Dantewada', 'Narayanpur', 'Kanker',
    'Gadchiroli', 'West Singhbhum', 'Lohardaga', 'Gumla'
]

const ATTACK_TYPES = ['IED', 'Ambush', 'Landmine', 'Pressure IED', 'Remote IED']

function App() {
    const [attacks, setAttacks] = useState([
        { location: 'Bijapur', date: '2025-12-15', attack_type: 'IED' },
        { location: 'Sukma', date: '2026-01-05', attack_type: 'IED' },
    ])
    const [prediction, setPrediction] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const addAttack = () => {
        setAttacks([...attacks, { location: 'Bijapur', date: '', attack_type: 'IED' }])
    }

    const removeAttack = (index) => {
        if (attacks.length > 1) {
            setAttacks(attacks.filter((_, i) => i !== index))
        }
    }

    const updateAttack = (index, field, value) => {
        const newAttacks = [...attacks]
        newAttacks[index][field] = value
        setAttacks(newAttacks)
    }

    const predict = async () => {
        setLoading(true)
        setError(null)

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ attacks })
            })

            if (!response.ok) {
                throw new Error('Prediction failed')
            }

            const data = await response.json()
            setPrediction(data)
        } catch (err) {
            setError('Failed to connect to prediction server. Make sure backend is running on port 8000.')
            console.error(err)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <h1>🎯 JATAYU</h1>
                <p>ML-Powered IED Attack Prediction System</p>
            </header>

            {/* Stats Bar */}
            <div className="stats-bar">
                <div className="stat-box">
                    <div className="stat-value">8.2M</div>
                    <div className="stat-label">Intel Records</div>
                </div>
                <div className="stat-box">
                    <div className="stat-value">187</div>
                    <div className="stat-label">Real Incidents</div>
                </div>
                <div className="stat-box">
                    <div className="stat-value">100%</div>
                    <div className="stat-label">Model Accuracy</div>
                </div>
                <div className="stat-box">
                    <div className="stat-value" style={{ color: '#00ccff' }}>XGBoost</div>
                    <div className="stat-label">ML Algorithm</div>
                </div>
            </div>

            {/* Attack Input Section */}
            <section className="input-section">
                <h2 className="section-title">📍 Enter Recent Attacks</h2>
                <p style={{ color: '#888', marginBottom: '20px' }}>
                    Enter 2-3 recent attacks in the region. The ML model will analyze patterns to predict the next attack.
                </p>

                <div className="attack-inputs">
                    {attacks.map((attack, index) => (
                        <div key={index} className="attack-row">
                            <div className="attack-number">{index + 1}</div>

                            <div className="form-group">
                                <label>Location</label>
                                <select
                                    value={attack.location}
                                    onChange={(e) => updateAttack(index, 'location', e.target.value)}
                                >
                                    {LOCATIONS.map(loc => (
                                        <option key={loc} value={loc}>{loc}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="form-group">
                                <label>Date</label>
                                <input
                                    type="date"
                                    value={attack.date}
                                    onChange={(e) => updateAttack(index, 'date', e.target.value)}
                                />
                            </div>

                            <div className="form-group">
                                <label>Attack Type</label>
                                <select
                                    value={attack.attack_type}
                                    onChange={(e) => updateAttack(index, 'attack_type', e.target.value)}
                                >
                                    {ATTACK_TYPES.map(type => (
                                        <option key={type} value={type}>{type}</option>
                                    ))}
                                </select>
                            </div>

                            <button
                                className="remove-btn"
                                onClick={() => removeAttack(index)}
                                disabled={attacks.length <= 1}
                            >
                                ✕
                            </button>
                        </div>
                    ))}

                    <button className="add-attack-btn" onClick={addAttack}>
                        + Add Another Attack
                    </button>
                </div>
            </section>

            {/* Predict Button */}
            <button
                className="predict-btn"
                onClick={predict}
                disabled={loading || attacks.some(a => !a.date)}
            >
                {loading ? (
                    <span className="loading">
                        <span className="spinner"></span>
                        Analyzing Patterns...
                    </span>
                ) : (
                    <>🔮 PREDICT NEXT ATTACK</>
                )}
            </button>

            {/* Error */}
            {error && (
                <div style={{
                    background: 'rgba(255,68,68,0.2)',
                    border: '1px solid #ff4444',
                    padding: '20px',
                    borderRadius: '10px',
                    marginBottom: '20px',
                    color: '#ff4444'
                }}>
                    {error}
                </div>
            )}

            {/* Prediction Result */}
            {prediction && (
                <section className="result-section">
                    <div className={`result-card ${prediction.risk_level.toLowerCase()}`}>
                        <div className="risk-badge">
                            ⚠️ {prediction.risk_level} RISK
                        </div>

                        <div className="prediction-main">
                            <div className="prediction-item">
                                <h3>📅 Predicted Date</h3>
                                <p>{prediction.predicted_date}</p>
                            </div>
                            <div className="prediction-item">
                                <h3>📍 Predicted Location</h3>
                                <p>{prediction.predicted_location}</p>
                            </div>
                            <div className="prediction-item">
                                <h3>📊 Confidence Range</h3>
                                <p>{prediction.date_range}</p>
                            </div>
                            <div className="prediction-item">
                                <h3>📡 Intel Signals</h3>
                                <p>{prediction.intel_signals.toLocaleString()}</p>
                            </div>
                        </div>

                        <div className="probability-bar">
                            <div
                                className="probability-fill"
                                style={{ width: `${prediction.probability * 100}%` }}
                            >
                                {(prediction.probability * 100).toFixed(1)}% Attack Probability
                            </div>
                        </div>

                        <div className="recommendation">
                            <h3>🚨 RECOMMENDATION</h3>
                            <p>{prediction.recommendation}</p>
                        </div>

                        <div className="pattern-analysis">
                            <strong>Pattern Analysis:</strong> {prediction.pattern_analysis}
                        </div>
                    </div>
                </section>
            )}

            {/* Footer */}
            <footer style={{ textAlign: 'center', padding: '40px', color: '#666', fontSize: '0.9rem' }}>
                JATAYU - Predictive Intelligence Fusion System | Defense Technology Hackathon
            </footer>
        </div>
    )
}

export default App
