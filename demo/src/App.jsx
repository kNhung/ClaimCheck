import React, { useState } from 'react'
import { verifyClaim } from './api/factcheck.js'
import './App.css'

function App() {
  const [claim, setClaim] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  const handleSubmit = async () => {
    console.log('Submit button clicked')
    
    if (!claim.trim()) {
      setError('Vui lòng nhập claim cần kiểm chứng')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    // Use today's date (format: DD-MM-YYYY for API)
    const today = new Date()
    const day = String(today.getDate()).padStart(2, '0')
    const month = String(today.getMonth() + 1).padStart(2, '0')
    const year = today.getFullYear()
    const apiDate = `${day}-${month}-${year}`
    console.log('Calling verifyClaim API with:', { claim: claim.trim(), date: apiDate })

    try {
      const response = await verifyClaim({
        claim: claim.trim(),
        date: apiDate,
      })
      console.log('API response received:', response)
      setResult(response)
      
      // Add to history
      const historyItem = {
        id: response.report_id || Date.now().toString(),
        claim: claim.trim(),
        verdict: response.verdict,
        date: apiDate,
        timestamp: new Date().toISOString(),
        fullResult: response
      }
      setHistory(prev => [historyItem, ...prev])
    } catch (err) {
      console.error('API error:', err)
      console.error('Error details:', err.response?.data)
      setError(err.response?.data?.detail || err.message || 'Đã xảy ra lỗi khi kiểm chứng')
    } finally {
      setLoading(false)
      console.log('Request completed')
    }
  }

  const handleHistoryClick = (historyItem) => {
    setResult(historyItem.fullResult)
  }

  const getVerdictClass = (verdict) => {
    if (!verdict) return ''
    return `verdict-${verdict.toLowerCase().replace(/\s+/g, '-')}`
  }

  return (
    <div className="app">
      <div className="sidebar">
        <h3 className="sidebar-title">Lịch sử kiểm chứng</h3>
        {history.length === 0 ? (
          <p className="sidebar-empty">Chưa có kết quả nào</p>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <div
                key={item.id}
                className={`history-item ${result?.report_id === item.id ? 'active' : ''}`}
                onClick={() => handleHistoryClick(item)}
              >
                <div className="history-claim">{item.claim}</div>
                <div className={`history-verdict ${getVerdictClass(item.verdict)}`}>
                  {item.verdict}
                </div>
                <div className="history-date">{item.date}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="main-content">
      <div className="hero-section">
        <div className="hero-content">
          <h1 className="hero-title">
            Vietnamese Fact-checking<br />
            <span className="hero-subtitle">by HCMZooS</span>
          </h1>
        </div>
        <div className="hero-divider"></div>
        <div className="hero-description-wrapper">
          <p className="hero-description">
            Ứng dụng trí tuệ nhân tạo giúp bạn kiểm chứng thông tin nhanh chóng, chính xác và đáng tin cậy.
          </p>
        </div>
      </div>

      <div className="container">
        <div className="input-section">
          <div className="input-row">
            <textarea
              id="claim"
              value={claim}
              onChange={(e) => setClaim(e.target.value)}
              placeholder="Nhập thông tin cần kiểm chứng..."
              disabled={loading}
              className="claim-input"
            />
          </div>
          <button 
            type="button" 
            onClick={handleSubmit}
            disabled={loading} 
            className="submit-btn"
          >
            {loading ? 'Đang kiểm chứng...' : 'Kiểm chứng'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            <strong>Lỗi:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-container">
            <h2>Kết quả kiểm chứng</h2>
            <div className="result-content">
              <div className="result-item">
                <strong>Verdict:</strong>
                <span className={`verdict verdict-${result.verdict?.toLowerCase().replace(/\s+/g, '-')}`}>
                  {result.verdict}
                </span>
              </div>
              <div className="result-item">
                <strong>Report ID:</strong>
                <span>{result.report_id}</span>
              </div>
              <div className="result-item">
                <strong>Claim:</strong>
                <span>{result.claim}</span>
              </div>
              <div className="result-item">
                <strong>Ngày:</strong>
                <span>{result.date}</span>
              </div>
              {result.model && (
                <div className="result-item">
                  <strong>Model:</strong>
                  <span>{result.model}</span>
                </div>
              )}
              <div className="result-item">
                <strong>Report Path:</strong>
                <span className="report-path">{result.report_path}</span>
              </div>
            </div>
          </div>
        )}
      </div>
      </div>
    </div>
  )
}

export default App

