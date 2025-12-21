import React, { useState, useEffect } from 'react'
import { verifyClaim, getReportMarkdown } from './api/factcheck.js'
import './App.css'

function App() {
  const [claim, setClaim] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const [reportData, setReportData] = useState(null)

  // Helper function to check if error is about invalid claim
  const isInvalidClaimError = (errorMsg) => {
    if (!errorMsg) return false
    return errorMsg.includes('Vui lòng nhập một câu claim hợp lệ') ||
      errorMsg.includes('không phải là một claim có thể kiểm chứng') ||
      errorMsg.includes('Lỗi. Hãy nhập thông tin cần kiểm chứng') ||
      errorMsg.includes('Lỗi. Hãy nhập 1 câu cần kiểm chứng')
  }

  const handleSubmit = async () => {
    console.log('Submit button clicked')

    if (!claim.trim()) {
      setError('Vui lòng nhập claim cần kiểm chứng')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setReportData(null) // Reset report data when submitting new claim

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

      // Check if verdict indicates an error case (null or invalid verdict with very short/invalid claim)
      const claimLength = claim.trim().length
      if (!response.verdict || response.verdict === null ||
        (response.verdict === 'Not Enough Evidence' && claimLength < 5)) {
        // Treat as error case - claim is too short or invalid
        const errorMessage = 'Lỗi. Hãy nhập thông tin cần kiểm chứng.'
        setError(errorMessage)
        setResult(null)
        setReportData(null)
      } else {
        setResult(response)
        setError(null) // Clear any previous errors

        // Fetch and parse report markdown
        if (response.report_id) {
          try {
            console.log('Fetching report markdown for report_id:', response.report_id)
            const markdown = await getReportMarkdown(response.report_id)
            console.log('Received markdown, length:', markdown?.length)
            const parsed = parseReportMarkdown(markdown)
            console.log('Parsed data:', parsed)
            setReportData(parsed)
          } catch (err) {
            console.error('Failed to fetch report markdown:', err)
            setReportData(null)
          }
        } else {
          console.log('No report_id in response')
        }

        // Add to history only if we have a valid result
        const historyItem = {
          id: response.report_id || Date.now().toString(),
          claim: claim.trim(),
          verdict: response.verdict,
          date: apiDate,
          timestamp: new Date().toISOString(),
          fullResult: response
        }
        setHistory(prev => [historyItem, ...prev])
      }
    } catch (err) {
      console.error('API error:', err)
      console.error('Error details:', err.response?.data)
      console.error('Error status:', err.response?.status)

      // Xử lý lỗi 504 Gateway Timeout
      if (err.response?.status === 504 || err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        const errorMessage = 'Quá trình kiểm chứng mất quá nhiều thời gian. Vui lòng thử lại với một claim ngắn gọn hơn hoặc kiểm tra lại sau.'
        setError(errorMessage)
        setLoading(false)
        return
      }

      // Extract error message from response
      let errorMessage = err.response?.data?.detail || err.message || 'Đã xảy ra lỗi khi kiểm chứng'

      // Check if error message indicates invalid claim (validation error)
      // Backend returns HTTP 400 with user-friendly message for invalid claims
      const isInvalidClaimError =
        errorMessage.includes('Vui lòng nhập một câu claim hợp lệ') ||
        errorMessage.includes('không phải là một claim có thể kiểm chứng') ||
        errorMessage.includes('Lỗi. Hãy nhập 1 câu cần kiểm chứng') ||
        errorMessage.includes('Filtered claim is empty') ||
        errorMessage.includes('empty')

      if (isInvalidClaimError) {
        // Use the message from backend if it's user-friendly, otherwise use default
        if (errorMessage.includes('Vui lòng nhập một câu claim hợp lệ')) {
          // Keep the backend message as-is (it's already user-friendly)
          errorMessage = errorMessage
        } else {
          // Fallback to simple message for old error formats
          errorMessage = 'Vui lòng nhập một câu claim hợp lệ để kiểm chứng. Câu bạn nhập không phải là một claim có thể kiểm chứng được.'
        }
      } else if (errorMessage.includes('Fact-checking failed:')) {
        // Extract the actual error message after "Fact-checking failed: "
        const actualError = errorMessage.replace('Fact-checking failed: ', '')
        if (actualError.includes('Vui lòng nhập một câu claim hợp lệ') ||
          actualError.includes('Lỗi. Hãy nhập 1 câu cần kiểm chứng')) {
          errorMessage = actualError.includes('Vui lòng nhập một câu claim hợp lệ')
            ? actualError
            : 'Vui lòng nhập một câu claim hợp lệ để kiểm chứng. Câu bạn nhập không phải là một claim có thể kiểm chứng được.'
        }
      }

      setError(errorMessage)
    } finally {
      setLoading(false)
      console.log('Request completed')
    }
  }

  const handleHistoryClick = async (historyItem) => {
    // Set the claim in the input field
    setClaim(historyItem.claim)
    setResult(historyItem.fullResult)
    // Fetch report data when clicking history item
    if (historyItem.fullResult?.report_id) {
      try {
        const markdown = await getReportMarkdown(historyItem.fullResult.report_id)
        const parsed = parseReportMarkdown(markdown)
        setReportData(parsed)
      } catch (err) {
        console.error('Failed to fetch report markdown:', err)
        setReportData(null)
      }
    }
  }

  // Parse report markdown to extract relevant information
  const parseReportMarkdown = (markdown) => {
    const data = {
      webSearchResults: [],
      evidenceSummary: []
    }

    if (!markdown) {
      console.log('parseReportMarkdown: No markdown provided')
      return data
    }

    console.log('parseReportMarkdown: Starting parse, markdown length:', markdown.length)

    // Parse BƯỚC 4: RAV sections
    const step4Regex = /📋 BƯỚC 4: RAV \(Evidence Ranking\) - (https?:\/\/[^\s]+)/g
    const step4Matches = [...markdown.matchAll(step4Regex)]

    console.log('parseReportMarkdown: Found', step4Matches.length, 'BƯỚC 4 sections')

    step4Matches.forEach((match, index) => {
      const url = match[1]
      const startIndex = match.index

      // Find the next BƯỚC 4 section or next major section
      let nextSectionIndex = markdown.indexOf('📋 BƯỚC 4:', startIndex + 1)
      if (nextSectionIndex < 0) {
        // Look for next major section divider
        nextSectionIndex = markdown.indexOf('📋 BƯỚC 6:', startIndex + 1)
      }
      if (nextSectionIndex < 0) {
        // Look for final sections
        nextSectionIndex = markdown.indexOf('📋 BƯỚC 7:', startIndex + 1)
      }

      const sectionContent = nextSectionIndex > 0
        ? markdown.substring(startIndex, nextSectionIndex)
        : markdown.substring(startIndex)

      console.log(`parseReportMarkdown: Processing BƯỚC 4 #${index + 1}, URL: ${url}, section length: ${sectionContent.length}`)

      // Extract "KẾT QUẢ:" summary - look for the line after "✅ KẾT QUẢ:"
      // But filter out "Đã chọn X chunk(s)...", "====", and score numbers
      const ketQuaMatch = sectionContent.match(/✅ KẾT QUẢ:[^\n]*(?:\n[^\n]*)?/s)
      let summary = ''
      if (ketQuaMatch) {
        let rawSummary = ketQuaMatch[0].replace(/✅ KẾT QUẢ:\s*/, '').trim()
        // Remove "Đã chọn X chunk(s)..." lines
        rawSummary = rawSummary.replace(/Đã chọn \d+ chunk\(s\)[^\n]*/gi, '')
        // Remove "====" lines
        rawSummary = rawSummary.replace(/=+\s*/g, '')
        // Remove score numbers - only remove score-specific patterns, keep content numbers
        rawSummary = rawSummary.replace(/\(score:\s*[0-9.-]+\s*\)/gi, '')
        rawSummary = rawSummary.replace(/score:\s*[0-9.-]+\s*/gi, '')
        rawSummary = rawSummary.replace(/Chunk\s+#\d+/gi, '')
        // Clean up multiple spaces
        rawSummary = rawSummary.replace(/\s+/g, ' ').trim()
        summary = rawSummary
        console.log(`parseReportMarkdown: Found summary for ${url}:`, summary.substring(0, 50))
      } else {
        console.log(`parseReportMarkdown: No summary found for ${url}`)
      }

      // Extract "Top 3 chunks sau khi re-rank" content
      const topChunksMatch = sectionContent.match(/Top 3 chunks sau khi re-rank \(cross-encoder scores\):(.*?)(?=✅ KẾT QUẢ:|📋 BƯỚC|==|$)/s)
      let topChunks = ''
      if (topChunksMatch) {
        // Extract all chunk texts after the header
        const chunksText = topChunksMatch[1]
        // Match each chunk line: [1] Chunk #X (score: Y): TEXT (may span multiple lines)
        // Pattern: [number] Chunk #number (score: number): TEXT
        const chunkRegex = /\[\d+\]\s+Chunk\s+#\d+\s*\(score:\s*[0-9.-]+\s*\):\s*(.*?)(?=\n\s*\[\d+\]|$)/gs
        const chunkMatches = [...chunksText.matchAll(chunkRegex)]
        if (chunkMatches.length > 0) {
          // Only extract the text part, remove any remaining score info
          topChunks = chunkMatches
            .map(match => {
              let text = match[1].trim()
              // Remove any remaining score patterns that might be in the text
              text = text.replace(/\(score:\s*[0-9.-]+\s*\)/gi, '')
              text = text.replace(/score:\s*[0-9.-]+\s*/gi, '')
              text = text.replace(/Chunk\s+#\d+/gi, '')
              // Clean up multiple spaces but keep content
              text = text.replace(/\s+/g, ' ').trim()
              return text
            })
            .filter(text => text.length > 0)
            .join('\n\n')
        }
      }

      data.webSearchResults.push({
        url,
        summary,
        topChunks
      })

      console.log(`parseReportMarkdown: Added web result #${index + 1}:`, {
        url,
        hasSummary: !!summary,
        hasTopChunks: !!topChunks
      })
    })

    console.log('parseReportMarkdown: Total webSearchResults:', data.webSearchResults.length)

    // Parse BƯỚC 6: Evidence selection - look for evidence after "🔍 BƯỚC 6: Chọn top_"
    const step6Index = markdown.indexOf('🔍 BƯỚC 6: Chọn top_')
    console.log('parseReportMarkdown: BƯỚC 6 index:', step6Index)

    if (step6Index > 0) {
      const step6Section = markdown.substring(step6Index)
      // Find [E0], [E1], [E2] content - pattern: [E0] Score: X.XXXX - TEXT
      const evidenceRegex = /\[E(\d+)\]\s+Score:[^\n-]*-\s*(.*?)(?=\n\s*\[E\d+\]|\n==|$)/gs
      const evidenceMatches = [...step6Section.matchAll(evidenceRegex)]

      console.log('parseReportMarkdown: Found', evidenceMatches.length, 'evidence matches')

      evidenceMatches.forEach(match => {
        const index = parseInt(match[1])
        let text = match[2].trim()
        // Clean up text: remove any score references that might remain
        text = text.replace(/Score:\s*[0-9.-]+\s*/gi, '')
        text = text.replace(/\(score:\s*[0-9.-]+\s*\)/gi, '')
        text = text.trim()

        if (index <= 2 && text) {
          data.evidenceSummary.push({
            index: index,
            text: text
          })
          console.log(`parseReportMarkdown: Added evidence [E${index}]`)
        }
      })
    } else {
      console.log('parseReportMarkdown: BƯỚC 6 section not found')
    }

    console.log('parseReportMarkdown: Final data:', {
      webSearchResultsCount: data.webSearchResults.length,
      evidenceSummaryCount: data.evidenceSummary.length
    })

    return data
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

          {error && !isInvalidClaimError(error) && (
            <div className="error-message">
              <strong>Lỗi:</strong> {error}
            </div>
          )}

          {(result || (error && isInvalidClaimError(error))) && (
            <div className="result-container">
              <h2>Kết quả kiểm chứng</h2>
              <div className="result-content">
                {error && isInvalidClaimError(error) ? (
                  <div className="result-verdict">
                    <span className="verdict verdict-error">
                      {error}
                    </span>
                  </div>
                ) : result && (
                  <div className="result-verdict">
                    <span className={`verdict verdict-${result.verdict?.toLowerCase().replace(/\s+/g, '-')}`}>
                      {result.verdict}
                    </span>
                  </div>
                )}

                {!error && reportData && reportData.webSearchResults && reportData.webSearchResults.length > 0 && (
                  <div className="result-section">
                    <h3>Kết quả tìm kiếm trang web:</h3>
                    {reportData.webSearchResults.map((item, idx) => (
                      <div key={idx} className="web-result-item">
                        <div className="web-result-url">
                          <a href={item.url} target="_blank" rel="noopener noreferrer">
                            {item.url}
                          </a>
                        </div>
                        {item.summary && (
                          <div className="web-result-summary">
                            <strong>Tổng hợp kết quả ở trang web:</strong> {item.summary}
                          </div>
                        )}
                        {item.topChunks && (
                          <div className="web-result-chunks">
                            <pre className="chunks-text">{item.topChunks}</pre>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {loading && (
                  <div className="result-section">
                    <p>Đang tải dữ liệu từ report...</p>
                  </div>
                )}

                {!error && reportData && reportData.evidenceSummary && reportData.evidenceSummary.length > 0 && (
                  <div className="result-section">
                    <h3>Bằng chứng sau cùng:</h3>
                    <ul className="evidence-list">
                      {reportData.evidenceSummary.map((item) => (
                        <li key={item.index} className="evidence-item">
                          {item.text}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App

