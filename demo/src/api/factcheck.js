import axios from 'axios'

// Luôn dùng /api trong production (qua Nginx proxy)
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  // Timeout 10 phút (600000ms) - đủ cho quá trình fact-checking
  timeout: 600000,
})

/**
 * Verify a claim using the fact-checking API
 * @param {Object} data - Claim data
 * @param {string} data.claim - The claim to verify
 * @param {string} data.date - Cut-off date in DD-MM-YYYY format
 * @param {number} [data.max_actions] - Maximum number of search actions (optional)
 * @param {string} [data.model_name] - Model name (optional)
 * @returns {Promise<Object>} Fact-check response
 */
export const verifyClaim = async (data) => {
  console.log('API: verifyClaim called with:', data)
  console.log('API: baseURL:', API_BASE_URL)
  console.log('API: full URL will be:', `${API_BASE_URL}/factcheck/verify`)
  
  try {
    const response = await api.post('/factcheck/verify', data)
    console.log('API: Response received:', response)
    return response.data
  } catch (error) {
    console.error('API: Error in verifyClaim:', error)
    console.error('API: Error config:', error.config)
    throw error
  }
}

/**
 * Get report by ID
 * @param {string} reportId - Report identifier
 * @returns {Promise<Object>} Report data
 */
export const getReport = async (reportId) => {
  const response = await api.get(`/reports/${reportId}`)
  return response.data
}

/**
 * Get report as JSON
 * @param {string} reportId - Report identifier
 * @returns {Promise<Object>} Report JSON data
 */
export const getReportJson = async (reportId) => {
  const response = await api.get(`/reports/${reportId}/json`)
  return response.data
}

/**
 * Get report as Markdown
 * @param {string} reportId - Report identifier
 * @returns {Promise<string>} Report Markdown content
 */
export const getReportMarkdown = async (reportId) => {
  const response = await api.get(`/reports/${reportId}/markdown`)
  return response.data
}

/**
 * Check health status
 * @returns {Promise<Object>} Health status
 */
export const checkHealth = async () => {
  const response = await api.get('/health')
  return response.data
}

export default api

