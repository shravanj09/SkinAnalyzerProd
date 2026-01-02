import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data'
  },
  timeout: 300000 // 300 seconds (5 minutes) - ml-custom needs 90s + buffer for all services
})

/**
 * Analyze facial image
 * @param {Blob} imageBlob - Image blob from camera
 * @returns {Promise<Object>} Analysis results
 */
export async function analyzeImage(imageBlob) {
  const formData = new FormData()
  formData.append('image', imageBlob, 'capture.jpg')

  try {
    const response = await apiClient.post('/api/v1/analyze', formData, {
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        console.log(`Upload progress: ${percentCompleted}%`)
      }
    })

    return response.data
  } catch (error) {
    console.error('API Error:', error)

    if (error.response) {
      // Server responded with error
      throw new Error(error.response.data?.detail || `Server error: ${error.response.status}`)
    } else if (error.request) {
      // Request made but no response
      throw new Error('No response from server. Please check if the API is running.')
    } else {
      // Something else happened
      throw new Error(error.message || 'Failed to analyze image')
    }
  }
}

/**
 * Get available models
 * @returns {Promise<Object>} Model information
 */
export async function getAvailableModels() {
  const response = await apiClient.get('/api/v1/models/available')
  return response.data
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await apiClient.get('/health')
  return response.data
}
