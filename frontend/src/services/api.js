import axios from 'axios';

// Create a base API client
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5001/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// API methods for the requirements extractor
const apiService = {
  /**
   * Get available model providers
   * @returns {Promise} List of available model providers
   */
  getProviders: async () => {
    const response = await api.get('/providers');
    return response.data;
  },

  /**
   * Get configuration
   * @returns {Promise} Current application configuration
   */
  getConfig: async () => {
    const response = await api.get('/config');
    return response.data;
  },

  /**
   * Save configuration
   * @param {Object} config - Configuration to save
   * @returns {Promise} Result of the operation
   */
  saveConfig: async (config) => {
    const response = await api.post('/config', config);
    return response.data;
  },

  /**
   * Check Ollama status
   * @param {string} serverUrl - Ollama server URL
   * @returns {Promise} Status of Ollama server and available models
   */
  checkOllamaStatus: async (serverUrl) => {
    const response = await api.get('/ollama/status', { params: { serverUrl } });
    return response.data;
  },

  /**
   * Process PDF file
   * @param {FormData} formData - Form data containing file and configuration
   * @param {Function} onProgress - Progress callback
   * @returns {Promise} Processing result
   */
  processPdf: async (formData, onProgress) => {
    const response = await api.post('/process', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });
    return response.data;
  },

  /**
   * Process directory of PDFs
   * @param {FormData} formData - Form data containing directory path and configuration
   * @param {Function} onProgress - Progress callback
   * @returns {Promise} Processing result
   */
  processBatch: async (formData, onProgress) => {
    const response = await api.post('/process/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });
    return response.data;
  },
};

export default apiService;