import axios from 'axios';
import apiService from './api';

// Mock the axios module
jest.mock('axios');

describe('API Service', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  test('getProviders calls the correct endpoint', async () => {
    // Setup
    const mockResponse = {
      data: [
        { id: 'openai', name: 'OpenAI', models: ['gpt-4o-mini', 'gpt-4'] },
        { id: 'anthropic', name: 'Anthropic', models: ['claude-3-opus', 'claude-3-sonnet'] }
      ]
    };
    axios.get.mockResolvedValue(mockResponse);

    // Execute
    const result = await apiService.getProviders();

    // Assert
    expect(axios.get).toHaveBeenCalledWith('/providers');
    expect(result).toEqual(mockResponse.data);
  });

  test('getConfig calls the correct endpoint', async () => {
    // Setup
    const mockResponse = {
      data: {
        app: { use_cache: true },
        providers: { openai: { enabled: true } },
        extraction: { provider: 'openai' }
      }
    };
    axios.get.mockResolvedValue(mockResponse);

    // Execute
    const result = await apiService.getConfig();

    // Assert
    expect(axios.get).toHaveBeenCalledWith('/config');
    expect(result).toEqual(mockResponse.data);
  });

  test('saveConfig calls the correct endpoint with data', async () => {
    // Setup
    const mockConfig = {
      app: { use_cache: false },
      providers: { openai: { enabled: false } }
    };
    const mockResponse = {
      data: { success: true }
    };
    axios.post.mockResolvedValue(mockResponse);

    // Execute
    const result = await apiService.saveConfig(mockConfig);

    // Assert
    expect(axios.post).toHaveBeenCalledWith('/config', mockConfig);
    expect(result).toEqual(mockResponse.data);
  });

  test('checkOllamaStatus calls the correct endpoint with serverUrl', async () => {
    // Setup
    const serverUrl = 'http://localhost:11434';
    const mockResponse = {
      data: { isRunning: true, models: ['llama3'] }
    };
    axios.get.mockResolvedValue(mockResponse);

    // Execute
    const result = await apiService.checkOllamaStatus(serverUrl);

    // Assert
    expect(axios.get).toHaveBeenCalledWith('/ollama/status', { params: { serverUrl } });
    expect(result).toEqual(mockResponse.data);
  });

  test('processPdf calls the correct endpoint with formData and reports progress', async () => {
    // Setup
    const mockFormData = new FormData();
    mockFormData.append('file', new Blob(['test']), 'test.pdf');
    
    const mockResponse = {
      data: { success: true, requirementsCount: 10 }
    };
    axios.post.mockResolvedValue(mockResponse);
    
    const mockProgressCallback = jest.fn();

    // Execute
    const result = await apiService.processPdf(mockFormData, mockProgressCallback);

    // Assert
    expect(axios.post).toHaveBeenCalledWith(
      '/process',
      mockFormData,
      expect.objectContaining({
        headers: { 'Content-Type': 'multipart/form-data' }
      })
    );
    expect(result).toEqual(mockResponse.data);
  });

  test('processBatch calls the correct endpoint with formData', async () => {
    // Setup
    const mockFormData = new FormData();
    mockFormData.append('directory', new Blob(['test']), 'folder');
    
    const mockResponse = {
      data: { success: true, totalFiles: 3 }
    };
    axios.post.mockResolvedValue(mockResponse);

    // Execute
    const result = await apiService.processBatch(mockFormData);

    // Assert
    expect(axios.post).toHaveBeenCalledWith(
      '/process/batch',
      mockFormData,
      expect.objectContaining({
        headers: { 'Content-Type': 'multipart/form-data' }
      })
    );
    expect(result).toEqual(mockResponse.data);
  });

  test('handles errors from the API', async () => {
    // Setup
    const errorMessage = 'Network Error';
    axios.get.mockRejectedValue(new Error(errorMessage));

    // Execute and Assert
    await expect(apiService.getProviders()).rejects.toThrow(errorMessage);
  });
});