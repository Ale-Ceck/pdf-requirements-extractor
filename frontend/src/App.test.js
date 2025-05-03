import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { ThemeProvider } from '@mui/material/styles';
import theme from './styles/theme';
import ApiService from './services/api';

// Mock the API service
jest.mock('./services/api', () => ({
  getConfig: jest.fn(),
  getProviders: jest.fn(),
  processPdf: jest.fn(),
  processBatch: jest.fn(),
  saveConfig: jest.fn()
}));

// Create a wrapper component to provide the theme
const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('App Component', () => {
  beforeEach(() => {
    // Reset the mocks
    jest.clearAllMocks();
    
    // Mock API responses
    ApiService.getConfig.mockResolvedValue({
      app: { use_offline_provider: false },
      providers: {
        openai: { enabled: true, api_key: null },
        anthropic: { enabled: false },
        ollama: { enabled: false }
      },
      extraction: { provider: 'openai', model: 'gpt-4o-mini' }
    });
    
    ApiService.getProviders.mockResolvedValue([
      { id: 'openai', name: 'OpenAI', models: ['gpt-4o-mini', 'gpt-4'] },
      { id: 'anthropic', name: 'Anthropic', models: ['claude-3-opus'] },
      { id: 'ollama', name: 'Ollama', models: ['llama3'] }
    ]);
    
    ApiService.processPdf.mockResolvedValue({
      success: true,
      requirementsCount: 10,
      outputFile: 'requirements.xlsx'
    });
    
    ApiService.saveConfig.mockResolvedValue({ success: true });
  });

  test('renders header and main components', async () => {
    await act(async () => {
      renderWithTheme(<App />);
    });
    
    // Check if header is rendered
    const headerElement = screen.getByText(/PDF Requirements Extractor/i);
    expect(headerElement).toBeInTheDocument();
    
    // Check if operation mode selector is rendered
    const onlineButton = screen.getByText(/Online \(API\)/i);
    const offlineButton = screen.getByText(/Offline \(Local\)/i);
    expect(onlineButton).toBeInTheDocument();
    expect(offlineButton).toBeInTheDocument();
  });

  test('initializes with configuration from API', async () => {
    await act(async () => {
      renderWithTheme(<App />);
    });
    
    await waitFor(() => {
      expect(ApiService.getConfig).toHaveBeenCalled();
      expect(ApiService.getProviders).toHaveBeenCalled();
    });
    
    // Check if online mode is selected (from mock response)
    const onlineWarning = screen.getByText(/Warning: Online mode sends data to external API services/i);
    expect(onlineWarning).toBeInTheDocument();
  });

  test('changes operation mode correctly', async () => {
    await act(async () => {
      renderWithTheme(<App />);
    });
    
    // Find and click the offline mode button
    const offlineButton = screen.getByText(/Offline \(Local\)/i);
    await act(async () => {
      userEvent.click(offlineButton);
    });
    
    // Check if mode has changed
    const offlineMessage = screen.getByText(/Offline mode: All processing happens locally/i);
    expect(offlineMessage).toBeInTheDocument();
  });

  test('shows empty log state initially', async () => {
    await act(async () => {
      renderWithTheme(<App />);
    });
    
    // Check if empty log message is displayed
    const emptyLogMessage = screen.getByText(/No processing logs to display/i);
    expect(emptyLogMessage).toBeInTheDocument();
  });

  // We won't test the file upload process here as it's more complex and would require mocking the File API
  // Those tests would be better suited for component tests
});