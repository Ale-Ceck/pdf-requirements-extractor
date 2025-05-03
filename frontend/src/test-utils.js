import React from 'react';
import { render } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import theme from './styles/theme';

// Custom render function that includes providers
const renderWithProviders = (ui, options = {}) => {
  // Wrap with all providers needed in the app
  const Wrapper = ({ children }) => {
    return (
      <ThemeProvider theme={theme}>
        {children}
      </ThemeProvider>
    );
  };
  
  return render(ui, { wrapper: Wrapper, ...options });
};

// Mock data factory functions
const createMockProviders = () => ([
  {
    id: 'openai',
    name: 'OpenAI',
    models: ['gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo'],
  },
  {
    id: 'anthropic',
    name: 'Anthropic',
    models: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
  },
  {
    id: 'ollama',
    name: 'Ollama',
    models: ['llama3', 'mistral-7b'],
  },
]);

const createMockConfig = () => ({
  app: {
    use_cache: true,
    cache_dir: '.requirement_cache',
    extract_tables: true,
    parallel_processing: true,
    max_workers: 3,
    chunk_size: 3,
    max_token_size: 4000,
    use_semantic_similarity: false,
    confidence_threshold: 0.8,
    retry_attempts: 3,
    adaptive_learning: true,
    patterns_file: 'requirement_patterns.json',
    use_offline_provider: false
  },
  providers: {
    openai: {
      api_key: null,
      default_model: 'gpt-4o-mini',
      enabled: true,
      provider_type: 'online'
    },
    anthropic: {
      api_key: null,
      default_model: 'claude-3-haiku-latest',
      enabled: false,
      provider_type: 'online'
    },
    ollama: {
      server_url: 'http://localhost:11434',
      default_model: 'llama3',
      enabled: false,
      provider_type: 'offline'
    }
  },
  extraction: {
    provider: 'openai',
    model: 'gpt-4o-mini',
    verification_strategy: 'different',
    verification_provider: null,
    verification_model: null
  }
});

const createMockLogs = () => ([
  {
    id: 1,
    message: 'Starting processing in ONLINE mode',
    type: 'info',
    timestamp: '12:34:56'
  },
  {
    id: 2,
    message: 'Using model: gpt-4o-mini',
    type: 'info',
    timestamp: '12:34:57'
  },
  {
    id: 3,
    message: 'Extracted 15 requirements',
    type: 'success',
    timestamp: '12:35:30'
  },
  {
    id: 4,
    message: 'Output saved to: requirements.xlsx',
    type: 'info',
    timestamp: '12:35:35'
  }
]);

// Export the custom render function and mock data factories
export {
  renderWithProviders,
  createMockProviders,
  createMockConfig,
  createMockLogs
};