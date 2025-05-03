import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ProcessingLog from './ProcessingLog';
import { ThemeProvider } from '@mui/material/styles';
import theme from '../styles/theme';

// Create a wrapper component to provide the theme
const renderWithTheme = (component) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('ProcessingLog Component', () => {
  const mockClearLogs = jest.fn();
  
  beforeEach(() => {
    mockClearLogs.mockClear();
  });

  test('renders empty state when no logs are provided', () => {
    renderWithTheme(
      <ProcessingLog 
        logs={[]} 
        progress={0} 
        isProcessing={false} 
        onClear={mockClearLogs} 
      />
    );
    
    // Check if empty state message is displayed
    const emptyStateMessage = screen.getByText(/No processing logs to display/i);
    expect(emptyStateMessage).toBeInTheDocument();
  });

  test('renders logs when provided', () => {
    const sampleLogs = [
      {
        id: 1,
        message: 'Starting processing in ONLINE mode',
        type: 'info',
        timestamp: '12:34:56'
      },
      {
        id: 2,
        message: 'Extracted 10 requirements',
        type: 'success',
        timestamp: '12:35:00'
      },
      {
        id: 3,
        message: 'Failed to process file',
        type: 'error',
        timestamp: '12:35:05'
      }
    ];
    
    renderWithTheme(
      <ProcessingLog 
        logs={sampleLogs} 
        progress={100} 
        isProcessing={false} 
        onClear={mockClearLogs} 
      />
    );
    
    // Check if logs are displayed
    const infoLog = screen.getByText(/Starting processing in ONLINE mode/i);
    const successLog = screen.getByText(/Extracted 10 requirements/i);
    const errorLog = screen.getByText(/Failed to process file/i);
    
    expect(infoLog).toBeInTheDocument();
    expect(successLog).toBeInTheDocument();
    expect(errorLog).toBeInTheDocument();
  });

  test('shows progress bar when processing', () => {
    renderWithTheme(
      <ProcessingLog 
        logs={[{ id: 1, message: 'Processing...', type: 'info', timestamp: '12:34:56' }]} 
        progress={50} 
        isProcessing={true} 
        onClear={mockClearLogs} 
      />
    );
    
    // Check if progress bar is rendered
    const progressBar = screen.getByRole('progressbar');
    expect(progressBar).toBeInTheDocument();
    expect(progressBar).toHaveAttribute('aria-valuenow', '50');
  });

  test('calls onClear when clear button is clicked', () => {
    renderWithTheme(
      <ProcessingLog 
        logs={[{ id: 1, message: 'Test log', type: 'info', timestamp: '12:34:56' }]} 
        progress={0} 
        isProcessing={false} 
        onClear={mockClearLogs} 
      />
    );
    
    // Find and click the clear button
    const clearButton = screen.getByText(/Clear/i);
    fireEvent.click(clearButton);
    
    // Check if onClear was called
    expect(mockClearLogs).toHaveBeenCalledTimes(1);
  });

  test('disables clear button when processing', () => {
    renderWithTheme(
      <ProcessingLog 
        logs={[{ id: 1, message: 'Processing...', type: 'info', timestamp: '12:34:56' }]} 
        progress={50} 
        isProcessing={true} 
        onClear={mockClearLogs} 
      />
    );
    
    // Find the clear button and check if it's disabled
    const clearButton = screen.getByText(/Clear/i);
    expect(clearButton).toBeDisabled();
  });

  test('shows processing chip when isProcessing is true', () => {
    renderWithTheme(
      <ProcessingLog 
        logs={[{ id: 1, message: 'Processing...', type: 'info', timestamp: '12:34:56' }]} 
        progress={50} 
        isProcessing={true} 
        onClear={mockClearLogs} 
      />
    );
    
    // Check if the processing chip is displayed
    const processingChip = screen.getByText(/Processing/i);
    expect(processingChip).toBeInTheDocument();
  });
});