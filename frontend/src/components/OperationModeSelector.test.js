import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import OperationModeSelector from './OperationModeSelector';
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

describe('OperationModeSelector Component', () => {
  const mockOnChange = jest.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  test('renders operation mode options correctly', () => {
    renderWithTheme(
      <OperationModeSelector 
        operationMode="online" 
        onChange={mockOnChange} 
      />
    );
    
    // Check if the mode buttons are rendered
    const onlineButton = screen.getByText(/Online \(API\)/i);
    const offlineButton = screen.getByText(/Offline \(Local\)/i);
    
    expect(onlineButton).toBeInTheDocument();
    expect(offlineButton).toBeInTheDocument();
  });

  test('displays online mode warning when online mode is selected', () => {
    renderWithTheme(
      <OperationModeSelector 
        operationMode="online" 
        onChange={mockOnChange} 
      />
    );
    
    // Check if the warning message is displayed
    const warningMessage = screen.getByText(/Warning: Online mode sends data to external API services/i);
    expect(warningMessage).toBeInTheDocument();
  });

  test('displays offline mode message when offline mode is selected', () => {
    renderWithTheme(
      <OperationModeSelector 
        operationMode="offline" 
        onChange={mockOnChange} 
      />
    );
    
    // Check if the offline message is displayed
    const offlineMessage = screen.getByText(/Offline mode: All processing happens locally/i);
    expect(offlineMessage).toBeInTheDocument();
  });

  test('calls onChange when switching from offline to online without confirmation', () => {
    renderWithTheme(
      <OperationModeSelector 
        operationMode="offline" 
        onChange={mockOnChange} 
      />
    );
    
    // Click the online mode button
    const onlineButton = screen.getByText(/Online \(API\)/i);
    fireEvent.click(onlineButton);
    
    // Check if the confirmation dialog is shown
    const dialogTitle = screen.getByText(/Switch to Online Mode/i);
    expect(dialogTitle).toBeInTheDocument();
    
    // The onChange should not be called yet
    expect(mockOnChange).not.toHaveBeenCalled();
  });

  test('calls onChange when switching from offline to online and confirming', async () => {
    renderWithTheme(
      <OperationModeSelector 
        operationMode="offline" 
        onChange={mockOnChange} 
      />
    );
    
    // Click the online mode button
    const onlineButton = screen.getByText(/Online \(API\)/i);
    fireEvent.click(onlineButton);
    
    // Check if the confirmation dialog is shown
    const dialogTitle = screen.getByText(/Switch to Online Mode/i);
    expect(dialogTitle).toBeInTheDocument();
    
    // Click the continue button
    const continueButton = screen.getByText(/Continue/i);
    fireEvent.click(continueButton);
    
    // Check if onChange was called with the correct mode
    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith('online');
    });
  });

  test('does not call onChange when switching from offline to online and canceling', () => {
    renderWithTheme(
      <OperationModeSelector 
        operationMode="offline" 
        onChange={mockOnChange} 
      />
    );
    
    // Click the online mode button
    const onlineButton = screen.getByText(/Online \(API\)/i);
    fireEvent.click(onlineButton);
    
    // Check if the confirmation dialog is shown
    const dialogTitle = screen.getByText(/Switch to Online Mode/i);
    expect(dialogTitle).toBeInTheDocument();
    
    // Click the cancel button
    const cancelButton = screen.getByText(/Cancel/i);
    fireEvent.click(cancelButton);
    
    // Check that onChange was not called
    expect(mockOnChange).not.toHaveBeenCalled();
  });

  test('calls onChange immediately when switching from online to offline', () => {
    renderWithTheme(
      <OperationModeSelector 
        operationMode="online" 
        onChange={mockOnChange} 
      />
    );
    
    // Click the offline mode button
    const offlineButton = screen.getByText(/Offline \(Local\)/i);
    fireEvent.click(offlineButton);
    
    // Check if onChange was called with the correct mode
    expect(mockOnChange).toHaveBeenCalledWith('offline');
  });
});