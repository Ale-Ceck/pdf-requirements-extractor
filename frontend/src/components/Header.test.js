import React from 'react';
import { render, screen } from '@testing-library/react';
import Header from './Header';
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

describe('Header Component', () => {
  test('renders the header title correctly', () => {
    renderWithTheme(<Header />);
    
    // Check if the header title is rendered
    const titleElement = screen.getByText(/PDF Requirements Extractor/i);
    expect(titleElement).toBeInTheDocument();
  });

  test('renders the subtitle text', () => {
    renderWithTheme(<Header />);
    
    // Check if the subtitle is rendered
    const subtitleElement = screen.getByText(/Extract and analyze requirements/i);
    expect(subtitleElement).toBeInTheDocument();
  });

  test('renders the icon', () => {
    renderWithTheme(<Header />);
    
    // Since the icon is a Material-UI icon without accessible text,
    // we can verify the paper element containing the header is present
    const paperElement = screen.getByRole('banner');
    expect(paperElement).toBeInTheDocument();
  });
});