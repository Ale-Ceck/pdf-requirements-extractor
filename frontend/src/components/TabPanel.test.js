import React from 'react';
import { render, screen } from '@testing-library/react';
import TabPanel from './TabPanel';
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

describe('TabPanel Component', () => {
  test('renders children when value matches index', () => {
    renderWithTheme(
      <TabPanel value={0} index={0}>
        <div>Tab Content</div>
      </TabPanel>
    );
    
    // Check if the content is displayed
    const content = screen.getByText(/Tab Content/i);
    expect(content).toBeInTheDocument();
  });

  test('does not render children when value does not match index', () => {
    renderWithTheme(
      <TabPanel value={1} index={0}>
        <div>Tab Content</div>
      </TabPanel>
    );
    
    // Check if the content is not displayed
    const content = screen.queryByText(/Tab Content/i);
    expect(content).not.toBeInTheDocument();
  });

  test('passes additional props to the root element', () => {
    renderWithTheme(
      <TabPanel value={0} index={0} data-testid="custom-tabpanel">
        <div>Tab Content</div>
      </TabPanel>
    );
    
    // Check if the custom attribute is present
    const tabPanel = screen.getByTestId('custom-tabpanel');
    expect(tabPanel).toBeInTheDocument();
  });

  test('has correct aria attributes', () => {
    renderWithTheme(
      <TabPanel value={0} index={0}>
        <div>Tab Content</div>
      </TabPanel>
    );
    
    // Check if aria attributes are correctly set
    const tabPanel = screen.getByRole('tabpanel');
    expect(tabPanel).toHaveAttribute('id', 'tabpanel-0');
    expect(tabPanel).toHaveAttribute('aria-labelledby', 'tab-0');
  });
});