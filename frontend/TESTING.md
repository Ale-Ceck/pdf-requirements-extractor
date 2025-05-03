# Testing Guide for PDF Requirements Extractor Frontend

This document outlines the testing strategy and provides instructions for running and writing tests for the React frontend of the PDF Requirements Extractor application.

## Testing Setup

The frontend uses the following testing tools:

- **Jest**: JavaScript testing framework
- **React Testing Library**: Testing utilities for React components
- **jest-fetch-mock**: Mock implementation for fetch
- **msw (Mock Service Worker)**: API mocking library for browser and Node

## Running Tests

To run the tests, use the following commands:

```bash
# Run tests in watch mode (default)
npm test

# Run tests with coverage report
npm run test:coverage
```

## Test Coverage

The project is configured to maintain at least 50% code coverage across:
- Statements
- Branches
- Functions
- Lines

You can view the coverage report in the console after running the coverage command, or open the detailed HTML report at `coverage/lcov-report/index.html`.

## Testing Structure

### 1. Component Tests

Component tests verify that each UI component renders correctly and behaves as expected. Tests are located next to the component files with a `.test.js` suffix.

Example:
```javascript
// Header.test.js
import { render, screen } from '@testing-library/react';
import Header from './Header';

test('renders the header title', () => {
  render(<Header />);
  const titleElement = screen.getByText(/PDF Requirements Extractor/i);
  expect(titleElement).toBeInTheDocument();
});
```

### 2. Service Tests

Service tests verify that API calls are made correctly. These tests mock external dependencies like `axios`.

Example:
```javascript
// api.test.js
import axios from 'axios';
import apiService from './api';

jest.mock('axios');

test('getProviders calls the correct endpoint', async () => {
  axios.get.mockResolvedValue({ data: [] });
  await apiService.getProviders();
  expect(axios.get).toHaveBeenCalledWith('/providers');
});
```

### 3. Integration Tests

Integration tests verify that components work together correctly. These tests use the App component to test multiple components working together.

## Testing Utilities

The project includes testing utilities at `src/test-utils.js`:

- `renderWithProviders`: Custom render function that includes the theme provider
- Mock data factories for providers, configuration, and logs

## Writing Tests

When writing tests, follow these guidelines:

1. **Test Behavior, Not Implementation**: Focus on what the component does, not how it does it
2. **Use Meaningful Assertions**: Write assertions that verify important behavior
3. **Mock External Dependencies**: Use Jest's mocking capabilities for external services
4. **Test User Interactions**: Use `userEvent` to simulate user interactions
5. **Test Accessibility**: Ensure components are accessible by using proper ARIA roles and attributes

### Example Test Structure

```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import { renderWithProviders } from '../test-utils';
import MyComponent from './MyComponent';

describe('MyComponent', () => {
  test('renders correctly', () => {
    renderWithProviders(<MyComponent />);
    // Assertions here
  });

  test('handles user interaction', () => {
    renderWithProviders(<MyComponent />);
    const button = screen.getByRole('button');
    fireEvent.click(button);
    // Assertions here
  });
});
```

## Debugging Tests

If a test fails, you can use the following techniques to debug:

1. Use `screen.debug()` to print the current state of the DOM
2. Use `console.log()` to inspect values
3. Use the `--verbose` flag with Jest to get more detailed output

Example:
```javascript
test('debugging example', () => {
  renderWithProviders(<MyComponent />);
  screen.debug(); // Prints the DOM structure
});
```

## Continuous Integration

Tests are automatically run as part of the CI/CD pipeline. Pull requests will fail if tests don't pass or code coverage drops below the threshold.