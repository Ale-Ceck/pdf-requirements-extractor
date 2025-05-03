le# PDF Requirements Extractor - React Frontend

This is the React-based frontend for the PDF Requirements Extractor application. It provides a modern, responsive user interface for extracting and analyzing requirements from PDF documents using AI.

## Features

- Modern, responsive UI built with React and Material-UI
- Support for both online (API-based) and offline (local) processing modes
- Configuration for various AI model providers (OpenAI, Anthropic, Together.ai, Ollama)
- File drag-and-drop functionality
- Batch processing support
- Real-time processing logs and progress tracking
- Advanced configuration options

## Setup and Installation

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Open [http://localhost:5001](http://localhost:5001) to view the application in your browser.

## Building for Production

To build the application for production:

```
npm run build
```

This creates an optimized production build in the `build` folder.

## Configuration

The frontend communicates with a backend API. By default, it connects to `http://localhost:5001/api`. You can change this by setting the `REACT_APP_API_URL` environment variable:

```
REACT_APP_API_URL=http://your-api-url npm start
```

## Folder Structure

- `src/components`: UI components
- `src/hooks`: Custom React hooks
- `src/pages`: Page components (for future routing)
- `src/services`: API service and other utilities
- `src/styles`: Styling and theme configurations