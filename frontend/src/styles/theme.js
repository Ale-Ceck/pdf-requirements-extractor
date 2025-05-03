import { createTheme } from '@mui/material/styles';

// Create a theme with the color palette from the existing PyQt6 app
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976D2',
      light: '#BBDEFB',
    },
    secondary: {
      main: '#4CAF50',
    },
    warning: {
      main: '#FF9800',
    },
    error: {
      main: '#f44336',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
    },
    background: {
      default: '#FFFFFF',
      paper: '#FAFAFA',
      alt: '#F5F5F5',
    },
    border: {
      main: '#DDDDDD',
    },
    highlight: {
      main: '#E1F5FE',
    },
  },
  typography: {
    fontFamily: 'Arial, sans-serif',
    h1: {
      fontSize: '1.8rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.3rem',
      fontWeight: 600,
    },
    body1: {
      fontSize: '1rem',
    },
    body2: {
      fontSize: '0.9rem',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 6,
          textTransform: 'none',
          fontWeight: 'bold',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 6,
          },
        },
      },
    },
  },
});

export default theme;