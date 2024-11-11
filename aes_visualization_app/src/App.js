import { createTheme, CssBaseline, ThemeProvider } from '@mui/material';
import React from 'react';
import './App.css';
import AESVisualization from './components/AESVisualization';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff9d',
    },
    background: {
      default: '#0a1628',
      paper: 'rgba(16, 42, 66, 0.7)',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
        contained: {
          backgroundColor: 'rgba(0, 255, 157, 0.1)',
          color: '#00ff9d',
          '&:hover': {
            backgroundColor: 'rgba(0, 255, 157, 0.2)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'rgba(16, 42, 66, 0.7)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="app-container">
        {/* Background layers */}
        <div className="background-layers">
          <div className="grid-overlay" />
          <div className="background-glow glow-1" />
          <div className="background-glow glow-2" />
        </div>

        {/* Content */}
        <div className="content-container">
          <header className="app-header">
            <div className="logo">
              <span className="logo-text">AESential</span>
              <div className="logo-glow" />
            </div>
            <a href="/docs" className="documentation-link">Documentation</a>
          </header>

          <AESVisualization />
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;