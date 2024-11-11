import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import IntegrationInstructionsIcon from '@mui/icons-material/IntegrationInstructions';
import SecurityIcon from '@mui/icons-material/Security';
import TimelineIcon from '@mui/icons-material/Timeline';
import {
  Box,
  Button,
  Container,
  Divider,
  Grid,
  IconButton,
  Paper,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography
} from '@mui/material';
import axios from 'axios';
import React, { useState } from 'react';

// Helper functions
const getChangedCells = (prevMatrix, currentMatrix) => {
  if (!prevMatrix || !currentMatrix) return new Set();
  
  const changes = new Set();
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      if (prevMatrix[i][j] !== currentMatrix[i][j]) {
        changes.add(`${i}-${j}`);
      }
    }
  }
  return changes;
};

const isValidHex = (str) => /^[0-9A-Fa-f]+$/.test(str);

// Enhanced color scheme for different matrix types
const colorScheme = {
  start: { bg: '#0d1b2a', border: '#1e3a5f', label: '#4CC9F0' },
  subBytes: { bg: '#0d1b2a', border: '#1e3a5f', label: '#F72585' },
  shiftRows: { bg: '#0d1b2a', border: '#1e3a5f', label: '#7209B7' },
  mixColumns: { bg: '#0d1b2a', border: '#1e3a5f', label: '#4361EE' },
  roundKey: { bg: '#1a2b3c', border: '#2d4d6d', label: '#7B61FF' }
};

const getMatrixColors = (matrixType) => {
  switch (matrixType?.toLowerCase()) {
    case 'subbytes':
      return colorScheme.subBytes;
    case 'shiftrows':
      return colorScheme.shiftRows;
    case 'mixcolumns':
      return colorScheme.mixColumns;
    case 'roundkey':
      return colorScheme.roundKey;
    default:
      return colorScheme.start;
  }
};

const features = [
  {
    icon: <SecurityIcon sx={{ color: '#00ff9d', fontSize: 30 }} />,
    title: "Advanced Encryption",
    description: "Industry standard AES encryption ensuring your data remains secure and private at all times"
  },
  {
    icon: <IntegrationInstructionsIcon sx={{ color: '#00ff9d', fontSize: 30 }} />,
    title: "Easy Integration",
    description: "Simple API integration with your existing applications. Get started in minutes, not hours"
  },
  {
    icon: <TimelineIcon sx={{ color: '#00ff9d', fontSize: 30 }} />,
    title: "Real-time Security",
    description: "Monitor your encryption status in real-time with our advanced dashboard and analytics!"
  }
];

const AESVisualization = () => {
  // State management
  const [currentTab, setCurrentTab] = useState(0);
  const [plaintext, setPlaintext] = useState('');
  const [ciphertext, setCiphertext] = useState('');
  const [key, setKey] = useState('5468617473206D79204B756E67204675');
  const [loading, setLoading] = useState(false);
  const [showTooltips, setShowTooltips] = useState(true);
  const [encryptionData, setEncryptionData] = useState(null);
  const [decryptionData, setDecryptionData] = useState(null);
  const [error, setError] = useState(null);

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
    setError(null);
    setEncryptionData(null);
    setDecryptionData(null);
  };

  const handleEncrypt = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/api/encrypt', {
        plaintext: plaintext,
        key: key
      });

      if (response.data.status === "error") {
        setError(response.data.message);
        setEncryptionData(null);
      } else {
        setEncryptionData(response.data);
        setCiphertext(response.data.ciphertext);
      }
    } catch (error) {
      setError(error.response?.data?.message || 'An error occurred while encrypting');
      setEncryptionData(null);
    }
    setLoading(false);
  };

  const handleDecrypt = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/api/decrypt', {
        ciphertext: ciphertext,
        key: key
      });

      if (response.data.status === "error") {
        setError(response.data.message);
        setDecryptionData(null);
      } else {
        setDecryptionData(response.data);
      }
    } catch (error) {
      setError(error.response?.data?.message || 'An error occurred while decrypting');
      setDecryptionData(null);
    }
    setLoading(false);
  };

  const renderMatrix = (matrix, label, tooltip, prevMatrix = null) => {
    if (!matrix?.data) return null;
    
    const colors = getMatrixColors(matrix.type);
    const changedCells = prevMatrix ? getChangedCells(prevMatrix, matrix.data) : new Set();

    return (
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Typography variant="subtitle1" sx={{ mr: 1, color: colors.label }}>
            {label}
          </Typography>
          {tooltip && (
            <Tooltip title={tooltip} arrow>
              <IconButton size="small" sx={{ color: '#8892b0' }}>
                <HelpOutlineIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
        <Grid container spacing={1} sx={{ maxWidth: 300 }}>
          {matrix.data.map((row, i) => (
            <Grid item xs={12} key={i}>
              <Grid container spacing={1}>
                {row.map((cell, j) => (
                  <Grid item xs={3} key={j}>
                    <Paper
                      sx={{
                        p: 1,
                        textAlign: 'center',
                        fontFamily: 'monospace',
                        fontSize: '0.875rem',
                        bgcolor: colors.bg,
                        border: '2px solid',
                        borderColor: changedCells.has(`${i}-${j}`) ? colors.label : colors.border,
                        transition: 'all 0.3s ease',
                      }}
                    >
                      {cell}
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  const renderForm = () => {
    const isEncryption = currentTab === 0;
    return (
      <Paper sx={{ p: 4, mb: 4 }}>
        <Typography variant="h5" sx={{ mb: 3, color: '#00ff9d' }}>
          {isEncryption ? 'Encryption' : 'Decryption'}
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label={isEncryption ? "Plaintext" : "Ciphertext (hex)"}
              variant="outlined"
              value={isEncryption ? plaintext : ciphertext}
              onChange={(e) => isEncryption 
                ? setPlaintext(e.target.value.toUpperCase())
                : setCiphertext(e.target.value.toUpperCase())
              }
              sx={{ mb: 2 }}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Key (hex)"
              variant="outlined"
              value={key}
              onChange={(e) => setKey(e.target.value.toUpperCase())}
              sx={{ mb: 2 }}
            />
          </Grid>
          <Grid item xs={12}>
            <Button
              variant="contained"
              onClick={isEncryption ? handleEncrypt : handleDecrypt}
              disabled={loading}
              fullWidth
              sx={{ 
                height: '50px',
                background: 'linear-gradient(45deg, #00ff9d 30%, #00f5d4 90%)',
                color: '#000000',
                fontWeight: 'bold',
                '&:hover': {
                  background: 'linear-gradient(45deg, #00f5d4 30%, #00ff9d 90%)',
                }
              }}
            >
              {loading ? (isEncryption ? 'Encrypting...' : 'Decrypting...') : (isEncryption ? 'Encrypt' : 'Decrypt')}
            </Button>
          </Grid>
        </Grid>
      </Paper>
    );
  };

  const renderResult = () => {
    const result = currentTab === 0 ? encryptionData?.ciphertext : decryptionData?.plaintext;
    if (!result) return null;

    return (
      <Paper sx={{ 
        p: 3, 
        mb: 4,
        background: 'linear-gradient(to right, #0d1b2a, #1e3a5f)',
        border: '1px solid #00ff9d'
      }}>
        <Typography variant="h6" gutterBottom sx={{ color: '#00ff9d' }}>
          {currentTab === 0 ? 'Encrypted Result' : 'Decrypted Result'}
        </Typography>
        <Divider sx={{ my: 2, borderColor: 'rgba(0, 255, 157, 0.1)' }} />
        <Typography 
          variant="body1" 
          fontFamily="monospace" 
          sx={{ 
            color: '#fff',
            wordBreak: 'break-all',
            p: 2,
            bgcolor: 'rgba(0, 255, 157, 0.05)',
            borderRadius: 1
          }}
        >
          {result}
        </Typography>
      </Paper>
    );
  };

  return (
    <Container maxWidth="xl" sx={{ pb: 8 }}>
      {/* Hero Section */}
      <Box sx={{ textAlign: 'center', mb: 8 }}>
        <Typography variant="h3" sx={{ mb: 2, fontWeight: 'bold' }}>
          Secure Your Data with AESential
        </Typography>
        <Typography variant="h6" sx={{ mb: 4, color: 'text.secondary' }}>
          Enterprise-grade encryption made simple. Protect your sensitive information with military-grade AES encryption.
        </Typography>
      </Box>

      {/* Features */}
      <Grid container spacing={4} sx={{ mb: 8 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={4} key={index}>
            <Paper sx={{ 
              p: 3, 
              height: '100%',
              background: 'linear-gradient(to bottom right, #0d1b2a, #1e3a5f)',
              border: '1px solid',
              borderColor: 'background.paper',
              '&:hover': {
                borderColor: '#00ff9d',
              },
              transition: 'all 0.3s ease'
            }}>
              <Box sx={{ mb: 2 }}>{feature.icon}</Box>
              <Typography variant="h6" sx={{ mb: 1, color: '#00ff9d' }}>
                {feature.title}
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                {feature.description}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      {/* Tabs for Encryption/Decryption */}
      <Paper sx={{ mb: 4, backgroundColor: 'background.paper' }}>
        <Tabs
          value={currentTab}
          onChange={handleTabChange}
          centered
          sx={{
            '& .MuiTab-root': {
              color: '#8892b0',
              '&.Mui-selected': {
                color: '#00ff9d',
              },
            },
            '& .MuiTabs-indicator': {
              backgroundColor: '#00ff9d',
            },
          }}
        >
          <Tab label="Encryption" />
          <Tab label="Decryption" />
        </Tabs>
      </Paper>

      {/* Form */}
      {renderForm()}

      {error && (
        <Paper sx={{ 
          p: 2, 
          mb: 3, 
          bgcolor: 'rgba(255, 0, 0, 0.1)',
          border: '1px solid rgba(255, 0, 0, 0.3)'
        }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}

      {/* Result Section */}
      {renderResult()}

      {/* Visualization */}
      {(encryptionData || decryptionData) && (
        <Paper sx={{ 
          p: 3,
          background: 'linear-gradient(to bottom, #0d1b2a, #1e3a5f)',
        }}>
          <Typography variant="h6" sx={{ mb: 3, color: '#00ff9d' }}>
            Step-by-Step Visualization
          </Typography>
          <Box sx={{ 
            maxHeight: 'calc(100vh - 250px)', 
            overflowY: 'auto',
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              backgroundColor: '#0d1b2a',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: '#1e3a5f',
              borderRadius: '4px',
            }
          }}>
            {(currentTab === 0 ? encryptionData : decryptionData)?.visualization?.rounds.map((round, roundIndex) => (
              <Paper key={roundIndex} sx={{ mb: 3, p: 3, bgcolor: 'background.paper' }}>
                <Typography variant="h6" gutterBottom sx={{ color: '#00ff9d' }}>
                  {round.title}
                </Typography>
                <Grid container spacing={3} alignItems="flex-start">
                  {round.matrices.map((matrix, matrixIndex) => (
                    <Grid item xs={12} sm={6} md={2.4} key={matrixIndex}>
                      {renderMatrix(
                        matrix,
                        matrix.label,
                        matrix.tooltip,
                        matrixIndex > 0 ? round.matrices[matrixIndex - 1].data : null
                      )}
                    </Grid>
                  ))}
                </Grid>
              </Paper>
            ))}
          </Box>
        </Paper>
      )}
    </Container>
  );
};

export default AESVisualization;