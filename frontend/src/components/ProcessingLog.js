import React from 'react';
import { 
  Paper, 
  Box, 
  Typography, 
  List, 
  ListItem, 
  ListItemText,
  LinearProgress,
  Button,
  Chip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import ClearIcon from '@mui/icons-material/Clear';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';

// Styled log item
const LogItem = styled(ListItem)(({ theme, type }) => ({
  padding: theme.spacing(0.75, 2),
  backgroundColor: type === 'error' 
    ? theme.palette.error.light
    : type === 'warning'
      ? theme.palette.warning.light
      : type === 'success'
        ? theme.palette.secondary.light
        : 'transparent',
  borderRadius: theme.shape.borderRadius,
  margin: theme.spacing(0.5, 0),
}));

// Log icon component
const LogIcon = ({ type }) => {
  switch (type) {
    case 'error':
      return <ErrorIcon fontSize="small" color="error" />;
    case 'warning':
      return <WarningIcon fontSize="small" color="warning" />;
    case 'success':
      return <CheckCircleIcon fontSize="small" color="success" />;
    default:
      return <InfoIcon fontSize="small" color="primary" />;
  }
};

const ProcessingLog = ({ logs, progress, isProcessing, onClear }) => {
  // If no logs, show placeholder
  if (logs.length === 0 && !isProcessing) {
    return (
      <Paper
        elevation={1}
        sx={{
          p: 3,
          borderRadius: 2,
          backgroundColor: 'background.paper',
        }}
      >
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          mb: 2
        }}>
          <Typography variant="h6">
            Processing Log
          </Typography>
          
          <Button
            size="small"
            startIcon={<ClearIcon />}
            onClick={onClear}
            disabled
          >
            Clear
          </Button>
        </Box>
        
        <Box 
          sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            justifyContent: 'center', 
            p: 4,
            color: 'text.secondary',
            borderRadius: 1,
            backgroundColor: 'background.alt',
            minHeight: 200,
          }}
        >
          <InfoIcon sx={{ fontSize: 40, mb: 2, opacity: 0.5 }} />
          <Typography variant="body1">
            No processing logs to display
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Process a PDF file to see logs here
          </Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper
      elevation={1}
      sx={{
        p: 3,
        borderRadius: 2,
        backgroundColor: 'background.paper',
      }}
    >
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        mb: 2
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="h6">
            Processing Log
          </Typography>
          
          {isProcessing && (
            <Chip 
              label="Processing" 
              color="primary" 
              size="small" 
              sx={{ ml: 1 }}
            />
          )}
        </Box>
        
        <Button
          size="small"
          startIcon={<ClearIcon />}
          onClick={onClear}
          disabled={isProcessing}
        >
          Clear
        </Button>
      </Box>
      
      {isProcessing && (
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          sx={{ mb: 2, borderRadius: 1 }} 
        />
      )}
      
      <Box
        sx={{
          minHeight: 200,
          maxHeight: 300,
          overflow: 'auto',
          backgroundColor: 'background.alt',
          borderRadius: 1,
          p: 1,
          fontFamily: 'monospace',
        }}
      >
        <List dense disablePadding>
          {logs.map((log) => (
            <LogItem key={log.id} type={log.type}>
              <LogIcon type={log.type} />
              <ListItemText
                primary={
                  <Box component="span" sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    gap: 1
                  }}>
                    <Typography 
                      component="span" 
                      variant="body2" 
                      color="text.secondary"
                      sx={{ 
                        fontFamily: 'monospace',
                        minWidth: '80px' 
                      }}
                    >
                      {log.timestamp}
                    </Typography>
                    
                    <Typography 
                      component="span" 
                      variant="body2"
                      sx={{ 
                        fontFamily: 'monospace',
                        fontWeight: log.type === 'error' || log.type === 'warning' ? 'bold' : 'normal'
                      }}
                    >
                      {log.message}
                    </Typography>
                  </Box>
                }
              />
            </LogItem>
          ))}
        </List>
      </Box>
    </Paper>
  );
};

export default ProcessingLog;