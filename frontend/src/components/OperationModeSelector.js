import React, { useState } from 'react';
import { 
  Paper, 
  Box, 
  Typography, 
  ToggleButtonGroup, 
  ToggleButton, 
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button
} from '@mui/material';
import CloudIcon from '@mui/icons-material/Cloud';
import ComputerIcon from '@mui/icons-material/Computer';

const OperationModeSelector = ({ operationMode, onChange }) => {
  const [showWarning, setShowWarning] = useState(false);
  
  const handleModeChange = (event, newMode) => {
    // Only proceed if a new mode is selected (prevent deselection)
    if (!newMode) return;
    
    // If switching to online mode, show warning dialog
    if (newMode === 'online' && operationMode === 'offline') {
      setShowWarning(true);
    } else {
      // Otherwise, update directly
      onChange(newMode);
    }
  };
  
  const handleConfirmSwitch = () => {
    onChange('online');
    setShowWarning(false);
  };
  
  const handleCancelSwitch = () => {
    setShowWarning(false);
  };
  
  return (
    <>
      <Paper 
        elevation={1}
        sx={{ 
          padding: 2, 
          borderRadius: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
          <Typography 
            variant="h3" 
            sx={{ 
              flexShrink: 0, 
              minWidth: '150px',
              fontWeight: 'bold' 
            }}
          >
            Operation Mode:
          </Typography>
          
          <ToggleButtonGroup
            value={operationMode}
            exclusive
            onChange={handleModeChange}
            aria-label="operation mode"
            sx={{ flexGrow: 1 }}
          >
            <ToggleButton 
              value="online" 
              aria-label="online mode"
              sx={{
                py: 1.5,
                px: 3,
                display: 'flex',
                gap: 1,
                borderRadius: '6px !important',
                fontWeight: 'bold',
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  }
                }
              }}
            >
              <CloudIcon />
              <Typography>Online (API)</Typography>
            </ToggleButton>
            
            <ToggleButton 
              value="offline" 
              aria-label="offline mode"
              sx={{
                py: 1.5,
                px: 3,
                display: 'flex',
                gap: 1,
                borderRadius: '6px !important',
                fontWeight: 'bold',
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  }
                }
              }}
            >
              <ComputerIcon />
              <Typography>Offline (Local)</Typography>
            </ToggleButton>
          </ToggleButtonGroup>
          
          {operationMode === 'online' ? (
            <Alert 
              severity="warning"
              sx={{ 
                flexBasis: '100%',
                mt: 1,
                borderRadius: 1.5
              }}
            >
              Warning: Online mode sends data to external API services
            </Alert>
          ) : (
            <Alert 
              severity="success"
              sx={{ 
                flexBasis: '100%',
                mt: 1,
                borderRadius: 1.5
              }}
            >
              Offline mode: All processing happens locally. No data is sent over the internet.
            </Alert>
          )}
        </Box>
      </Paper>
      
      {/* Confirmation Dialog */}
      <Dialog
        open={showWarning}
        onClose={handleCancelSwitch}
        aria-labelledby="online-mode-dialog-title"
        aria-describedby="online-mode-dialog-description"
      >
        <DialogTitle id="online-mode-dialog-title">
          Switch to Online Mode
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="online-mode-dialog-description">
            Switching to Online mode will send your data to external API services. Are you sure you want to continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelSwitch} color="primary">
            Cancel
          </Button>
          <Button onClick={handleConfirmSwitch} color="primary" variant="contained">
            Continue
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default OperationModeSelector;