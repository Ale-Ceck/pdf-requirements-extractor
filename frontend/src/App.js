import React, { useState, useEffect } from 'react';
import { Container, Box, Typography } from '@mui/material';
import Header from './components/Header';
import OperationModeSelector from './components/OperationModeSelector';
import ConfigPanel from './components/ConfigPanel';
import FileUploadPanel from './components/FileUploadPanel';
import ProcessingLog from './components/ProcessingLog';
import ApiService from './services/api';

function App() {
  const [operationMode, setOperationMode] = useState('online');
  const [availableProviders, setAvailableProviders] = useState([]);
  const [config, setConfig] = useState({
    app: {},
    providers: {},
    extraction: {}
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState(0);

  // Load initial configuration and providers
  useEffect(() => {
    const initializeApp = async () => {
      try {
        const configData = await ApiService.getConfig();
        setConfig(configData);
        
        // Set initial operation mode from config
        if (configData.app && configData.app.use_offline_provider) {
          setOperationMode('offline');
        }
        
        const providersData = await ApiService.getProviders();
        setAvailableProviders(providersData);
      } catch (error) {
        console.error('Error initializing app:', error);
        addLog('Error loading configuration', 'error');
      }
    };
    
    initializeApp();
  }, []);

  // Update config when operation mode changes
  useEffect(() => {
    const updateConfigMode = () => {
      setConfig(prevConfig => ({
        ...prevConfig,
        app: {
          ...prevConfig.app,
          use_offline_provider: operationMode === 'offline'
        }
      }));
    };
    
    updateConfigMode();
  }, [operationMode]);

  // Add a log message
  const addLog = (message, type = 'info') => {
    setLogs(prevLogs => [
      ...prevLogs,
      {
        id: Date.now(),
        message,
        type,
        timestamp: new Date().toLocaleTimeString()
      }
    ]);
  };

  // Clear logs
  const clearLogs = () => {
    setLogs([]);
  };

  // Handle configuration changes
  const handleConfigChange = (newConfig) => {
    setConfig(prevConfig => ({
      ...prevConfig,
      ...newConfig
    }));
  };

  // Handle file processing
  const handleProcessFiles = async (data) => {
    const { files, outputPath, isBatch } = data;
    
    // Prepare form data
    const formData = new FormData();
    
    if (isBatch) {
      formData.append('directory', files[0]);
    } else {
      formData.append('file', files[0]);
    }
    
    if (outputPath) {
      formData.append('outputPath', outputPath);
    }
    
    // Add configuration to form data
    formData.append('config', JSON.stringify(config));
    
    setIsProcessing(true);
    setProgress(0);
    clearLogs();
    addLog(`Starting ${isBatch ? 'batch' : 'single file'} processing in ${operationMode.toUpperCase()} mode`);
    
    try {
      const result = isBatch
        ? await ApiService.processBatch(formData, setProgress)
        : await ApiService.processPdf(formData, setProgress);
      
      addLog('Processing completed successfully!');
      
      if (isBatch) {
        addLog(`Processed ${result.successCount} files successfully`);
        if (result.failedCount > 0) {
          addLog(`Failed to process ${result.failedCount} files`, 'warning');
        }
        addLog(`Results saved to: ${result.outputDirectory}`);
      } else {
        addLog(`Extracted ${result.requirementsCount} requirements`);
        addLog(`Output saved to: ${result.outputFile}`);
      }
    } catch (error) {
      console.error('Processing error:', error);
      addLog(`Error: ${error.message || 'Unknown error occurred'}`, 'error');
    } finally {
      setIsProcessing(false);
      setProgress(100);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, minHeight: '100vh' }}>
        <Header />
        
        <OperationModeSelector 
          operationMode={operationMode} 
          onChange={setOperationMode}
        />
        
        <ConfigPanel
          operationMode={operationMode}
          availableProviders={availableProviders}
          config={config}
          onChange={handleConfigChange}
        />
        
        <FileUploadPanel
          onProcess={handleProcessFiles}
          isProcessing={isProcessing}
        />
        
        <ProcessingLog
          logs={logs}
          progress={progress}
          isProcessing={isProcessing}
          onClear={clearLogs}
        />
      </Box>
    </Container>
  );
}

export default App;