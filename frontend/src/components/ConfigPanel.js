import React, { useState, useEffect } from 'react';
import { 
  Paper, 
  Box, 
  Typography, 
  Tabs, 
  Tab, 
  TextField, 
  MenuItem, 
  Button, 
  FormControl, 
  InputLabel, 
  Select,
  Grid,
  CircularProgress,
  Alert,
  Divider
} from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import ApiService from '../services/api';
import AdvancedSettingsDialog from './AdvancedSettingsDialog';

const ConfigPanel = ({ operationMode, availableProviders, config, onChange }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [onlineConfig, setOnlineConfig] = useState({
    provider: '',
    model: '',
    apiKey: '',
    verificationStrategy: 'different',
    verificationProvider: '',
    verificationModel: ''
  });
  const [offlineConfig, setOfflineConfig] = useState({
    serverUrl: 'http://localhost:11434',
    model: ''
  });
  const [ollamaStatus, setOllamaStatus] = useState('unchecked');
  const [ollamaModels, setOllamaModels] = useState([]);
  const [isCheckingOllama, setIsCheckingOllama] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  
  // Filter providers by type
  const onlineProviders = availableProviders.filter(p => 
    p.id === 'openai' || p.id === 'anthropic' || p.id === 'together'
  );
  const offlineProviders = availableProviders.filter(p => 
    p.id === 'ollama'
  );
  
  // Set the active tab based on operation mode
  useEffect(() => {
    setActiveTab(operationMode === 'online' ? 0 : 1);
  }, [operationMode]);
  
  // Initialize configuration when config or providers change
  useEffect(() => {
    if (config && config.extraction && availableProviders.length > 0) {
      // Set online configuration
      setOnlineConfig({
        provider: config.extraction.provider || 'openai',
        model: config.extraction.model || '',
        apiKey: config.providers?.[config.extraction.provider]?.api_key || '',
        verificationStrategy: config.extraction.verification_strategy || 'different',
        verificationProvider: config.extraction.verification_provider || '',
        verificationModel: config.extraction.verification_model || ''
      });
      
      // Set offline configuration
      setOfflineConfig({
        serverUrl: config.providers?.ollama?.server_url || 'http://localhost:11434',
        model: config.extraction.model || ''
      });
      
      // If in offline mode, check Ollama status
      if (operationMode === 'offline') {
        checkOllamaStatus();
      }
    }
  }, [config, availableProviders, operationMode]);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Check Ollama status
  const checkOllamaStatus = async () => {
    setIsCheckingOllama(true);
    setOllamaStatus('checking');
    
    try {
      const result = await ApiService.checkOllamaStatus(offlineConfig.serverUrl);
      
      if (result.isRunning) {
        setOllamaStatus('running');
        setOllamaModels(result.models || []);
      } else {
        setOllamaStatus('not-running');
        setOllamaModels([]);
      }
    } catch (error) {
      console.error('Error checking Ollama status:', error);
      setOllamaStatus('error');
      setOllamaModels([]);
    } finally {
      setIsCheckingOllama(false);
    }
  };
  
  // Find available models for a provider
  const getModelsForProvider = (providerId) => {
    const provider = availableProviders.find(p => p.id === providerId);
    return provider ? provider.models : [];
  };
  
  // Handle online configuration changes
  const handleOnlineConfigChange = (field, value) => {
    setOnlineConfig(prev => ({ ...prev, [field]: value }));
    
    // Update model list if provider changes
    if (field === 'provider') {
      const defaultModel = getModelsForProvider(value)[0] || '';
      setOnlineConfig(prev => ({ ...prev, model: defaultModel }));
    }
    
    // Update verification model list if verification provider changes
    if (field === 'verificationProvider') {
      const defaultModel = getModelsForProvider(value)[0] || '';
      setOnlineConfig(prev => ({ ...prev, verificationModel: defaultModel }));
    }
  };
  
  // Handle offline configuration changes
  const handleOfflineConfigChange = (field, value) => {
    setOfflineConfig(prev => ({ ...prev, [field]: value }));
  };
  
  // Save configuration
  const saveConfig = () => {
    if (operationMode === 'online') {
      // Prepare online configuration
      const newConfig = {
        extraction: {
          provider: onlineConfig.provider,
          model: onlineConfig.model,
          verification_strategy: onlineConfig.verificationStrategy
        },
        providers: {
          [onlineConfig.provider]: {
            api_key: onlineConfig.apiKey,
            enabled: true
          }
        }
      };
      
      // Add verification provider and model if strategy is 'specific'
      if (onlineConfig.verificationStrategy === 'specific') {
        newConfig.extraction.verification_provider = onlineConfig.verificationProvider;
        newConfig.extraction.verification_model = onlineConfig.verificationModel;
      }
      
      onChange(newConfig);
    } else {
      // Prepare offline configuration
      const newConfig = {
        extraction: {
          provider: 'ollama',
          model: offlineConfig.model,
          verification_strategy: 'same'
        },
        providers: {
          ollama: {
            server_url: offlineConfig.serverUrl,
            enabled: true
          }
        }
      };
      
      onChange(newConfig);
    }
  };
  
  // Render online configuration panel
  const renderOnlineConfig = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Alert 
          severity="warning"
          sx={{ borderRadius: 1.5, mb: 2 }}
        >
          In Online mode, your content will be sent to external API servers via the internet
        </Alert>
      </Grid>
      
      <Grid item xs={12} sm={6}>
        <FormControl fullWidth>
          <InputLabel id="online-provider-label">Online Provider</InputLabel>
          <Select
            labelId="online-provider-label"
            id="online-provider"
            value={onlineConfig.provider}
            label="Online Provider"
            onChange={(e) => handleOnlineConfigChange('provider', e.target.value)}
          >
            {onlineProviders.map(provider => (
              <MenuItem key={provider.id} value={provider.id}>
                {provider.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12} sm={6}>
        <FormControl fullWidth>
          <InputLabel id="online-model-label">Extraction Model</InputLabel>
          <Select
            labelId="online-model-label"
            id="online-model"
            value={onlineConfig.model}
            label="Extraction Model"
            onChange={(e) => handleOnlineConfigChange('model', e.target.value)}
          >
            {getModelsForProvider(onlineConfig.provider).map(model => (
              <MenuItem key={model} value={model}>
                {model}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12}>
        <TextField
          fullWidth
          id="api-key"
          label="API Key"
          type="password"
          value={onlineConfig.apiKey}
          onChange={(e) => handleOnlineConfigChange('apiKey', e.target.value)}
          placeholder="Enter your API key or it will be loaded from environment"
        />
      </Grid>
      
      <Grid item xs={12}>
        <Divider sx={{ my: 1 }} />
        <Typography variant="h6" gutterBottom>
          Verification Settings
        </Typography>
      </Grid>
      
      <Grid item xs={12} sm={6}>
        <FormControl fullWidth>
          <InputLabel id="verification-strategy-label">Verification Strategy</InputLabel>
          <Select
            labelId="verification-strategy-label"
            id="verification-strategy"
            value={onlineConfig.verificationStrategy}
            label="Verification Strategy"
            onChange={(e) => handleOnlineConfigChange('verificationStrategy', e.target.value)}
          >
            <MenuItem value="same">Same Provider</MenuItem>
            <MenuItem value="different">Different Provider</MenuItem>
            <MenuItem value="specific">Specific Provider</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      
      {onlineConfig.verificationStrategy === 'specific' && (
        <>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel id="verification-provider-label">Verification Provider</InputLabel>
              <Select
                labelId="verification-provider-label"
                id="verification-provider"
                value={onlineConfig.verificationProvider}
                label="Verification Provider"
                onChange={(e) => handleOnlineConfigChange('verificationProvider', e.target.value)}
              >
                {onlineProviders.map(provider => (
                  <MenuItem key={provider.id} value={provider.id}>
                    {provider.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel id="verification-model-label">Verification Model</InputLabel>
              <Select
                labelId="verification-model-label"
                id="verification-model"
                value={onlineConfig.verificationModel}
                label="Verification Model"
                onChange={(e) => handleOnlineConfigChange('verificationModel', e.target.value)}
                disabled={!onlineConfig.verificationProvider}
              >
                {getModelsForProvider(onlineConfig.verificationProvider).map(model => (
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </>
      )}
    </Grid>
  );
  
  // Render offline configuration panel
  const renderOfflineConfig = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Alert 
          severity="success"
          sx={{ borderRadius: 1.5, mb: 2 }}
        >
          Offline mode: All processing happens locally. No data is sent over the internet.
        </Alert>
      </Grid>
      
      <Grid item xs={12}>
        <Paper
          variant="outlined"
          sx={{ p: 2, borderRadius: 2 }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Ollama Service
            </Typography>
            
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Typography 
                variant="body1" 
                sx={{ 
                  mr: 2,
                  color: ollamaStatus === 'running' 
                    ? 'success.main' 
                    : ollamaStatus === 'checking' 
                      ? 'text.secondary' 
                      : 'error.main'
                }}
              >
                {ollamaStatus === 'unchecked' && 'Not checked'}
                {ollamaStatus === 'checking' && 'Checking...'}
                {ollamaStatus === 'running' && `Running - ${ollamaModels.length} models available`}
                {ollamaStatus === 'not-running' && 'Not running'}
                {ollamaStatus === 'error' && 'Error connecting'}
              </Typography>
              
              <Button
                variant="outlined"
                size="small"
                onClick={checkOllamaStatus}
                disabled={isCheckingOllama}
                startIcon={isCheckingOllama ? <CircularProgress size={20} /> : <RefreshIcon />}
              >
                Check Status
              </Button>
            </Box>
          </Box>
          
          <TextField
            fullWidth
            id="server-url"
            label="Ollama Server URL"
            value={offlineConfig.serverUrl}
            onChange={(e) => handleOfflineConfigChange('serverUrl', e.target.value)}
            margin="normal"
          />
        </Paper>
      </Grid>
      
      <Grid item xs={12}>
        <Paper
          variant="outlined"
          sx={{ p: 2, borderRadius: 2 }}
        >
          <Typography variant="h6" gutterBottom>
            Model Selection
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <FormControl fullWidth>
              <InputLabel id="ollama-model-label">Available Models</InputLabel>
              <Select
                labelId="ollama-model-label"
                id="ollama-model"
                value={offlineConfig.model}
                label="Available Models"
                onChange={(e) => handleOfflineConfigChange('model', e.target.value)}
                disabled={ollamaModels.length === 0}
              >
                {ollamaModels.map(model => (
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Button
              variant="outlined"
              size="small"
              onClick={checkOllamaStatus}
              disabled={isCheckingOllama}
            >
              Refresh
            </Button>
          </Box>
          
          {ollamaStatus === 'not-running' && (
            <Alert 
              severity="error"
              sx={{ mt: 2, borderRadius: 1.5 }}
            >
              Ollama doesn't seem to be running. Please start Ollama and try again.
              If you don't have Ollama installed, visit: https://ollama.com/download
            </Alert>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
  
  return (
    <Paper 
      elevation={1}
      sx={{ 
        borderRadius: 2,
        overflow: 'hidden'
      }}
    >
      <Tabs
        value={activeTab}
        onChange={handleTabChange}
        aria-label="configuration tabs"
        sx={{
          backgroundColor: 'background.alt',
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Tab 
          label="Online Configuration" 
          disabled={operationMode !== 'online'} 
        />
        <Tab 
          label="Offline Configuration" 
          disabled={operationMode !== 'offline'} 
        />
      </Tabs>
      
      <Box sx={{ p: 3 }}>
        {activeTab === 0 && renderOnlineConfig()}
        {activeTab === 1 && renderOfflineConfig()}
        
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          mt: 3
        }}>
          <Button
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setShowAdvancedSettings(true)}
          >
            Advanced Settings
          </Button>
          
          <Button
            variant="contained"
            color="primary"
            startIcon={<SaveIcon />}
            onClick={saveConfig}
          >
            Save Settings
          </Button>
        </Box>
      </Box>
      
      <AdvancedSettingsDialog 
        open={showAdvancedSettings} 
        onClose={() => setShowAdvancedSettings(false)}
        config={config.app || {}}
        onSave={(newAdvancedConfig) => {
          onChange({ app: newAdvancedConfig });
          setShowAdvancedSettings(false);
        }}
      />
    </Paper>
  );
};

export default ConfigPanel;