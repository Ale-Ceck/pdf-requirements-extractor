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
  FormControlLabel,
  InputLabel, 
  Select,
  Switch,
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
    verificationEnabled: true,
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
        verificationEnabled: config.extraction.verification_enabled !== false, // Default to true if not specified
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
    if (field === 'provider') {
      const newProvider = value;
      const defaultModel = getModelsForProvider(newProvider)[0] || '';
      const newApiKey = config.providers?.[newProvider]?.api_key || ''; // Reset API key
      setOnlineConfig(prev => ({
        ...prev,
        provider: newProvider,
        model: defaultModel,
        apiKey: newApiKey
      }));
      // Also reset verification provider if it was the same as the old provider
      if (onlineConfig.verificationStrategy !== 'specific' || onlineConfig.verificationProvider === onlineConfig.provider) {
           setOnlineConfig(prev => ({ ...prev, verificationProvider: '', verificationModel: '' }));
      }
  
    } else {
      setOnlineConfig(prev => ({ ...prev, [field]: value }));
    }
  
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
  const saveConfig = async () => {
    let updatedPart = {};
    if (operationMode === 'online') {
      // Prepare online configuration
       updatedPart = {
        extraction: {
          provider: onlineConfig.provider,
          model: onlineConfig.model,
          verification_enabled: onlineConfig.verificationEnabled,
          verification_strategy: onlineConfig.verificationStrategy
        },
        providers: {
          ...(config.providers || {}), //Preserve existing providers
          [onlineConfig.provider]: {
            ...((config.providers && config.providers[onlineConfig.provider]) || {}), // Preserve existing provider settings
            api_key: onlineConfig.apiKey,
            enabled: true
          }
        }
      };
      
      // Explicitly log the verification state to verify it's being set correctly
      console.log("Setting verification_enabled to:", onlineConfig.verificationEnabled);

      // Clear out the other provider if it exists and is different from the current one to avoid stale data
      const otherProviderKey = onlineProviders.map(p => p.id).find(id => id !== onlineConfig.provider);
      if (otherProviderKey && updatedPart.providers[otherProviderKey]) {
        delete updatedPart.providers[otherProviderKey];
      }

      // Add verification provider and model if strategy is 'specific'
      if (onlineConfig.verificationStrategy === 'specific' && onlineConfig.verificationEnabled) {
        updatedPart.extraction.verification_provider = onlineConfig.verificationProvider;
        updatedPart.extraction.verification_model = onlineConfig.verificationModel;
      } else {
        delete updatedPart.extraction.verification_provider;
        delete updatedPart.extraction.verification_model;
      }
    
    } else {
      // Prepare offline configuration
      updatedPart = {
        extraction: {
          provider: 'ollama',
          model: offlineConfig.model,
          verification_enabled: false,  // Always disable verification in offline mode
          verification_strategy: 'same'
        },
        providers: {
          ...(config.providers || {}), // Preserve existing providers
          ollama: {
            ...((config.providers && config.providers.ollama) || {}), // Preserve existing provider settings
            server_url: offlineConfig.serverUrl,
            enabled: true
          }
        }
      };
     // Clear out online provider settings if switching to offline
     onlineProviders.forEach(p => {
      if (updatedPart.providers[p.id]) {
          delete updatedPart.providers[p.id]; 
      }
  });
}
      
      //Merge with existing config
      const newFullConfig = {
        ...config,
        extraction: {
          ...(config.extraction || {}),
          ...updatedPart.extraction
        },
        providers: {
          ...(config.providers || {}),
          ...updatedPart.providers
        }
    };
    
    try {
      // First update the local state through the onChange callback
      onChange(newFullConfig);
      
      // Then actually save the configuration to the backend
      const result = await ApiService.saveConfig(newFullConfig);
      console.log("Configuration saved to server:", result);
      
      // Show a temporary success message
      alert("Configuration saved successfully!");
    } catch (error) {
      console.error("Error saving configuration:", error);
      alert("Failed to save configuration to server. Please try again.");
    }
  }
  
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
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            Verification Settings
          </Typography>
          <FormControl component="fieldset">
            <FormControlLabel
              control={
                <Switch
                  checked={onlineConfig.verificationEnabled}
                  onChange={(e) => handleOnlineConfigChange('verificationEnabled', e.target.checked)}
                  color="primary"
                />
              }
              label="Enable Verification"
            />
          </FormControl>
        </Box>
      </Grid>
      
      {onlineConfig.verificationEnabled && (
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
      )}
      
      {onlineConfig.verificationEnabled && onlineConfig.verificationStrategy === 'specific' && (
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
        <Alert 
          severity="info"
          sx={{ borderRadius: 1.5, mb: 2 }}
        >
          Verification process is automatically disabled in offline mode to optimize performance.
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
          const newFullConfig = {
            ...config, //Spread existing config
            app: newAdvancedConfig // Update app settings
          };
          onChange(newFullConfig);
          setShowAdvancedSettings(false);
        }}
      />
    </Paper>
  );
};

export default ConfigPanel;