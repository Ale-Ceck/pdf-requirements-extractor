import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  FormGroup,
  FormControlLabel,
  Checkbox,
  TextField,
  Slider,
  Typography,
  Grid,
  InputAdornment
} from '@mui/material';
import TabPanel from './TabPanel';

const AdvancedSettingsDialog = ({ open, onClose, config, onSave }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [settings, setSettings] = useState({
    // Processing settings
    use_cache: true,
    cache_dir: '.requirement_cache',
    extract_tables: true,
    parallel_processing: true,
    max_workers: 3,
    
    // Document settings
    chunk_size: 3,
    max_token_size: 4000,
    use_semantic_similarity: false,
    
    // Verification settings
    confidence_threshold: 0.8,
    retry_attempts: 3,
    
    // Learning settings
    adaptive_learning: true,
    patterns_file: 'requirement_patterns.json'
  });
  
  // Initialize settings from config
  useEffect(() => {
    if (config) {
      setSettings(prev => ({
        ...prev,
        ...config
      }));
    }
  }, [config]);
  
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  const handleCheckboxChange = (event) => {
    const { name, checked } = event.target;
    setSettings(prev => ({ ...prev, [name]: checked }));
  };
  
  const handleTextFieldChange = (event) => {
    const { name, value } = event.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };
  
  const handleNumberChange = (event) => {
    const { name, value } = event.target;
    setSettings(prev => ({ ...prev, [name]: Number(value) }));
  };
  
  const handleSliderChange = (name) => (event, newValue) => {
    setSettings(prev => ({ ...prev, [name]: newValue }));
  };
  
  const handleSave = () => {
    onSave(settings);
  };
  
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      aria-labelledby="advanced-settings-dialog-title"
    >
      <DialogTitle id="advanced-settings-dialog-title">
        Advanced Settings
      </DialogTitle>
      
      <DialogContent dividers>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange} 
            aria-label="advanced settings tabs"
          >
            <Tab label="Processing" />
            <Tab label="Document" />
            <Tab label="Verification" />
            <Tab label="Learning" />
          </Tabs>
        </Box>
        
        {/* Processing Tab */}
        <TabPanel value={activeTab} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={settings.use_cache}
                      onChange={handleCheckboxChange}
                      name="use_cache"
                    />
                  }
                  label="Use Cache"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={settings.extract_tables}
                      onChange={handleCheckboxChange}
                      name="extract_tables"
                    />
                  }
                  label="Extract Tables"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={settings.parallel_processing}
                      onChange={handleCheckboxChange}
                      name="parallel_processing"
                    />
                  }
                  label="Parallel Processing"
                />
              </FormGroup>
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Cache Directory"
                name="cache_dir"
                value={settings.cache_dir}
                onChange={handleTextFieldChange}
                disabled={!settings.use_cache}
                margin="normal"
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Max Workers"
                name="max_workers"
                type="number"
                value={settings.max_workers}
                onChange={handleNumberChange}
                disabled={!settings.parallel_processing}
                inputProps={{ min: 1, max: 10 }}
                margin="normal"
              />
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Document Tab */}
        <TabPanel value={activeTab} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>
                Chunk Size (pages)
              </Typography>
              <Slider
                value={settings.chunk_size}
                onChange={handleSliderChange('chunk_size')}
                valueLabelDisplay="auto"
                step={1}
                marks
                min={1}
                max={10}
                sx={{ marginBottom: 4 }}
              />
              <TextField 
                fullWidth 
                label="Chunk Size" 
                type="number" 
                name="chunk_size"
                value={settings.chunk_size}
                onChange={handleNumberChange}
                InputProps={{
                  endAdornment: <InputAdornment position="end">pages</InputAdornment>,
                }}
                inputProps={{ min: 1, max: 10 }}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>
                Max Token Size
              </Typography>
              <Slider
                value={settings.max_token_size}
                onChange={handleSliderChange('max_token_size')}
                valueLabelDisplay="auto"
                step={100}
                min={1000}
                max={8000}
                sx={{ marginBottom: 4 }}
              />
              <TextField 
                fullWidth 
                label="Max Token Size" 
                type="number" 
                name="max_token_size"
                value={settings.max_token_size}
                onChange={handleNumberChange}
                InputProps={{
                  endAdornment: <InputAdornment position="end">tokens</InputAdornment>,
                }}
                inputProps={{ min: 1000, max: 8000, step: 100 }}
              />
            </Grid>
            
            <Grid item xs={12}>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={settings.use_semantic_similarity}
                      onChange={handleCheckboxChange}
                      name="use_semantic_similarity"
                    />
                  }
                  label="Use Semantic Similarity"
                />
              </FormGroup>
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Verification Tab */}
        <TabPanel value={activeTab} index={2}>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>
                Confidence Threshold
              </Typography>
              <Slider
                value={settings.confidence_threshold}
                onChange={handleSliderChange('confidence_threshold')}
                valueLabelDisplay="auto"
                step={0.1}
                marks
                min={0.1}
                max={1.0}
                sx={{ marginBottom: 4 }}
              />
              <TextField 
                fullWidth 
                label="Confidence Threshold" 
                type="number" 
                name="confidence_threshold"
                value={settings.confidence_threshold}
                onChange={handleNumberChange}
                inputProps={{ min: 0.1, max: 1.0, step: 0.1 }}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Typography gutterBottom>
                Retry Attempts
              </Typography>
              <Slider
                value={settings.retry_attempts}
                onChange={handleSliderChange('retry_attempts')}
                valueLabelDisplay="auto"
                step={1}
                marks
                min={1}
                max={5}
                sx={{ marginBottom: 4 }}
              />
              <TextField 
                fullWidth 
                label="Retry Attempts" 
                type="number" 
                name="retry_attempts"
                value={settings.retry_attempts}
                onChange={handleNumberChange}
                inputProps={{ min: 1, max: 5 }}
              />
            </Grid>
          </Grid>
        </TabPanel>
        
        {/* Learning Tab */}
        <TabPanel value={activeTab} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormGroup>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={settings.adaptive_learning}
                      onChange={handleCheckboxChange}
                      name="adaptive_learning"
                    />
                  }
                  label="Adaptive Learning"
                />
              </FormGroup>
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Patterns File"
                name="patterns_file"
                value={settings.patterns_file}
                onChange={handleTextFieldChange}
                disabled={!settings.adaptive_learning}
                margin="normal"
              />
            </Grid>
          </Grid>
        </TabPanel>
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} color="primary">
          Cancel
        </Button>
        <Button onClick={handleSave} color="primary" variant="contained">
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AdvancedSettingsDialog;