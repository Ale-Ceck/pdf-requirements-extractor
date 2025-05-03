import React, { useState, useRef } from 'react';
import {
  Paper,
  Box,
  Typography,
  Button,
  TextField,
  FormControl,
  FormControlLabel,
  RadioGroup,
  Radio,
  Grid,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import FolderIcon from '@mui/icons-material/Folder';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import CloseIcon from '@mui/icons-material/Close';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

// Styled component for the drop zone
const DropZone = styled('div')(({ theme, isDragging, disabled }) => ({
  border: `2px dashed ${isDragging ? theme.palette.primary.main : theme.palette.border.main}`,
  borderRadius: theme.shape.borderRadius * 2,
  backgroundColor: isDragging ? theme.palette.highlight.main : theme.palette.background.alt,
  padding: theme.spacing(3),
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  textAlign: 'center',
  minHeight: 150,
  cursor: disabled ? 'not-allowed' : 'pointer',
  transition: 'all 0.2s ease-in-out',
  opacity: disabled ? 0.7 : 1,
}));

const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

const FileUploadPanel = ({ onProcess, isProcessing }) => {
  const [files, setFiles] = useState([]);
  const [outputPath, setOutputPath] = useState('');
  const [processingType, setProcessingType] = useState('single');
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Handle drag events
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (isProcessing) return;
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (isProcessing) return;
    setIsDragging(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (isProcessing) return;

    const droppedFiles = Array.from(e.dataTransfer.files);
    
    // Check if the files are PDFs for single mode
    if (processingType === 'single') {
      const pdfFiles = droppedFiles.filter(file => file.type === 'application/pdf');
      if (pdfFiles.length > 0) {
        setFiles([pdfFiles[0]]); // Take only the first PDF
        
        // Generate suggested output path
        const fileName = pdfFiles[0].name.replace('.pdf', '') + '_requirements.xlsx';
        setOutputPath(fileName);
      }
    } else {
      // For batch mode, allow directories or multiple PDFs
      setFiles(droppedFiles);
    }
  };

  // Handle file input change
  const handleFileChange = (e) => {
    if (isProcessing) return;
    
    const selectedFiles = Array.from(e.target.files);
    if (selectedFiles.length === 0) return;
    
    if (processingType === 'single') {
      setFiles([selectedFiles[0]]);
      
      // Generate suggested output path
      const fileName = selectedFiles[0].name.replace('.pdf', '') + '_requirements.xlsx';
      setOutputPath(fileName);
    } else {
      setFiles(selectedFiles);
    }
  };

  // Handle click on drop zone
  const handleDropZoneClick = () => {
    if (isProcessing) return;
    fileInputRef.current.click();
  };

  // Handle processing type change
  const handleProcessingTypeChange = (e) => {
    const newType = e.target.value;
    setProcessingType(newType);
    setFiles([]);
    setOutputPath('');
  };

  // Handle file removal
  const handleRemoveFile = () => {
    setFiles([]);
    setOutputPath('');
  };

  // Handle process button click
  const handleProcess = () => {
    if (files.length === 0) return;
    
    onProcess({
      files,
      outputPath,
      isBatch: processingType === 'batch'
    });
  };

  // Render file preview
  const renderFilePreview = () => {
    if (files.length === 0) return null;
    
    return (
      <Paper
        variant="outlined"
        sx={{
          p: 2,
          mt: 2,
          borderRadius: 2,
          display: 'flex',
          alignItems: 'center',
          backgroundColor: 'background.alt',
        }}
      >
        {processingType === 'single' ? (
          <>
            <PictureAsPdfIcon color="primary" sx={{ fontSize: 30, mr: 2 }} />
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="body1" fontWeight="medium">
                {files[0].name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {(files[0].size / 1024 / 1024).toFixed(2)} MB
              </Typography>
            </Box>
          </>
        ) : (
          <>
            <FolderIcon color="primary" sx={{ fontSize: 30, mr: 2 }} />
            <Box sx={{ flexGrow: 1 }}>
              <Typography variant="body1" fontWeight="medium">
                {files.length} file(s) selected
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {processingType === 'batch' ? 'Batch processing' : 'Single file processing'}
              </Typography>
            </Box>
          </>
        )}
        
        {!isProcessing && (
          <IconButton 
            size="small" 
            onClick={handleRemoveFile}
            color="error"
          >
            <CloseIcon />
          </IconButton>
        )}
      </Paper>
    );
  };

  return (
    <Paper
      elevation={1}
      sx={{
        p: 3,
        borderRadius: 2,
      }}
    >
      <Typography variant="h6" gutterBottom>
        File Selection
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <FormControl component="fieldset" sx={{ mb: 2 }}>
            <RadioGroup
              row
              name="processing-type"
              value={processingType}
              onChange={handleProcessingTypeChange}
            >
              <FormControlLabel 
                value="single" 
                control={<Radio />} 
                label="Single File" 
                disabled={isProcessing}
              />
              <FormControlLabel 
                value="batch" 
                control={<Radio />} 
                label="Batch Directory" 
                disabled={isProcessing}
              />
            </RadioGroup>
          </FormControl>
        </Grid>
        
        <Grid item xs={12}>
          <DropZone
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={handleDropZoneClick}
            isDragging={isDragging}
            disabled={isProcessing}
          >
            <VisuallyHiddenInput
              type="file"
              accept={processingType === 'single' ? '.pdf' : ''}
              multiple={processingType === 'batch'}
              ref={fileInputRef}
              onChange={handleFileChange}
              disabled={isProcessing}
            />
            
            <CloudUploadIcon
              sx={{
                fontSize: 60,
                color: 'primary.main',
                mb: 2,
                opacity: 0.8,
              }}
            />
            
            <Typography variant="h6" gutterBottom>
              {processingType === 'single'
                ? 'Drop PDF File Here'
                : 'Drop PDF Files or a Folder Here'
              }
            </Typography>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              or
            </Typography>
            
            <Button
              variant="outlined"
              disabled={isProcessing}
            >
              Browse Files
            </Button>
          </DropZone>
          
          {renderFilePreview()}
        </Grid>
        
        <Grid item xs={12}>
          <TextField
            fullWidth
            label={processingType === 'single' ? 'Output Excel File' : 'Output Directory'}
            placeholder={processingType === 'single' ? 'example_requirements.xlsx' : 'output_directory'}
            value={outputPath}
            onChange={(e) => setOutputPath(e.target.value)}
            disabled={isProcessing}
            sx={{ mb: 3 }}
          />
        </Grid>
        
        <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            startIcon={isProcessing ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
            onClick={handleProcess}
            disabled={isProcessing || files.length === 0}
            sx={{ px: 4 }}
          >
            {isProcessing ? 'Processing...' : 'Process PDF(s)'}
          </Button>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default FileUploadPanel;