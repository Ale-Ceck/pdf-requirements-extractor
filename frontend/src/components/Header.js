import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';

const Header = () => {
  return (
    <Paper
      elevation={1}
      sx={{
        padding: 2,
        borderRadius: 2,
        display: 'flex',
        alignItems: 'center',
        backgroundColor: 'background.paper',
      }}
    >
      <DescriptionIcon
        sx={{
          fontSize: 40,
          color: 'primary.main',
          marginRight: 2,
        }}
      />
      <Box>
        <Typography variant="h1">
          PDF Requirements Extractor
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Extract and analyze requirements from PDF documents using AI
        </Typography>
      </Box>
    </Paper>
  );
};

export default Header;