"""
Configuration Manager for PDF Requirements Extractor

This module handles loading, saving, and managing configuration settings
for the application and its model providers.
"""

import os
import json
from typing import Dict, Any, Optional, List


class ConfigManager:
    """
    Manages application configuration, including model provider settings
    """
    
    DEFAULT_CONFIG_FILE = "requirements_extractor_config.json"
    
    DEFAULT_CONFIG = {
        # Application settings
        "app": {
            "chunk_size": 3,
            "max_token_size": 4000,
            "confidence_threshold": 0.8,
            "use_cache": True,
            "cache_dir": ".requirement_cache",
            "parallel_processing": True,
            "max_workers": 3,
            "extract_tables": True,
            "retry_attempts": 3,
            "adaptive_learning": True,
            "patterns_file": "requirement_patterns.json",
            "use_semantic_similarity": False
        },
        
        # Provider configurations
        "providers": {
            "openai": {
                "api_key": None,  # Will be loaded from environment if not specified
                "default_model": "gpt-4o-mini",
                "enabled": True
            },
            "anthropic": {
                "api_key": None,  # Will be loaded from environment if not specified
                "default_model": "claude-3-5-haiku-latest",
                "enabled": False
            },
            "together": {
                "api_key": None,  # Will be loaded from environment if not specified
                "default_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "enabled": False
            }
            # Additional providers can be added here
        },
        
        # Extraction settings
        "extraction": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "verification_strategy": "different",  # same, different, specific
            "verification_provider": None,  # If None, use different provider
            "verification_model": None     # If None, use default model of verification provider
        }
    }
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file, or None to use default
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> bool:
        """
        Load configuration from file
        
        Returns:
            True if config was loaded, False otherwise
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update config with loaded values, preserving defaults for missing keys
                self._update_nested_dict(self.config, loaded_config)
                return True
            return False
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def save_config(self) -> bool:
        """
        Save current configuration to file
        
        Returns:
            True if config was saved, False otherwise
        """
        try:
            # Ensure sensitive information like API keys is not saved
            # unless explicitly requested
            config_to_save = self._prepare_config_for_saving()
            
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration settings"""
        return self.config["app"].copy()
    
    def get_provider_config(self, provider_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider
        
        Args:
            provider_id: Provider identifier
        
        Returns:
            Provider configuration dictionary
        """
        providers = self.config.get("providers", {})
        return providers.get(provider_id, {}).copy()
    
    def get_extraction_config(self) -> Dict[str, Any]:
        """Get extraction configuration settings"""
        return self.config["extraction"].copy()
    
    def update_app_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update application configuration
        
        Args:
            new_config: New configuration values
        """
        self.config["app"].update(new_config)
    
    def update_provider_config(self, provider_id: str, new_config: Dict[str, Any]) -> None:
        """
        Update configuration for a specific provider
        
        Args:
            provider_id: Provider identifier
            new_config: New configuration values
        """
        if "providers" not in self.config:
            self.config["providers"] = {}
        
        if provider_id not in self.config["providers"]:
            self.config["providers"][provider_id] = {}
        
        self.config["providers"][provider_id].update(new_config)
    
    def update_extraction_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update extraction configuration
        
        Args:
            new_config: New configuration values
        """
        self.config["extraction"].update(new_config)
    
    def get_enabled_providers(self) -> List[str]:
        """
        Get list of enabled provider IDs
        
        Returns:
            List of enabled provider IDs
        """
        enabled = []
        for provider_id, config in self.config.get("providers", {}).items():
            if config.get("enabled", False):
                enabled.append(provider_id)
        return enabled
    
    def _update_nested_dict(self, target: Dict, source: Dict) -> None:
        """
        Update a nested dictionary with values from another dictionary
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value
    
    def _prepare_config_for_saving(self) -> Dict[str, Any]:
        """
        Prepare configuration for saving, removing sensitive information
        
        Returns:
            Sanitized configuration dictionary
        """
        config_to_save = self.config.copy()
        
        # Don't save API keys unless they were explicitly set in the config
        # (as opposed to being loaded from environment variables)
        for provider_id, provider_config in config_to_save.get("providers", {}).items():
            if "api_key" in provider_config and provider_config["api_key"] == os.environ.get(f"{provider_id.upper()}_API_KEY"):
                provider_config["api_key"] = None
        
        return config_to_save