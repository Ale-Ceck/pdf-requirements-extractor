"""
Model Provider Registry for PDF Requirements Extractor

This module implements a registry pattern for model providers,
allowing dynamic registration and discovery of providers.
"""

from typing import Dict, List, Type, Any, Optional
from model_providers import ModelProvider


class ModelProviderRegistry:
    """
    Registry for model providers that allows dynamic registration and discovery
    of different AI model providers throughout the application.
    """
    
    _providers: Dict[str, Type[ModelProvider]] = {}
    _initialized_providers: Dict[str, ModelProvider] = {}
    
    @classmethod
    def register(cls, provider_id: str, provider_class: Type[ModelProvider]) -> None:
        """
        Register a new model provider with the registry
        
        Args:
            provider_id: Unique identifier for this provider
            provider_class: The provider class (must inherit from ModelProvider)
        """
        if not issubclass(provider_class, ModelProvider):
            raise TypeError("Provider class must inherit from ModelProvider")
        
        cls._providers[provider_id] = provider_class
    
    @classmethod
    def get_provider_class(cls, provider_id: str) -> Type[ModelProvider]:
        """
        Get a provider class by its ID
        
        Args:
            provider_id: The provider identifier
            
        Returns:
            The provider class
            
        Raises:
            ValueError: If the provider is not registered
        """
        if provider_id not in cls._providers:
            raise ValueError(f"Provider '{provider_id}' not registered")
        
        return cls._providers[provider_id]
    
    @classmethod
    def get_provider(cls, provider_id: str, initialize: bool = True, **kwargs) -> ModelProvider:
        """
        Get an instance of a provider by its ID, optionally initializing it
        
        Args:
            provider_id: The provider identifier
            initialize: Whether to initialize the provider if not already initialized
            **kwargs: Parameters to pass to the provider's initialize method
            
        Returns:
            An instance of the requested provider
            
        Raises:
            ValueError: If the provider is not registered
        """
        # Check if we already have an initialized instance
        if provider_id in cls._initialized_providers:
            return cls._initialized_providers[provider_id]
        
        # Get the provider class
        provider_class = cls.get_provider_class(provider_id)
        
        # Create a new instance
        provider = provider_class()
        
        # Initialize if requested
        if initialize:
            provider.initialize(**kwargs)
        
        # Store the initialized provider
        cls._initialized_providers[provider_id] = provider
        
        return provider
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Return a list of all registered provider IDs
        
        Returns:
            List of provider IDs
        """
        return list(cls._providers.keys())
    
    @classmethod
    def get_provider_info(cls) -> List[Dict[str, Any]]:
        """
        Return detailed information about all registered providers
        
        Returns:
            List of dictionaries with provider information
        """
        return [
            {
                "id": provider_id,
                "name": provider_class.get_provider_name(),
                "models": provider_class.get_available_models()
            }
            for provider_id, provider_class in cls._providers.items()
        ]
    
    @classmethod
    def initialize_all(cls, config: Dict[str, Any]) -> List[str]:
        """
        Initialize all registered providers with their respective configurations
        
        Args:
            config: Dictionary mapping provider IDs to their configuration
            
        Returns:
            List of successfully initialized provider IDs
        """
        successful = []
        
        for provider_id, provider_class in cls._providers.items():
            provider = provider_class()
            provider_config = config.get(provider_id, {})
            
            if provider.initialize(**provider_config):
                cls._initialized_providers[provider_id] = provider
                successful.append(provider_id)
        
        return successful


# Register built-in providers
def register_default_providers():
    """Register the default built-in providers"""
    
    from model_providers import OpenAIProvider, AnthropicProvider, TogetherAIProvider, OllamaProvider
    
    # Register online providers
    ModelProviderRegistry.register("openai", OpenAIProvider)
    ModelProviderRegistry.register("anthropic", AnthropicProvider)
    ModelProviderRegistry.register("together", TogetherAIProvider)
    
    # Register offline providers
    ModelProviderRegistry.register("ollama", OllamaProvider)
    
    # Additional providers can be registered here as they're implemented