"""
Model Provider Abstraction Layer for PDF Requirements Extractor

This module provides a framework for integrating various AI model providers
into the requirements extraction system.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


class ModelProvider(ABC):
    """
    Abstract base class for all model providers.
    Each provider must implement these methods.
    """
    
    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """Return the human-readable name of this provider"""
        pass
    
    @classmethod
    @abstractmethod
    def get_available_models(cls) -> List[str]:
        """Return a list of available models from this provider"""
        pass
    
    @abstractmethod
    def extract_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Send a prompt to the model and get a text response for requirements extraction
        
        Args:
            prompt: The prompt to send to the model
            model: Specific model to use (or default if None)
            **kwargs: Additional provider-specific options
            
        Returns:
            The model's response as text
        """
        pass
    
    @abstractmethod
    def verify_text(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """
        Send a verification prompt and get structured data response
        
        Args:
            prompt: The verification prompt to send
            model: Specific model to use (or default if None)
            **kwargs: Additional provider-specific options
            
        Returns:
            Dictionary containing the verification results
        """
        pass
    
    @abstractmethod
    def initialize(self, api_key: str = None, **kwargs) -> bool:
        """
        Initialize the provider with necessary credentials
        
        Args:
            api_key: The API key for this provider
            **kwargs: Additional provider-specific settings
            
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is properly initialized and available for use
        
        Returns:
            True if the provider is ready to use, False otherwise
        """
        pass
    
    def get_default_model(self) -> str:
        """
        Get the default model for this provider
        
        Returns:
            The name of the default model
        """
        models = self.get_available_models()
        return models[0] if models else ""


class OpenAIProvider(ModelProvider):
    """Implementation of ModelProvider for OpenAI API"""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def __init__(self):
        self.api_key = None
        self.client = None
        self.initialized = False
        self.available_models = [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-o4-mini",
            "gpt-3.5-turbo"
        ]
    
    @classmethod
    def get_provider_name(cls) -> str:
        return "OpenAI"
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        return [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-o4-mini",
            "gpt-3.5-turbo"
        ]
    
    def get_default_model(self) -> str:
        return self.DEFAULT_MODEL
    
    def initialize(self, api_key: str = None, **kwargs) -> bool:
        try:
            import openai
            
            # Get API key from parameter, environment, or kwargs
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            
            # Initialize client
            if self.api_key:
                openai.api_key = self.api_key
                self.client = openai
                self.initialized = True
                return True
            
            return False
        except ImportError:
            print("OpenAI package not installed. Please install with 'pip install openai'")
            return False
    
    def is_available(self) -> bool:
        return self.initialized and self.api_key is not None
    
    def extract_text(self, prompt: str, model: str = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI provider not initialized")
            
        model = model or self.DEFAULT_MODEL
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error calling OpenAI API: {str(e)}")
    
    def verify_text(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("OpenAI provider not initialized")
            
        model = model or self.DEFAULT_MODEL
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            raise RuntimeError(f"Error during verification with OpenAI API: {str(e)}")


class AnthropicProvider(ModelProvider):
    """Implementation of ModelProvider for Anthropic API"""
    
    DEFAULT_MODEL = "claude-3-5-haiku-latest"
    
    def __init__(self):
        self.api_key = None
        self.client = None
        self.initialized = False
        self.available_models = [
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest", 
            "claude-3-opus", 
            "claude-3-5-sonnet"
        ]
    
    @classmethod
    def get_provider_name(cls) -> str:
        return "Anthropic"
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        return [
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest", 
            "claude-3-opus", 
            "claude-3-5-sonnet"
        ]
    
    def get_default_model(self) -> str:
        return self.DEFAULT_MODEL
    
    def initialize(self, api_key: str = None, **kwargs) -> bool:
        try:
            import anthropic
            
            # Get API key from parameter, environment or kwargs
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            
            # Initialize client
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.initialized = True
                return True
            
            return False
        except ImportError:
            print("Anthropic package not installed. Please install with 'pip install anthropic'")
            return False
    
    def is_available(self) -> bool:
        return self.initialized and self.api_key is not None
    
    def extract_text(self, prompt: str, model: str = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Anthropic provider not initialized")
            
        model = model or self.DEFAULT_MODEL
        max_tokens = kwargs.get("max_tokens", 4000)
        
        try:
            response = self.client.messages.create(
                model=model,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=max_tokens
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Error calling Anthropic API: {str(e)}")
    
    def verify_text(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Anthropic provider not initialized")
            
        model = model or self.DEFAULT_MODEL
        max_tokens = kwargs.get("max_tokens", 4000)
        
        try:
            response = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            
            # Extract text and parse JSON
            response_text = response.content[0].text
            
            import json
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                # Attempt to extract JSON from the text if it's not pure JSON
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                    return result
                    
                raise RuntimeError("Failed to parse Anthropic response as JSON")
                
        except Exception as e:
            raise RuntimeError(f"Error during verification with Anthropic API: {str(e)}")


class TogetherAIProvider(ModelProvider):
    """Implementation of ModelProvider for Together.ai API"""
    
    DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    def __init__(self):
        self.api_key = None
        self.initialized = False
        self.available_models = [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-2-70b-chat-hf",
            "togethercomputer/llama-2-7b-chat",
            "codellama/CodeLlama-34b-Instruct-hf"
        ]
    
    @classmethod
    def get_provider_name(cls) -> str:
        return "Together.ai"
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        return [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-2-70b-chat-hf",
            "togethercomputer/llama-2-7b-chat",
            "codellama/CodeLlama-34b-Instruct-hf"
        ]
    
    def get_default_model(self) -> str:
        return self.DEFAULT_MODEL
    
    def initialize(self, api_key: str = None, **kwargs) -> bool:
        try:
            import requests
            
            # Get API key from parameter, environment, or kwargs
            self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
            
            # Initialize if API key is provided
            if self.api_key:
                # Test the API key with a minimal request
                headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                url = "https://api.together.xyz/models"
                response = requests.get(url, headers=headers)
                
                # Check if the API key is valid
                if response.status_code == 200:
                    self.initialized = True
                    # Optionally update available models list from API response
                    return True
                else:
                    print(f"Error initializing Together.ai: {response.status_code} - {response.text}")
                    return False
            
            return False
        except ImportError:
            print("Requests package not installed. Please install with 'pip install requests'")
            return False
    
    def is_available(self) -> bool:
        return self.initialized and self.api_key is not None
    
    def extract_text(self, prompt: str, model: str = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Together.ai provider not initialized")
            
        model = model or self.DEFAULT_MODEL
        
        try:
            import requests
            import json
            
            url = "https://api.together.xyz/inference"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Format request based on the model
            if "llama" in model.lower():
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            elif "mistral" in model.lower():
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            else:
                formatted_prompt = prompt
            
            data = {
                "model": model,
                "prompt": formatted_prompt,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0)
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                return result.get("output", {}).get("choices", [{}])[0].get("text", "")
            else:
                raise RuntimeError(f"Error from Together.ai API: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise RuntimeError(f"Error calling Together.ai API: {str(e)}")
    
    def verify_text(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Together.ai provider not initialized")
            
        model = model or self.DEFAULT_MODEL
        
        try:
            # First get the text response
            response_text = self.extract_text(prompt, model, **kwargs)
            
            # Try to parse it as JSON
            import json
            import re
            
            # Clean the response - some models might wrap JSON in markdown code blocks
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, response_text)
            if match:
                json_str = match.group(1)
            else:
                json_str = response_text
            
            # Try to find JSON object in the text
            try:
                # Look for the first { and the last } to extract JSON
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_obj = json_str[start_idx:end_idx+1]
                    result = json.loads(json_obj)
                    return result
                else:
                    raise ValueError("No JSON object found in response")
                    
            except json.JSONDecodeError:
                # If parsing fails, try to construct a minimal valid response
                if "verified" in response_text.lower():
                    is_verified = "true" in response_text.lower() or "yes" in response_text.lower()
                    confidence = 0.9 if is_verified else 0.2
                    reason = "Based on model assessment (non-JSON response)"
                    
                    return {
                        "verified": is_verified,
                        "confidence": confidence,
                        "reason": reason
                    }
                else:
                    raise ValueError("Could not extract verification result from response")
                
        except Exception as e:
            raise RuntimeError(f"Error during verification with Together.ai API: {str(e)}")

# You can add more providers here, like:
# class GeminiProvider(ModelProvider):
#     """Implementation for Google's Gemini models"""
#     pass