"""
Extraction Service for PDF Requirements Extractor

This module provides the high-level services for extracting requirements
using the registered model providers.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from provider_registry import ModelProviderRegistry
from model_providers import ModelProvider


class ExtractionService:
    """
    Service for extracting requirements from text using AI models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the extraction service
        
        Args:
            config: Extraction configuration
        """
        self.config = config
        self.logger = logging.getLogger("ExtractionService")
        
        # Get the primary extraction provider
        provider_id = config.get("provider", "openai")
        try:
            self.extraction_provider = ModelProviderRegistry.get_provider(provider_id)
        except ValueError:
            self.logger.error(f"Provider {provider_id} not registered, falling back to first available")
            available_providers = ModelProviderRegistry.get_available_providers()
            if not available_providers:
                raise RuntimeError("No model providers available")
            
            provider_id = available_providers[0]
            self.extraction_provider = ModelProviderRegistry.get_provider(provider_id)
        
        self.extraction_model = config.get("model") or self.extraction_provider.get_default_model()
    
    def create_extraction_prompt(self, chunk: str) -> str:
        """
        Create a prompt for requirements extraction
        
        Args:
            chunk: Text chunk to extract requirements from
            
        Returns:
            Formatted prompt
        """
        return f"""
        Extract all requirements from the following text. Requirements typically have the following characteristics:
        1. A unique identifier/code (such as REQ-123, SRS-01, FR-100, etc.)
        2. A descriptive statement of what the system must do or a constraint it must meet
        
        For each requirement you identify, provide:
        1. The requirement code EXACTLY as it appears in the text
        2. The requirement description (full text of the requirement)
        
        Format each requirement EXACTLY as follows (keep this exact format):
        CODE: [requirement code]
        DESCRIPTION: [requirement description]
        
        Important guidelines:
        - Extract ALL requirements present in the text, even if the format varies
        - Include the full description, even if it spans multiple paragraphs, including any notes or comments
        - Mantain the original structure and formatting of the requirement text
        - Do not summarize or paraphrase the requirements
        - Don't invent or generate requirements that aren't in the original text
        - Don't make assumptions about the requirement pattern - extract exactly what's there
        - If no requirements are found, respond with "No requirements found."
        
        Here's the text:
        {chunk}
        """
    
    def create_verification_prompt(self, requirement: Dict[str, str], context: str) -> str:
        """
        Create a prompt for requirement verification
        
        Args:
            requirement: The requirement to verify
            context: Relevant context from the source document
            
        Returns:
            Formatted verification prompt
        """
        return f"""
        Verify if this requirement description accurately matches the original text. 
        
        Requirement code: {requirement["code"]}
        Requirement description: {requirement["description"]}
        
        Review the surrounding context from the PDF and determine:
        1. If this code and description are actually present in the original text
        2. If this description is accurate and complete
        3. Assign a confidence score from 0.0 to 1.0
        
        Context from PDF (relevant sample):
        {context}
        
        Return ONLY a JSON object with format:
        {{"verified": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
        """
    
    def extract_requirements(self, chunk: str) -> List[Dict[str, str]]:
        """
        Extract requirements from a text chunk
        
        Args:
            chunk: Text chunk to process
            
        Returns:
            List of extracted requirements
        """
        try:
            # Create prompt
            prompt = self.create_extraction_prompt(chunk)
            
            # Call provider
            llm_output = self.extraction_provider.extract_text(
                prompt=prompt,
                model=self.extraction_model
            )
            
            # Parse requirements
            return self.parse_requirements(llm_output)
            
        except Exception as e:
            self.logger.error(f"Error extracting requirements: {e}")
            return []
    
    def verify_requirement(
        self, 
        requirement: Dict[str, str], 
        context: str
    ) -> Dict[str, Any]:
        """
        Verify an extracted requirement against the original context
        
        Args:
            requirement: The requirement to verify
            context: Relevant context from the source document
            
        Returns:
            Verification result
        """
        # Check if verification is enabled - explicitly log the value to debug
        verification_enabled = self.config.get("verification_enabled", True)
        self.logger.info(f"Verification enabled setting: {verification_enabled}")
        
        # Skip verification if it's disabled
        if not verification_enabled:
            self.logger.info("Verification process is disabled, skipping verification")
            return {
                "verified": True,
                "confidence": 1.0,
                "reason": "Verification process is disabled"
            }
            
        # Determine which provider to use for verification
        verification_strategy = self.config.get("verification_strategy", "different")
        verification_provider = None
        
        if verification_strategy == "same":
            # Use same provider as extraction
            verification_provider = self.extraction_provider
            verification_model = self.extraction_model
            
        elif verification_strategy == "specific":
            # Use specific provider and model
            provider_id = self.config.get("verification_provider")
            if not provider_id:
                # Fall back to different provider
                verification_strategy = "different"
            else:
                try:
                    verification_provider = ModelProviderRegistry.get_provider(provider_id)
                    verification_model = self.config.get("verification_model") or verification_provider.get_default_model()
                except ValueError:
                    # Provider not available, fall back
                    verification_strategy = "different"
        
        if verification_strategy == "different":
            # Use a different provider than extraction
            available_providers = ModelProviderRegistry.get_available_providers()
            
            # Find a provider different from the extraction provider
            for provider_id in available_providers:
                if provider_id != self.config.get("provider"):
                    try:
                        verification_provider = ModelProviderRegistry.get_provider(provider_id)
                        verification_model = verification_provider.get_default_model()
                        break
                    except ValueError:
                        continue
            
            # If no alternative found, use the same provider but try a different model
            if not verification_provider:
                verification_provider = self.extraction_provider
                
                # Try to use a different model
                available_models = verification_provider.get_available_models()
                for model in available_models:
                    if model != self.extraction_model:
                        verification_model = model
                        break
                else:
                    # No alternative model found, use the same model
                    verification_model = self.extraction_model
        
        # Create verification prompt
        prompt = self.create_verification_prompt(requirement, context)
        
        try:
            # Call verification provider
            result = verification_provider.verify_text(
                prompt=prompt,
                model=verification_model
            )
            
            # Ensure the result has the expected format
            if not isinstance(result, dict):
                raise ValueError("Verification result is not a dictionary")
            
            # Ensure required keys are present
            required_keys = ["verified", "confidence", "reason"]
            for key in required_keys:
                if key not in result:
                    result[key] = None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during verification: {e}")
            return {
                "verified": False,
                "confidence": 0.0,
                "reason": f"Verification failed: {str(e)}"
            }
    
    def parse_requirements(self, llm_output: str) -> List[Dict[str, str]]:
        """
        Parse structured requirements from LLM output
        
        Args:
            llm_output: Raw output from the LLM
            
        Returns:
            List of parsed requirements
        """
        if "No requirements found" in llm_output:
            return []
                
        pattern = r"CODE: (.*?)\nDESCRIPTION: (.*?)(?=\nCODE:|$)"
        matches = re.findall(pattern, llm_output, re.DOTALL)
        
        if not matches and "CODE:" in llm_output and "DESCRIPTION:" in llm_output:
            # Fallback parsing for non-standard formatting
            lines = llm_output.split('\n')
            requirements = []
            current_code = None
            current_description = []
            
            for line in lines:
                if line.startswith("CODE:"):
                    if current_code and current_description:
                        requirements.append({
                            "code": current_code.strip(),
                            "description": "\n".join(current_description).strip(),
                            "source_type": "text"
                        })
                    current_code = line.replace("CODE:", "").strip()
                    current_description = []
                elif line.startswith("DESCRIPTION:"):
                    current_description.append(line.replace("DESCRIPTION:", "").strip())
                elif current_description:
                    current_description.append(line)
                    
            if current_code and current_description:
                requirements.append({
                    "code": current_code.strip(),
                    "description": "\n".join(current_description).strip(),
                    "source_type": "text"
                })
            return requirements
        
        requirements = []
        for code, description in matches:
            requirements.append({
                "code": code.strip(),
                "description": description.strip(),
                "source_type": "text"
            })
        
        return requirements