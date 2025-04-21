"""
PDF Requirements Extractor

A tool for extracting and validating requirements from PDF documents
using AI-powered natural language processing.
"""

from provider_registry import ModelProviderRegistry, register_default_providers

# Register the default providers
register_default_providers()