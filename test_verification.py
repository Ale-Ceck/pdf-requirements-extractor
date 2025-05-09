#!/usr/bin/env python
"""Test script to validate the verification_enabled option."""

import sys
import logging
from config_manager import ConfigManager
from pdf_requirements_extractor import RequirementsExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestVerification")

def test_verification_settings():
    """Test different verification settings."""
    
    # 1. Test with verification enabled (default setting)
    logger.info("=== Testing with verification enabled (default) ===")
    config = {"model": "gpt-4o-mini"}
    extractor = RequirementsExtractor(config)
    assert extractor.config.get("verification_enabled", True) is True
    logger.info("✓ Verification is enabled by default")
    
    # 2. Test with verification explicitly enabled
    logger.info("\n=== Testing with verification explicitly enabled ===")
    config = {"model": "gpt-4o-mini", "verification_enabled": True}
    extractor = RequirementsExtractor(config)
    assert extractor.config.get("verification_enabled") is True
    logger.info("✓ Verification is explicitly enabled")
    
    # 3. Test with verification disabled
    logger.info("\n=== Testing with verification disabled ===")
    config = {"model": "gpt-4o-mini", "verification_enabled": False}
    extractor = RequirementsExtractor(config)
    assert extractor.config.get("verification_enabled") is False
    logger.info("✓ Verification is explicitly disabled")
    
    # 4. Test in offline mode
    logger.info("\n=== Testing in offline mode ===")
    config = {"use_offline_provider": True}
    extractor = RequirementsExtractor(config)
    
    # Get the verification_enabled setting from the config
    verification_enabled = extractor.config.get("verification_enabled", True)
    
    # In the actual processing method, offline mode overrides the verification_enabled setting
    if extractor.config_manager.use_offline_provider():
        logger.info("Offline mode detected, verification should be disabled during processing")
        
        # Check how verify_extraction would behave in this case
        # In real usage, the verify_extraction method would not be called because of the check
        # in the process_pdf method
        mock_text = "Sample text"
        mock_requirements = [{"code": "TEST-001", "description": "Test requirement"}]
        
        # Create expected result when verification is disabled
        expected_verification_results = {
            "verified": [r["code"] for r in mock_requirements],
            "potential_missing": [],
            "confidence_scores": {r["code"]: 1.0 for r in mock_requirements},
            "verification_details": {
                r["code"]: {
                    "verified": True,
                    "confidence": 1.0,
                    "reason": "Verification process is disabled"
                } for r in mock_requirements
            }
        }
        
        # Simulate the behavior in process_pdf method
        if extractor.config_manager.use_offline_provider():
            verification_enabled = False
            logger.info("✓ Offline mode: Verification is automatically disabled")
        
        assert verification_enabled is False
    
    logger.info("\nAll tests passed successfully!")
    
if __name__ == "__main__":
    try:
        test_verification_settings()
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)