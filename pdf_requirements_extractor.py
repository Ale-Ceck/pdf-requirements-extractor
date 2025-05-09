import pdfplumber
import pandas as pd
import re
import os
import sys
import json
import hashlib
import difflib
import concurrent.futures
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv, find_dotenv

# Import refactor architecture components
from model_providers import ModelProvider
from provider_registry import ModelProviderRegistry, register_default_providers
from config_manager import ConfigManager
from extraction_service import ExtractionService

# Register default providers
register_default_providers()


class RequirementsExtractor:
    def __init__(self, config=None):
        """Initialize with configuration."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RequirementsExtractor")
        
        # Load environment variables
        load_dotenv(find_dotenv())
        
        # Initialize with default configuration
        self.config_manager = ConfigManager()
        
        # Get default app configuration
        self.config = self.config_manager.get_app_config()
        
        # Get default extraction configuration
        extraction_config = self.config_manager.get_extraction_config()
        self.config.update(extraction_config)
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Initialize providers
        self._initialize_providers()
        
        # Set up cache directory
        if self.config["use_cache"]:
            os.makedirs(self.config["cache_dir"], exist_ok=True)
            
        # Load previously learned patterns if adaptive learning is enabled
        self.learned_patterns = []
        if self.config["adaptive_learning"] and os.path.exists(self.config["patterns_file"]):
            try:
                with open(self.config["patterns_file"], 'r') as f:
                    self.learned_patterns = json.load(f)
                    self.logger.info(f"Loaded {len(self.learned_patterns)} previously learned patterns")
            except Exception as e:
                self.logger.warning(f"Error loading patterns file: {e}")
        
        # Initialize extraction service
        self.extraction_service = ExtractionService(self.config)

    def _initialize_providers(self):
        """Initialize all model providers"""
        # Check if we should prefer offline providers
        use_offline = self.config_manager.use_offline_provider()
        
        # Get all provider configurations
        provider_configs = {}
        for provider_id in ModelProviderRegistry.get_available_providers():
            provider_configs[provider_id] = self.config_manager.get_provider_config(provider_id)
        
        # Initialize offline providers first if preferred
        if use_offline:
            self.logger.info("Preferring offline providers if available")
            self._initialize_offline_providers(provider_configs)
            
            # Check if any offline providers are available
            offline_providers = self.config_manager.get_offline_providers()
            if offline_providers:
                # Set the first available offline provider as default
                provider_id = offline_providers[0]
                self.config["provider"] = provider_id
                self.logger.info(f"Using offline provider: {provider_id}")
                return
                
            # If no offline providers are available, fall back to online providers
            self.logger.warning("No offline providers available, falling back to online providers")
        
        # Initialize online providers
        self._initialize_online_providers(provider_configs)
        
        # Get primary extraction provider from config
        provider_id = self.config.get("provider", "openai")
        self.logger.info(f"Using provider: {provider_id}")
    
    def _initialize_offline_providers(self, provider_configs):
        """Initialize offline providers"""
        # Initialize Ollama if enabled
        if provider_configs.get("ollama", {}).get("enabled", False):
            ollama_config = provider_configs["ollama"]
            
            try:
                ollama_provider = ModelProviderRegistry.get_provider("ollama", initialize=True, **ollama_config)
                
                if ollama_provider.is_available():
                    self.logger.info(f"Ollama initialized successfully with {len(ollama_provider.available_models)} models")
                    
                    # Set as default provider if offline is preferred
                    if self.config_manager.use_offline_provider():
                        self.config["provider"] = "ollama"
                        self.config["model"] = ollama_provider.get_default_model()
                else:
                    self.logger.warning("Ollama enabled but not available. Check if Ollama is running.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Ollama provider: {e}")
    
    def _initialize_online_providers(self, provider_configs):
        """Initialize online providers"""
        # Check if OpenAI is enabled
        if provider_configs.get("openai", {}).get("enabled", False):
            # Get OpenAI provider config
            openai_config = provider_configs["openai"]
            
            # Update with any API key from user config
            if "api_key" in self.config:
                openai_config["api_key"] = self.config["api_key"]
            
            # Initialize the provider
            try:
                openai_provider = ModelProviderRegistry.get_provider("openai", initialize=True, **openai_config)
                if not openai_provider.is_available():
                    self.logger.warning("OpenAI enabled but API key not found or invalid.")
                else:
                    # Set as default provider if no offline provider is preferred or available
                    if not self.config_manager.use_offline_provider():
                        self.config["provider"] = "openai"
                        self.config["model"] = openai_provider.get_default_model()
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Check if Anthropic is enabled
        if provider_configs.get("anthropic", {}).get("enabled", False):
            # Get Anthropic provider config
            anthropic_config = provider_configs["anthropic"]
            
            # Update with any API key from user config
            if "anthropic_api_key" in self.config:
                anthropic_config["api_key"] = self.config["anthropic_api_key"]
            
            # Initialize the provider
            try:
                anthropic_provider = ModelProviderRegistry.get_provider("anthropic", initialize=True, **anthropic_config)
                if not anthropic_provider.is_available():
                    self.logger.warning("Anthropic enabled but unavailable. Check API key.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic provider: {e}")
        
        # Check if Together.ai is enabled
        if provider_configs.get("together", {}).get("enabled", False):
            # Get Together.ai provider config
            together_config = provider_configs["together"]
            
            # Update with any API key from user config
            if "together_api_key" in self.config:
                together_config["api_key"] = self.config["together_api_key"]
            
            # Initialize the provider
            try:
                together_provider = ModelProviderRegistry.get_provider("together", initialize=True, **together_config)
                if not together_provider.is_available():
                    self.logger.warning("Together.ai enabled but unavailable. Check API key.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Together.ai provider: {e}")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF document, page by page."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        text_by_page = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_by_page.append(page.extract_text() or "")  # Handle None returns
        return text_by_page
    
    def extract_tables_from_pdf(self, pdf_path):
        """Extract tables from PDF."""
        tables_by_page = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    tables_by_page.append(tables)
                else:
                    tables_by_page.append([])
        return tables_by_page
    
    def analyze_pdf_structure(self, text_by_page):
        """Analyze PDF structure to determine optimal chunking strategy."""
        # Check document length
        total_length = sum(len(page) for page in text_by_page)
        
        # Check for section patterns
        section_pattern = re.compile(r'^\s*\d+(\.\d+)*\s+[A-Z]', re.MULTILINE)
        section_counts = [len(section_pattern.findall(page)) for page in text_by_page]
        
        # Determine if document has clear section structure
        has_clear_sections = sum(section_counts) > len(text_by_page) / 3
        
        # Determine optimal chunking method
        if has_clear_sections and total_length > 20000:
            self.logger.info("Using semantic chunking for structured document")
            return "semantic"
        elif total_length < 10000:
            self.logger.info("Document is small, processing as single chunk")
            return "single"
        else:
            self.logger.info("Using standard page-based chunking")
            return "standard"
    
    def process_tables(self, tables_by_page):
        """Extract requirements from tables."""
        table_requirements = []
        
        for page_num, page_tables in enumerate(tables_by_page):
            for table_num, table in enumerate(page_tables):
                # Check if this looks like a requirements table
                if table and len(table) > 1:  # Has headers and data
                    headers = table[0]
                    
                    # Look for requirement code and description columns
                    code_col = None
                    desc_col = None
                    
                    for i, header in enumerate(headers):
                        if header and isinstance(header, str):
                            header_lower = header.lower()
                            if any(term in header_lower for term in ['id', 'code', 'req', 'number']):
                                code_col = i
                            elif any(term in header_lower for term in ['desc', 'text', 'requirement', 'content']):
                                desc_col = i
                    
                    # Extract requirements if we found the right columns
                    if code_col is not None and desc_col is not None:
                        for row in table[1:]:  # Skip header
                            if len(row) > max(code_col, desc_col) and row[code_col] and row[desc_col]:
                                code = str(row[code_col]).strip()
                                desc = str(row[desc_col]).strip()
                                if code and desc:
                                    table_requirements.append({
                                        "code": code,
                                        "description": desc,
                                        "source": f"Table {table_num+1}, Page {page_num+1}",
                                        "source_type": "table"
                                    })
        
        self.logger.info(f"Extracted {len(table_requirements)} requirements from tables")
        return table_requirements
    
    def chunk_document(self, text_by_page, chunk_size=None):
        """Split document into manageable chunks by pages."""
        if chunk_size is None:
            chunk_size = self.config["chunk_size"]
            
        chunks = []
        for i in range(0, len(text_by_page), chunk_size):
            chunk = "\n\n".join(text_by_page[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    def semantic_chunk_document(self, text_by_page):
        """Split document based on section breaks or requirement boundaries."""
        full_text = "\n\n".join(text_by_page)
        # Look for section dividers like headers
        # This regex matches common section headers like "## Section Title", "2.1 Section Title", etc.
        sections = re.split(r'(?:\n\s*#{1,3}\s+|\n\s*\d+\.\d*\s+[A-Z]|\n\s*SECTION \d+:)', full_text)
        chunks = []
        
        current_chunk = ""
        current_size = 0
        max_size = self.config["max_token_size"]  # Approximate token limit
        
        for section in sections:
            if not section.strip():
                continue
                
            section_size = len(section) // 4  # Rough char-to-token estimate
            if current_size + section_size > max_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = section
                current_size = section_size
            else:
                current_chunk += section
                current_size += section_size
                
        if current_chunk:
            chunks.append(current_chunk)
        
        self.logger.info(f"Document semantically split into {len(chunks)} chunks.")
        return chunks
    
    def get_cache_key(self, content):
        """Generate a cache key for the content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
    def extract_requirements_with_retry(self, chunk):
        """Extract requirements with automatic retry on failure."""
        try:
            if self.config["use_cache"]:
                return self.extract_requirements_with_cache(chunk)
            else:
                # Use our extraction service
                llm_output = self.extract_requirements_without_cache(chunk)
                return llm_output
        except Exception as e:
            self.logger.error(f"Error in LLM call, retrying: {e}")
            raise  # Re-raise to trigger retry
    
    def extract_requirements_with_cache(self, chunk):
        """Extract requirements with caching."""
        cache_key = self.get_cache_key(chunk)
        cache_file = os.path.join(self.config["cache_dir"], f"{cache_key}.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                try:
                    cached_result = json.load(f)
                    self.logger.info("Using cached LLM response")
                    return cached_result['response']
                except Exception as e:
                    self.logger.warning(f"Cache file corrupted: {e}")
                    # Cache may be corrupted, proceed to API call
        
        # Make API call
        response = self.extract_requirements_without_cache(chunk)
        
        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump({'response': response}, f)
        except Exception as e:
            self.logger.warning(f"Failed to write to cache: {e}")
        
        return response
    
    def extract_requirements_without_cache(self, chunk):
        """Extract requirements using the configured provider through our extraction service"""
        try:
            # Create extraction prompt
            prompt = self.extraction_service.create_extraction_prompt(chunk)
            
            # Get provider from registry
            provider_id = self.config.get("provider", "openai")
            provider = ModelProviderRegistry.get_provider(provider_id)
            
            # Get the model to use
            model = self.config.get("model") or provider.get_default_model()
            
            # Call the provider
            response = provider.extract_text(prompt=prompt, model=model)
            return response
            
        except Exception as e:
            self.logger.error(f"Error extracting requirements: {e}")
            
            # Try fallback providers if available
            fallback_providers = []
            
            # If main provider was OpenAI, try Anthropic as fallback
            if provider_id == "openai" and self.config.get("enable_anthropic", False):
                fallback_providers.append("anthropic")
            # If main provider was Anthropic, try OpenAI as fallback
            elif provider_id == "anthropic":
                fallback_providers.append("openai")
            
            # Try fallback providers
            for fallback_id in fallback_providers:
                try:
                    self.logger.info(f"Attempting fallback to {fallback_id}")
                    fallback = ModelProviderRegistry.get_provider(fallback_id)
                    if fallback.is_available():
                        fallback_model = fallback.get_default_model()
                        response = fallback.extract_text(prompt=prompt, model=fallback_model)
                        return response
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback to {fallback_id} failed: {fallback_error}")
            
            # If all providers fail, re-raise the original exception
            raise RuntimeError(f"All extraction providers failed: {e}")
    
    def parse_requirements(self, llm_output):
        """Parse structured requirements from LLM output with improved error handling."""
        # Use our new extraction service to parse requirements
        return self.extraction_service.parse_requirements(llm_output)
    
    def process_single_chunk(self, chunk):
        """Process a single chunk to extract requirements."""
        try:
            llm_output = self.extract_requirements_with_retry(chunk)
            return self.parse_requirements(llm_output)
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}")
            return []
    
    def process_chunks_parallel(self, chunks, max_workers=None):
        """Process chunks in parallel."""
        if max_workers is None:
            max_workers = self.config["max_workers"]
            
        all_requirements = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(self.process_single_chunk, chunk): i for i, chunk in enumerate(chunks)}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    chunk_requirements = future.result()
                    self.logger.info(f"Processed chunk {chunk_index+1}/{len(chunks)}: found {len(chunk_requirements)} requirements")
                    all_requirements.extend(chunk_requirements)
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_index+1}: {e}")
        return all_requirements
    
    def process_chunks_sequential(self, chunks):
        """Process chunks sequentially."""
        all_requirements = []
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            chunk_requirements = self.process_single_chunk(chunk)
            all_requirements.extend(chunk_requirements)
        return all_requirements
    
    def normalize_whitespace(self, s):
        return re.sub(r'\s+', ' ', s).strip().lower()

    def validate_requirements(self, requirements, full_text=None):
        """Validate extracted requirements for quality."""
        validation_results = []

        if full_text is None:
            self.logger.error("Full text is required for validation")
            return validation_results
        
        for req in requirements:
            req_code = req.get("code")
            req_description = req.get("description")
            result = {"code": req_code, "issues": [], "similarity_score": None}
            
            code_found_in_text = re.search(re.escape(req_code), full_text)
            # Check if code is found in the text
            if not code_found_in_text:
                result["issues"].append(f"Code '{req_code}' not found in text")
            else:
                # Extract relevant context after the code
                start_pos = code_found_in_text.end()
                context = full_text[start_pos:min(len(full_text), start_pos + 2000)]
                # Normalize whitespace for better matching
                context_clean = self.normalize_whitespace(context)
                description_clean = self.normalize_whitespace(req_description)

                # Check if the description is present in the context
                if description_clean in context_clean:
                    result["similarity_score"] = 1.0
                else:
                    # If not, check for similarity
                    s = difflib.SequenceMatcher(None, description_clean, context_clean)
                    similarity = s.ratio()
                    result["similarity_score"] = round(similarity, 2)
                    result["issues"].append(f"Manual check required: Description does not perfectly match the text (similarity: {similarity:.2f})")

            # Set validation status
            result["status"] = "valid" if not result["issues"] else "warning"
            validation_results.append(result)
        
        return validation_results
    
    def find_relevant_context(self, full_text, requirement):
        """Find relevant context for a requirement in the full text."""
        # Try to find the requirement code in the text first
        code_matches = re.finditer(re.escape(requirement["code"]), full_text)
        contexts = []
        
        for match in code_matches:
            start_pos = max(0, match.start() - 50)
            end_pos = min(len(full_text), match.end() + 2000)
            contexts.append(full_text[start_pos:end_pos])
            
            # Limit to 3 context windows to keep the prompt size manageable
            if len(contexts) >= 3:
                break
                
        if not contexts:
            # If code not found, try keyword search using significant words from description
            words = re.findall(r'\b[A-Za-z]{5,}\b', requirement["description"])
            if words:
                # Use the 5 most significant words (likely not stopwords due to length)
                for word in words[:5]:
                    if word.lower() in full_text.lower():
                        # Case-insensitive keyword search
                        idx = full_text.lower().find(word.lower())
                        start_pos = max(0, idx - 300)
                        end_pos = min(len(full_text), idx + 700)
                        contexts.append(full_text[start_pos:end_pos])
                        if len(contexts) >= 3:
                            break
        
        # If still no context found, just use the first part of the text
        if not contexts:
            contexts.append(full_text[:2000])
            
        return "\n...\n".join(contexts)
    
    def verify_extraction(self, pdf_text, extracted_requirements):
        """Verify extracted requirements using a different LLM to reduce bias."""
        verification_results = {
            "verified": [],
            "potential_missing": [],
            "confidence_scores": {},
            "verification_details": {}
        }
        
        # Verify each requirement
        for req in extracted_requirements:
            # Find relevant context for this requirement
            relevant_context = self.find_relevant_context(pdf_text, req)
            
            try:
                # Use extraction service to verify the requirement
                result = self.extraction_service.verify_requirement(req, relevant_context)
                
                # Store verification results
                verification_results["confidence_scores"][req["code"]] = result["confidence"]
                verification_results["verification_details"][req["code"]] = result
                
                if result["verified"] and result["confidence"] > self.config["confidence_threshold"]:
                    verification_results["verified"].append(req["code"])
                else:
                    verification_results["potential_missing"].append(
                        f"{req['code']} (verification failed: {result['reason']})"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error during verification of {req['code']}: {e}")
                # Fallback to the original similarity-based method
                max_similarity = 0
                description = req["description"]
                
                for i in range(0, len(pdf_text) - 50, 50):
                    substr = pdf_text[i:i+200]
                    similarity = difflib.SequenceMatcher(None, description, substr).ratio()
                    max_similarity = max(max_similarity, similarity)
                
                verification_results["confidence_scores"][req["code"]] = max_similarity
                verification_results["verification_details"][req["code"]] = {
                    "verified": max_similarity > self.config["confidence_threshold"],
                    "confidence": max_similarity,
                    "reason": "Fallback to string similarity due to LLM error",
                    "error": str(e)
                }
                
                if max_similarity > self.config["confidence_threshold"]:
                    verification_results["verified"].append(req["code"])
                else:
                    verification_results["potential_missing"].append(
                        f"{req['code']} (low confidence: {max_similarity:.2f})"
                    )
        
        return verification_results
    
    def merge_and_deduplicate_requirements(self, requirements):
        """Merge and deduplicate requirements from different sources."""
        # Create a dictionary to hold unique requirements by code
        unique_reqs = {}
        
        for req in requirements:
            code = req["code"]
            if code in unique_reqs:
                # If this requirement is from a table and the existing one is from text,
                # or if this description is longer, prefer this one
                existing = unique_reqs[code]
                if (req.get("source_type") == "table" and existing.get("source_type") != "table") or \
                   (len(req["description"]) > len(existing["description"])):
                    unique_reqs[code] = req
            else:
                unique_reqs[code] = req
        
        return list(unique_reqs.values())
    
    def find_potential_missing_requirements(self, pdf_text, extracted_requirements):
        """Use LLM to identify potentially missed requirements without relying on patterns."""
        
        # Check if verification is enabled
        verification_enabled = self.config.get("verification_enabled", True)
        
        # For offline mode, always disable verification
        if self.config_manager.use_offline_provider():
            verification_enabled = False
        
        # Skip the process if verification is disabled
        if not verification_enabled:
            self.logger.info("Verification process is disabled, skipping missing requirements check")
            return []
        
        # Get extracted codes for reference
        extracted_codes = [req["code"] for req in extracted_requirements]
        
        # Create a prompt for finding missed requirements
        prompt = f"""
        Review this text and identify any requirement codes that appear to be requirements 
        but might have been missed in our extraction.
        
        Requirements typically have a code/identifier followed by a description.
        We've already identified these requirement codes: {', '.join(extracted_codes)}
        
        Please identify any OTHER requirement codes that appear in the text but are NOT in the list above.
        Return your answer as a JSON list of strings, containing only the requirement codes.
        If you don't find any additional requirements, return an empty list.
        
        Text to analyze:
        {pdf_text}
        """
        
        # Use the same strategy as verification to select a provider
        verification_strategy = self.config.get("verification_strategy", "different")
        provider_id = self.config.get("provider", "openai")
        
        # If verification strategy is "different", use a different provider than extraction
        if verification_strategy == "different":
            available_providers = ModelProviderRegistry.get_available_providers()
            for p_id in available_providers:
                if p_id != provider_id:
                    provider_id = p_id
                    break
        
        # If verification strategy is "specific", use the specified provider
        elif verification_strategy == "specific":
            specific_provider = self.config.get("verification_provider")
            if specific_provider:
                provider_id = specific_provider
        
        potential_missing = []
        
        try:
            # Get the provider
            provider = ModelProviderRegistry.get_provider(provider_id)
            
            # Get the model
            if verification_strategy == "specific" and self.config.get("verification_model"):
                model = self.config.get("verification_model")
            else:
                model = provider.get_default_model()
            
            # Call the provider
            result = provider.verify_text(prompt=prompt, model=model)
            
            # Process the result
            if isinstance(result, list):
                potential_missing = result
            elif "missing_codes" in result:
                potential_missing = result["missing_codes"]
                
        except Exception as e:
            self.logger.error(f"Error finding potential missing requirements: {e}")
        
        return potential_missing
    
    def export_to_excel(self, requirements, validation_results, verification_results, output_path):
        """Export requirements and verification results to Excel."""
        # Create DataFrame for requirements with source info
        req_data = []
        for req in requirements:
            req_data.append({
                "Code": req["code"],
                "Description": req["description"],
                "Source Type": req.get("source_type", "unknown"),
                "Source": req.get("source", ""),
                "Confidence": verification_results["confidence_scores"].get(req["code"], 0.0),
                "Verified": req["code"] in verification_results["verified"]
            })
        df_reqs = pd.DataFrame(req_data)
        
        # Create DataFrame for validation
        val_data = []
        for val in validation_results:
            val_data.append({
                "Code": val["code"],
                "Status": val["status"],
                "Issues": ", ".join(val["issues"]) if val["issues"] else "None",
                "Similarity Score": val["similarity_score"]
            })
        df_validation = pd.DataFrame(val_data)
        
        # Create DataFrame for verification
        df_verification = pd.DataFrame({
            "Verified Codes": verification_results["verified"] + [""] * (
                max(0, len(verification_results["potential_missing"]) - len(verification_results["verified"]))
            ),
            "Potential Missing/Low Confidence": verification_results["potential_missing"] + [""] * (
                max(0, len(verification_results["verified"]) - len(verification_results["potential_missing"]))
            )
        })
        
        # Create confidence scores DataFrame
        confidence_items = list(verification_results["confidence_scores"].items())
        df_confidence = pd.DataFrame(confidence_items, columns=["Requirement Code", "Confidence Score"])
        
        # Create verification details DataFrame
        verification_details = []
        for code, details in verification_results["verification_details"].items():
            verification_details.append({
                "Code": code,
                "Verified": details.get("verified", False),
                "Confidence": details.get("confidence", 0.0),
                "Reason": details.get("reason", "")
            })
        df_verification_details = pd.DataFrame(verification_details)
        
        # Create summary sheet data
        summary_data = {
            "Total Requirements": len(requirements),
            "From Tables": sum(1 for r in requirements if r.get("source_type") == "table"),
            "From Text": sum(1 for r in requirements if r.get("source_type") == "text"),
            "Verified Requirements": len(verification_results["verified"]),
            "Potential Missing": len(verification_results["potential_missing"]),
            "Valid Requirements": sum(1 for v in validation_results if v["status"] == "valid"),
            "Requirements with Issues": sum(1 for v in validation_results if v["status"] != "valid"),
            "Average Confidence Score": sum(verification_results["confidence_scores"].values()) / 
                                       (len(verification_results["confidence_scores"]) or 1)
        }
        df_summary = pd.DataFrame([summary_data])
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(output_path) as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            df_reqs.to_excel(writer, sheet_name="Requirements", index=False)
            df_validation.to_excel(writer, sheet_name="Validation", index=False)
            df_verification.to_excel(writer, sheet_name="Verification", index=False)
            df_confidence.to_excel(writer, sheet_name="Confidence Scores", index=False)
            df_verification_details.to_excel(writer, sheet_name="Verification Details", index=False)
        
        self.logger.info(f"Exported {len(requirements)} requirements to {output_path}")
        return output_path
    
    def process_pdf(self, pdf_path, output_path=None):
        """Process a PDF to extract, verify and export requirements."""
        if output_path is None:
            # Generate output path based on input file if not provided
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = f"{base_name}_requirements.xlsx"
        
        self.logger.info(f"Processing {pdf_path}...")
        
        # Extract text and tables
        text_by_page = self.extract_text_from_pdf(pdf_path)
        full_text = "\n\n".join(text_by_page)
        
        # Analyze document structure
        chunking_method = self.analyze_pdf_structure(text_by_page)
        
        # Choose chunking method based on analysis
        if chunking_method == "semantic":
            chunks = self.semantic_chunk_document(text_by_page)
        elif chunking_method == "single":
            chunks = ["\n\n".join(text_by_page)]  # Single chunk
        else:
            # For standard chunking, adjust chunk size based on page length
            avg_page_length = sum(len(page) for page in text_by_page) / len(text_by_page)
            if avg_page_length > 3000:
                chunk_size = max(1, self.config["chunk_size"] // 2)
            else:
                chunk_size = self.config["chunk_size"]
            chunks = self.chunk_document(text_by_page, chunk_size)
        
        self.logger.info(f"Document split into {len(chunks)} chunks.")

        # Process chunks
        if self.config["parallel_processing"]:
            all_text_requirements = self.process_chunks_parallel(chunks)
        else:
            all_text_requirements = self.process_chunks_sequential(chunks)
        
        # Process tables if configured
        all_table_requirements = []
        if self.config["extract_tables"]:
            tables_by_page = self.extract_tables_from_pdf(pdf_path)
            all_table_requirements = self.process_tables(tables_by_page)
        
        # Combine all requirements
        all_requirements = all_text_requirements + all_table_requirements

        # Merge and deduplicate requirements
        unique_requirements = self.merge_and_deduplicate_requirements(all_requirements)

        self.logger.info(f"Extracted {len(unique_requirements)} unique requirements.")
        
        # Validate requirements
        validation_results = self.validate_requirements(unique_requirements, full_text)
        
        # Check if verification is enabled
        verification_enabled = self.config.get("verification_enabled", True)
        
        # For offline mode, always disable verification
        if self.config_manager.use_offline_provider():
            verification_enabled = False
            self.logger.info("Offline mode detected: Verification process disabled")
        
        # Explicitly log the verification enabled status
        self.logger.info(f"VERIFICATION ENABLED STATUS: {verification_enabled}")
        self.logger.info(f"Raw config value: {self.config.get('verification_enabled')}")
        
        if verification_enabled:
            # Verify extraction
            verification_results = self.verify_extraction(full_text, unique_requirements)
            self.logger.info(f"Verified {len(verification_results['verified'])} requirements.")
            
            if verification_results['potential_missing']:
                self.logger.warning(f"{len(verification_results['potential_missing'])} potential issues found")
        else:
            # Create a default verification result with all requirements marked as verified
            self.logger.info("Verification process is disabled, skipping verification")
            verification_results = {
                "verified": [req["code"] for req in unique_requirements],
                "potential_missing": [],
                "confidence_scores": {req["code"]: 1.0 for req in unique_requirements},
                "verification_details": {
                    req["code"]: {
                        "verified": True,
                        "confidence": 1.0,
                        "reason": "Verification process is disabled"
                    } for req in unique_requirements
                }
            }
        
        # Export results
        output_file = self.export_to_excel(
            unique_requirements, 
            validation_results, 
            verification_results, 
            output_path
        )
        
        return {
            "requirements": unique_requirements,
            "validation": validation_results,
            "verification": verification_results,
            "output_file": output_file
        }

    def batch_process(self, pdf_directory, output_directory=None):
        """Process all PDFs in a directory."""
        if output_directory is None:
            output_directory = os.path.join(pdf_directory, "requirements_output")
        
        os.makedirs(output_directory, exist_ok=True)
        
        results = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        self.logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory}")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            output_path = os.path.join(output_directory, f"{os.path.splitext(pdf_file)[0]}_requirements.xlsx")
            
            try:
                self.logger.info(f"Processing {pdf_file}...")
                result = self.process_pdf(pdf_path, output_path)
                results.append({
                    "file": pdf_file,
                    "status": "success",
                    "requirements_count": len(result["requirements"]),
                    "output_file": output_path
                })
                self.logger.info(f"Successfully processed {pdf_file}")
            except Exception as e:
                self.logger.error(f"Error processing {pdf_file}: {str(e)}")
                results.append({
                    "file": pdf_file,
                    "status": "error",
                    "error": str(e)
                })
        
        # Create summary report
        summary_df = pd.DataFrame(results)
        summary_path = os.path.join(output_directory, "batch_summary.xlsx")
        summary_df.to_excel(summary_path, index=False)
        
        self.logger.info(f"Batch processing complete. Summary saved to {summary_path}")
        return results

def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract requirements from PDF documents")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("--output", "-o", help="Output file or directory")
    parser.add_argument("--batch", "-b", action="store_true", help="Process directory of PDFs in batch mode")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--semantic-chunking", action="store_true", help="Use semantic chunking")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--no-tables", action="store_true", help="Disable table extraction")
    parser.add_argument("--no-verification", action="store_true", help="Disable verification process")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    
    # Provider selection arguments
    provider_group = parser.add_argument_group("Provider Options")
    provider_group.add_argument("--provider", choices=["openai", "anthropic", "together", "ollama"], 
                              help="Specify provider to use")
    provider_group.add_argument("--enable-anthropic", action="store_true", help="Enable Anthropic Claude as fallback")
    provider_group.add_argument("--use-offline", action="store_true", 
                              help="Use offline providers (like Ollama) if available")
    provider_group.add_argument("--enable-ollama", action="store_true", 
                              help="Enable Ollama local LLM provider")
    provider_group.add_argument("--ollama-server", default="http://localhost:11434",
                              help="Ollama server URL (default: http://localhost:11434)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "model": args.model,
        "use_semantic_chunking": args.semantic_chunking,
        "use_cache": not args.no_cache,
        "parallel_processing": not args.no_parallel,
        "extract_tables": not args.no_tables,
        "verification_enabled": not args.no_verification,
        "max_workers": args.workers,
        "enable_anthropic": args.enable_anthropic,
        "use_offline_provider": args.use_offline
    }
    
    # Handle provider selection
    if args.provider:
        config["provider"] = args.provider
        
    # Configure Ollama if enabled
    if args.enable_ollama:
        config["ollama"] = {
            "enabled": True,
            "server_url": args.ollama_server
        }
    
    try:
        extractor = RequirementsExtractor(config)
        
        if args.batch:
            # Batch processing mode
            result = extractor.batch_process(args.input, args.output)
            print(f"Successfully processed {sum(1 for r in result if r['status'] == 'success')} PDF files")
            print(f"Failed to process {sum(1 for r in result if r['status'] == 'error')} PDF files")
        else:
            # Single file processing mode
            result = extractor.process_pdf(args.input, args.output)
            print(f"Successfully processed {args.input}")
            print(f"Extracted {len(result['requirements'])} requirements")
            print(f"Output saved to {result['output_file']}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()