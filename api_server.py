"""
API Server for PDF Requirements Extractor

This module provides a REST API for the PDF Requirements Extractor application.
It allows the React frontend to interact with the extraction functionality.
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import the requirements extractor components
from config_manager import ConfigManager
from pdf_requirements_extractor import RequirementsExtractor
from provider_registry import ModelProviderRegistry, register_default_providers

# Register default providers
register_default_providers()

# Create Flask app
app = Flask(__name__, static_folder='frontend/build')
CORS(app)  # Enable CORS for all routes

# Initialize configuration manager
config_manager = ConfigManager()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Get all available providers
@app.route('/api/providers', methods=['GET'])
def get_providers():
    provider_info = ModelProviderRegistry.get_provider_info()
    return jsonify(provider_info)

# Get current configuration
@app.route('/api/config', methods=['GET'])
def get_config():
    app_config = config_manager.get_app_config()
    extraction_config = config_manager.get_extraction_config()
    
    # Get provider configurations
    provider_configs = {}
    for provider_id in ModelProviderRegistry.get_available_providers():
        provider_configs[provider_id] = config_manager.get_provider_config(provider_id)
    
    # Return consolidated configuration
    return jsonify({
        'app': app_config,
        'providers': provider_configs,
        'extraction': extraction_config
    })

# Save configuration
@app.route('/api/config', methods=['POST'])
def save_config():
    data = request.json
    
    # Update app configuration
    if 'app' in data:
        config_manager.update_app_config(data['app'])
    
    # Update provider configurations
    if 'providers' in data:
        for provider_id, provider_config in data['providers'].items():
            config_manager.update_provider_config(provider_id, provider_config)
    
    # Update extraction configuration
    if 'extraction' in data:
        config_manager.update_extraction_config(data['extraction'])
    
    # Save configuration to file
    if config_manager.save_config():
        return jsonify({'success': True, 'message': 'Configuration saved successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to save configuration'}), 500

# Check Ollama status
@app.route('/api/ollama/status', methods=['GET'])
def check_ollama_status():
    server_url = request.args.get('serverUrl', 'http://localhost:11434')
    
    # Initialize Ollama provider to check status
    try:
        ollama_provider = ModelProviderRegistry.get_provider(
            "ollama", 
            initialize=True, 
            server_url=server_url
        )
        
        if ollama_provider.is_available():
            return jsonify({
                'isRunning': True,
                'models': ollama_provider.available_models
            })
        else:
            return jsonify({'isRunning': False, 'models': []})
    except Exception as e:
        return jsonify({
            'isRunning': False,
            'error': str(e),
            'models': []
        }), 500

# Process a single PDF file
@app.route('/api/process', methods=['POST'])
def process_pdf():
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload a PDF file.'}), 400
    
    # Get configuration from request
    config_data = {}
    if 'config' in request.form:
        try:
            config_data = json.loads(request.form['config'])
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid configuration format'}), 400
    
    # Get output path from request or generate one
    output_path = None
    if 'outputPath' in request.form and request.form['outputPath'].strip():
        output_path = request.form['outputPath'].strip()
    
    # Save the file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(file_path)
    
    # Create extractor with provided configuration
    extractor = RequirementsExtractor(config_data)
    
    try:
        # Process the PDF
        if output_path is None:
            # Generate output path in the temp directory
            output_path = os.path.join(
                temp_dir,
                os.path.splitext(secure_filename(file.filename))[0] + '_requirements.xlsx'
            )
        
        # Run the extraction
        result = extractor.process_pdf(file_path, output_path)
        
        # Create response with results
        response = {
            'success': True,
            'requirementsCount': len(result['requirements']),
            'outputFile': result['output_file'],
            'validRequirements': sum(1 for v in result['validation'] if v['status'] == 'valid'),
            'verifiedRequirements': len(result['verification']['verified'])
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        try:
            os.remove(file_path)
        except:
            pass

# Process a batch of PDF files
@app.route('/api/process/batch', methods=['POST'])
def process_batch():
    # Check if directory is specified
    if 'directory' not in request.files:
        return jsonify({'error': 'No directory provided'}), 400
    
    # Get configuration from request
    config_data = {}
    if 'config' in request.form:
        try:
            config_data = json.loads(request.form['config'])
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid configuration format'}), 400
    
    # Get output directory from request or generate one
    output_dir = None
    if 'outputPath' in request.form and request.form['outputPath'].strip():
        output_dir = request.form['outputPath'].strip()
    
    # Create temporary directory for uploaded files
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, 'pdfs'), exist_ok=True)
    
    # Save all PDF files to the temporary directory
    files = request.files.getlist('directory')
    pdf_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            file_path = os.path.join(temp_dir, 'pdfs', secure_filename(file.filename))
            file.save(file_path)
            pdf_files.append(file_path)
    
    if not pdf_files:
        return jsonify({'error': 'No valid PDF files found'}), 400
    
    # Create extractor with provided configuration
    extractor = RequirementsExtractor(config_data)
    
    try:
        # Process the batch
        if output_dir is None:
            # Generate output directory in the temp directory
            output_dir = os.path.join(temp_dir, 'requirements_output')
        
        # Run the batch processing
        results = extractor.batch_process(os.path.join(temp_dir, 'pdfs'), output_dir)
        
        # Count successes and failures
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'error')
        
        # Create response with results
        response = {
            'success': True,
            'totalFiles': len(results),
            'successCount': success_count,
            'failedCount': failed_count,
            'outputDirectory': output_dir
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)