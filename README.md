# PDF Requirements Extractor

## Overview
The **PDF Requirements Extractor** is a tool designed to extract requirements from PDF documents, particularly from software requirement specifications (SRS) and similar technical documents. The tool utilizes **AI-powered natural language processing (NLP)** and **table extraction** techniques to identify structured requirements efficiently.

## Features
- Extracts textual and tabular requirements from PDF documents.
- Uses **OpenAI GPT models** for enhanced accuracy in extracting requirements.
- Supports **semantic chunking** to improve contextual understanding.
- **Parallel processing** for improved performance.
- **Adaptive learning**: learns patterns over time for better extractions.
- Verifies extracted requirements using **a separate AI model** to reduce bias.
- Supports **batch processing** for multiple PDFs.
- Exports results to **Excel (.xlsx)** for easy review and further processing.

## Why Use This Project?
- Automates **requirement extraction**, reducing manual effort.
- Enhances accuracy through **AI-powered verification**.
- Supports **scalability** by handling multiple documents efficiently.
- Provides structured outputs that can be easily reviewed and processed further.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- Required Python packages (install using the command below)

### Setup
Clone the repository:
```sh
$ git clone https://github.com/your-repo/pdf-requirements-extractor.git
$ cd pdf-requirements-extractor
```
Install dependencies:
```sh
$ pip install -r requirements.txt
```
Set up API keys (create a `.env` file in the project directory):
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
```

## Usage
### Extract requirements from a single PDF:
```sh
$ python pdf_requirements_extractor.py input.pdf -o output.xlsx
```
### Batch process multiple PDFs in a directory:
```sh
$ python pdf_requirements_extractor.py input_directory -b -o output_directory
```
### Additional Options:
- `--semantic-chunking`: Use smarter chunking for structured PDFs.
- `--no-cache`: Disable caching to force re-processing.
- `--no-parallel`: Disable parallel processing for sequential execution.
- `--no-tables`: Ignore table-based requirements extraction.
- `--workers X`: Set the number of parallel workers.

## Output Format
The extracted requirements are saved in an Excel file with multiple sheets:
- **Summary**: Overview of extracted requirements.
- **Requirements**: List of all extracted requirements.
- **Validation**: Validation results (formatting, completeness checks).
- **Verification**: AI-based verification results.
- **Confidence Scores**: Confidence scores for each requirement.

## Contributing
1. Fork the repository.
2. Create a new branch (`feature-new-extraction`).
3. Commit your changes.
4. Push to the branch and create a Pull Request.

## License
This project is licensed under the MIT License.

---
For any issues or feature requests, feel free to open an issue on GitHub!

