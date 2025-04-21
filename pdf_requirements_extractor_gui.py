import sys
import os
import json
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QComboBox, QRadioButton, 
                            QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QGroupBox, 
                            QTextEdit, QFileDialog, QMessageBox, QFrame, QScrollArea,
                            QSplitter, QProgressBar)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread, QMimeData, QUrl
from PyQt6.QtGui import QIcon, QFont, QPixmap, QDragEnterEvent, QDropEvent

from pdf_requirements_extractor import RequirementsExtractor
from config_manager import ConfigManager
from provider_registry import ModelProviderRegistry


class DragDropLineEdit(QLineEdit):
    """A QLineEdit that supports drag and drop for files"""
    fileDropped = pyqtSignal(str)  # Signal emitted when a file is dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.pdf') or os.path.isdir(file_path):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.pdf') or os.path.isdir(file_path):
                    self.fileDropped.emit(file_path)
                    event.acceptProposedAction()


class DropAreaWidget(QWidget):
    """A widget that supports drag and drop for files"""
    fileDropped = pyqtSignal(str)  # Signal emitted when a file is dropped
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setAcceptDrops(True)
        self.highlight = False
        
        # Styling
        self.normal_style = """
            DropAreaWidget {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 20px;
            }
        """
        self.highlight_style = """
            DropAreaWidget {
                border: 2px dashed #3498db;
                border-radius: 5px;
                background-color: #e6f3fb;
                padding: 20px;
            }
        """
        self.setStyleSheet(self.normal_style)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.pdf') or os.path.isdir(file_path):
                    self.highlight = True
                    self.setStyleSheet(self.highlight_style)
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave events"""
        self.highlight = False
        self.setStyleSheet(self.normal_style)
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events"""
        self.highlight = False
        self.setStyleSheet(self.normal_style)
        
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.pdf') or os.path.isdir(file_path):
                    self.fileDropped.emit(file_path)
                    event.acceptProposedAction()
    
    def paintEvent(self, event):
        """Custom paint event to draw drop message"""
        super().paintEvent(event)
        from PyQt6.QtGui import QPainter, QColor, QPen
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set text color
        color = QColor("#3498db") if self.highlight else QColor("#999")
        painter.setPen(QPen(color))
        
        # Draw drop text
        font = self.font()
        font.setPointSize(12)
        painter.setFont(font)
        
        text = "Drop PDF File or Folder Here"
        rect = self.rect()
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class WorkerThread(QThread):
    """Worker thread for background processing"""
    update_log = pyqtSignal(str)
    task_complete = pyqtSignal(bool, str)
    progress_update = pyqtSignal(int, int)  # current, total
    
    def __init__(self, config, input_path, output_path, processing_type):
        super().__init__()
        self.config = config
        self.input_path = input_path
        self.output_path = output_path
        self.processing_type = processing_type
        
    def run(self):
        try:
            # Initialize the extractor with our configuration
            extractor = RequirementsExtractor(self.config)
            
            # Log configuration
            self.update_log.emit(f"Using model: {self.config.get('model')}")
            self.update_log.emit(f"Verification strategy: {self.config.get('verification_strategy')}")
            
            # Process based on type
            if self.processing_type == "batch":
                # Actual batch processing
                self.update_log.emit(f"Processing directory: {self.input_path}")
                
                # Generate output directory if not provided
                output_dir = self.output_path if self.output_path else os.path.join(self.input_path, 'requirements_output')
                
                # Execute batch processing
                results = extractor.batch_process(self.input_path, output_dir)
                
                # Report results
                success_count = sum(1 for r in results if r['status'] == 'success')
                failed_count = sum(1 for r in results if r['status'] == 'error')
                
                self.update_log.emit(f"Processing complete.")
                self.update_log.emit(f"Successfully processed {success_count} files.")
                
                if failed_count > 0:
                    self.update_log.emit(f"Failed to process {failed_count} files.")
                
                self.update_log.emit(f"Results saved to: {output_dir}")
                
            else:
                # Single file processing
                self.update_log.emit(f"Processing file: {self.input_path}")
                
                # Generate output path if not provided
                if not self.output_path:
                    base_name = os.path.splitext(os.path.basename(self.input_path))[0]
                    self.output_path = f"{base_name}_requirements.xlsx"
                
                # Execute single file processing
                result = extractor.process_pdf(self.input_path, self.output_path)
                
                self.update_log.emit(f"Processing complete.")
                self.update_log.emit(f"Extracted {len(result['requirements'])} requirements.")
                self.update_log.emit(f"Output saved to: {result['output_file']}")
            
            # Signal completion
            self.task_complete.emit(True, "Processing completed successfully!")
            
        except Exception as e:
            self.update_log.emit(f"Error: {str(e)}")
            self.task_complete.emit(False, f"An error occurred: {str(e)}")


class RequirementsExtractorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF Requirements Extractor")
        self.setMinimumSize(900, 700)
        
        # Initialize variables
        self.worker_thread = None
        self.is_processing = False
        
        # Initialize configuration manager
        self.config_manager = ConfigManager()
        
        # Create the UI
        self.setup_ui()
        
        # Load configuration and apply to UI
        self.load_config()
        
    def setup_ui(self):
        """Create the main UI layout"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # Header with title
        header_layout = QHBoxLayout()
        logo_label = QLabel("Requirements Extractor")
        logo_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_layout.addWidget(logo_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)
        
        # Create a splitter for configuration and log sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        
        # Configuration area (top part)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model selection section
        self.create_model_section(config_layout)
        
        # File selection section
        self.create_file_section(config_layout)
        
        # Advanced settings button
        adv_button = QPushButton("Advanced Settings")
        adv_button.setIcon(QIcon.fromTheme("preferences-system"))
        adv_button.clicked.connect(self.show_advanced_settings)
        config_layout.addWidget(adv_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        # Add a separator line
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        config_layout.addWidget(line2)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Settings")
        save_button.setIcon(QIcon.fromTheme("document-save"))
        save_button.clicked.connect(self.save_config)
        action_layout.addWidget(save_button)
        
        reset_button = QPushButton("Reset")
        reset_button.setIcon(QIcon.fromTheme("edit-undo"))
        reset_button.clicked.connect(self.reset_config)
        action_layout.addWidget(reset_button)
        
        action_layout.addStretch()
        
        self.process_button = QPushButton("Process PDF(s)")
        self.process_button.setIcon(QIcon.fromTheme("system-run"))
        self.process_button.clicked.connect(self.process_files)
        self.process_button.setStyleSheet("QPushButton { background-color: #1E88E5; color: white; padding: 8px 12px; }")
        action_layout.addWidget(self.process_button)
        
        config_layout.addLayout(action_layout)
        
        # Add config widget to splitter
        splitter.addWidget(config_widget)
        
        # Log section (bottom part)
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        
        # Create log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { background-color: #F5F5F5; }")
        log_layout.addWidget(self.log_text)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        log_layout.addWidget(self.progress_bar)
        
        # Add log widget to splitter
        splitter.addWidget(log_group)
        
        # Set initial splitter sizes
        splitter.setSizes([500, 200])
        
        main_layout.addWidget(splitter)
        
    def create_model_section(self, parent_layout):
        """Create model selection section based on registered providers"""
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Get available providers and models
        providers_info = ModelProviderRegistry.get_provider_info()
        
        # Create model lists from all providers
        self.available_models = {}
        self.all_models = []
        
        for provider_info in providers_info:
            provider_id = provider_info["id"]
            provider_models = provider_info["models"]
            self.available_models[provider_id] = provider_models
            self.all_models.extend(provider_models)
        
        # Extraction provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        
        for provider_info in providers_info:
            self.provider_combo.addItem(provider_info["name"], provider_info["id"])
        
        self.provider_combo.currentIndexChanged.connect(self.on_provider_change)
        provider_layout.addWidget(self.provider_combo)
        provider_layout.addStretch()
        model_layout.addLayout(provider_layout)
        
        # Extraction model
        extraction_layout = QHBoxLayout()
        extraction_layout.addWidget(QLabel("Extraction Model:"))
        self.extraction_model_combo = QComboBox()
        extraction_layout.addWidget(self.extraction_model_combo)
        extraction_layout.addStretch()
        model_layout.addLayout(extraction_layout)
        
        # Verification model
        verification_layout = QHBoxLayout()
        verification_layout.addWidget(QLabel("Verification Strategy:"))
        self.verification_model_combo = QComboBox()
        self.verification_model_combo.addItems(["different", "same", "specific"])
        self.verification_model_combo.currentIndexChanged.connect(self.on_verification_strategy_change)
        verification_layout.addWidget(self.verification_model_combo)
        verification_layout.addStretch()
        model_layout.addLayout(verification_layout)
        
        # Verification provider
        self.verification_provider_layout = QHBoxLayout()
        self.verification_provider_layout.addWidget(QLabel("Verification Provider:"))
        self.verification_provider_combo = QComboBox()
        
        for provider_info in providers_info:
            self.verification_provider_combo.addItem(provider_info["name"], provider_info["id"])
        
        self.verification_provider_combo.currentIndexChanged.connect(self.on_verification_provider_change)
        self.verification_provider_layout.addWidget(self.verification_provider_combo)
        self.verification_provider_layout.addStretch()
        self.verification_provider_widget = QWidget()
        self.verification_provider_widget.setLayout(self.verification_provider_layout)
        self.verification_provider_widget.setVisible(False)
        model_layout.addWidget(self.verification_provider_widget)
        
        # Specific verification model
        self.specific_model_layout = QHBoxLayout()
        self.specific_model_layout.addWidget(QLabel("Verification Model:"))
        self.verification_model_name_combo = QComboBox()
        self.specific_model_layout.addWidget(self.verification_model_name_combo)
        self.specific_model_layout.addStretch()
        self.specific_model_widget = QWidget()
        self.specific_model_widget.setLayout(self.specific_model_layout)
        self.specific_model_widget.setVisible(False)
        model_layout.addWidget(self.specific_model_widget)
        
        parent_layout.addWidget(model_group)
        
    def create_file_section(self, parent_layout):
        """Create file selection section with drag and drop support"""
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # Create a drop area widget
        self.drop_area = DropAreaWidget(self)
        self.drop_area.fileDropped.connect(self.handle_dropped_file)
        file_layout.addWidget(self.drop_area)
        
        # Input file/directory
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input PDF:"))
        self.input_path_edit = DragDropLineEdit()
        self.input_path_edit.setPlaceholderText("Drag and drop a PDF file here or use Browse button →")
        self.input_path_edit.fileDropped.connect(self.handle_dropped_file)
        input_layout.addWidget(self.input_path_edit)
        input_browse_button = QPushButton("Browse")
        input_browse_button.clicked.connect(self.browse_input)
        input_layout.addWidget(input_browse_button)
        file_layout.addLayout(input_layout)
        
        # Output file/directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Location:"))
        self.output_path_edit = QLineEdit()
        output_layout.addWidget(self.output_path_edit)
        output_browse_button = QPushButton("Browse")
        output_browse_button.clicked.connect(self.browse_output)
        output_layout.addWidget(output_browse_button)
        file_layout.addLayout(output_layout)
        
        # Processing type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Processing Type:"))
        
        self.single_radio = QRadioButton("Single File")
        self.single_radio.setChecked(True)
        self.batch_radio = QRadioButton("Batch Directory")
        self.single_radio.toggled.connect(self.on_processing_type_change)
        
        type_layout.addWidget(self.single_radio)
        type_layout.addWidget(self.batch_radio)
        type_layout.addStretch()
        file_layout.addLayout(type_layout)
        
        parent_layout.addWidget(file_group)
    
    def on_provider_change(self):
        """Handle change in AI provider"""
        provider_id = self.provider_combo.currentData()
        
        # Update models for this provider
        self.extraction_model_combo.clear()
        if provider_id in self.available_models:
            self.extraction_model_combo.addItems(self.available_models[provider_id])
            
        # Set default model
        provider_config = self.config_manager.get_provider_config(provider_id)
        default_model = provider_config.get("default_model", "")
        
        if default_model and self.extraction_model_combo.findText(default_model) >= 0:
            self.extraction_model_combo.setCurrentText(default_model)
    
    def on_verification_provider_change(self):
        """Handle change in verification provider"""
        provider_id = self.verification_provider_combo.currentData()
        
        # Update models for this provider
        self.verification_model_name_combo.clear()
        if provider_id in self.available_models:
            self.verification_model_name_combo.addItems(self.available_models[provider_id])
            
        # Set default model
        provider_config = self.config_manager.get_provider_config(provider_id)
        default_model = provider_config.get("default_model", "")
        
        if default_model and self.verification_model_name_combo.findText(default_model) >= 0:
            self.verification_model_name_combo.setCurrentText(default_model)
    
    def on_verification_strategy_change(self):
        """Handle change in verification strategy"""
        strategy = self.verification_model_combo.currentText()
        self.verification_provider_widget.setVisible(strategy == "specific")
        self.specific_model_widget.setVisible(strategy == "specific")
    
    def on_processing_type_change(self):
        """Handle change in processing type"""
        is_batch = self.batch_radio.isChecked()
        
        # Update placeholder and dialog selection based on type
        if is_batch:
            self.input_path_edit.setPlaceholderText("Directory containing PDF files")
            self.output_path_edit.setPlaceholderText("Output directory for extracted requirements")
        else:
            self.input_path_edit.setPlaceholderText("Path to PDF file")
            self.output_path_edit.setPlaceholderText("Output Excel file for extracted requirements")
    
    def handle_dropped_file(self, file_path):
        """Handle the dropped file"""
        # Set the input path
        self.input_path_edit.setText(file_path)
        
        # If it's a PDF file, switch to single file mode
        if file_path.lower().endswith('.pdf'):
            self.single_radio.setChecked(True)
        # If it's a directory, switch to batch mode
        elif os.path.isdir(file_path):
            self.batch_radio.setChecked(True)
        
        # Generate suggested output path
        if file_path.lower().endswith('.pdf'):
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            suggested_output = f"{base_name}_requirements.xlsx"
            self.output_path_edit.setText(suggested_output)
    
    def show_advanced_settings(self):
        """Show advanced settings dialog based on app configuration"""
        # Get current app configuration
        app_config = self.config_manager.get_app_config()
        
        # Create a proper dialog window
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        
        settings_dialog = QDialog(self)
        settings_dialog.setWindowTitle("Advanced Settings")
        settings_dialog.setMinimumSize(600, 450)
        settings_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        
        # Main vertical layout for the dialog
        main_layout = QVBoxLayout(settings_dialog)
        
        # Create tab widget and add it to the dialog
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Processing tab
        processing_tab = QWidget()
        processing_layout = QVBoxLayout(processing_tab)
        
        use_cache_check = QCheckBox("Use Cache")
        use_cache_check.setChecked(app_config.get("use_cache", True))
        processing_layout.addWidget(use_cache_check)
        
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("Cache Directory:"))
        cache_dir_edit = QLineEdit(app_config.get("cache_dir", ".requirement_cache"))
        cache_layout.addWidget(cache_dir_edit)
        processing_layout.addLayout(cache_layout)
        
        extract_tables_check = QCheckBox("Extract Tables")
        extract_tables_check.setChecked(app_config.get("extract_tables", True))
        processing_layout.addWidget(extract_tables_check)
        
        parallel_check = QCheckBox("Parallel Processing")
        parallel_check.setChecked(app_config.get("parallel_processing", True))
        processing_layout.addWidget(parallel_check)
        
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Max Workers:"))
        workers_spin = QSpinBox()
        workers_spin.setRange(1, 10)
        workers_spin.setValue(app_config.get("max_workers", 3))
        workers_layout.addWidget(workers_spin)
        workers_layout.addStretch()
        processing_layout.addLayout(workers_layout)
        
        processing_layout.addStretch()
        tabs.addTab(processing_tab, "Processing")
        
        # Document tab
        document_tab = QWidget()
        document_layout = QVBoxLayout(document_tab)
        
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(QLabel("Chunk Size (pages):"))
        chunk_spin = QSpinBox()
        chunk_spin.setRange(1, 10)
        chunk_spin.setValue(app_config.get("chunk_size", 3))
        chunk_layout.addWidget(chunk_spin)
        chunk_layout.addStretch()
        document_layout.addLayout(chunk_layout)
        
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Max Token Size:"))
        token_spin = QSpinBox()
        token_spin.setRange(1000, 8000)
        token_spin.setSingleStep(100)
        token_spin.setValue(app_config.get("max_token_size", 4000))
        token_layout.addWidget(token_spin)
        token_layout.addStretch()
        document_layout.addLayout(token_layout)
        
        semantic_check = QCheckBox("Use Semantic Similarity")
        semantic_check.setChecked(app_config.get("use_semantic_similarity", False))
        document_layout.addWidget(semantic_check)
        
        document_layout.addStretch()
        tabs.addTab(document_tab, "Document")
        
        # Verification Models tab
        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)
        
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        confidence_spin = QDoubleSpinBox()
        confidence_spin.setRange(0.1, 1.0)
        confidence_spin.setSingleStep(0.1)
        confidence_spin.setValue(app_config.get("confidence_threshold", 0.8))
        confidence_layout.addWidget(confidence_spin)
        confidence_layout.addStretch()
        models_layout.addLayout(confidence_layout)
        
        retry_layout = QHBoxLayout()
        retry_layout.addWidget(QLabel("Retry Attempts:"))
        retry_spin = QSpinBox()
        retry_spin.setRange(1, 5)
        retry_spin.setValue(app_config.get("retry_attempts", 3))
        retry_layout.addWidget(retry_spin)
        retry_layout.addStretch()
        models_layout.addLayout(retry_layout)
        
        models_layout.addStretch()
        tabs.addTab(models_tab, "Verification Model")
        
        # Learning tab
        learning_tab = QWidget()
        learning_layout = QVBoxLayout(learning_tab)
        
        adaptive_check = QCheckBox("Adaptive Learning")
        adaptive_check.setChecked(app_config.get("adaptive_learning", True))
        learning_layout.addWidget(adaptive_check)
        
        patterns_layout = QHBoxLayout()
        patterns_layout.addWidget(QLabel("Patterns File:"))
        patterns_edit = QLineEdit(app_config.get("patterns_file", "requirement_patterns.json"))
        patterns_layout.addWidget(patterns_edit)
        learning_layout.addLayout(patterns_layout)
        
        learning_layout.addStretch()
        tabs.addTab(learning_tab, "Learning")
        
        # Add button box (OK/Cancel)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(settings_dialog.accept)
        button_box.rejected.connect(settings_dialog.reject)
        main_layout.addWidget(button_box)
        
        # Show dialog and process results
        if settings_dialog.exec():
            # Dialog was accepted, update app configuration
            updated_config = {
                "use_cache": use_cache_check.isChecked(),
                "cache_dir": cache_dir_edit.text(),
                "extract_tables": extract_tables_check.isChecked(),
                "parallel_processing": parallel_check.isChecked(),
                "max_workers": workers_spin.value(),
                "chunk_size": chunk_spin.value(),
                "max_token_size": token_spin.value(),
                "use_semantic_similarity": semantic_check.isChecked(),
                "confidence_threshold": confidence_spin.value(),
                "retry_attempts": retry_spin.value(),
                "adaptive_learning": adaptive_check.isChecked(),
                "patterns_file": patterns_edit.text()
            }
            
            # Update the configuration
            self.config_manager.update_app_config(updated_config)
    
    def browse_input(self):
        """Open file dialog to select input"""
        if self.batch_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select Directory with PDF files")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select PDF file", 
                                                filter="PDF files (*.pdf);;All files (*.*)")
        
        if path:
            self.input_path_edit.setText(path)
    
    def browse_output(self):
        """Open file dialog to select output location"""
        if self.batch_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save Output As", 
                                                filter="Excel files (*.xlsx)")
        
        if path:
            self.output_path_edit.setText(path)
    
    def log(self, message):
        """Add message to log window"""
        self.log_text.append(message)
    
    def process_files(self):
        """Process files according to settings"""
        if self.is_processing:
            QMessageBox.information(self, "Processing", "A task is already running. Please wait.")
            return
        
        # Validate inputs
        input_path = self.input_path_edit.text()
        if not input_path:
            QMessageBox.critical(self, "Error", "Please select an input file or directory.")
            return
        
        if not os.path.exists(input_path):
            QMessageBox.critical(self, "Error", "Input path does not exist.")
            return
        
        # Get current provider configuration
        provider_id = self.provider_combo.currentData()
        model = self.extraction_model_combo.currentText()
        
        # Create extraction config
        extraction_config = {
            "provider": provider_id,
            "model": model,
            "verification_strategy": self.verification_model_combo.currentText()
        }
        
        # Set verification provider/model if needed
        if self.verification_model_combo.currentText() == "specific":
            extraction_config["verification_provider"] = self.verification_provider_combo.currentData()
            extraction_config["verification_model"] = self.verification_model_name_combo.currentText()
        
        # Get app config
        app_config = self.config_manager.get_app_config()
        
        # Combine configs
        config = app_config.copy()
        config.update(extraction_config)
        
        # Start processing in a separate thread
        self.is_processing = True
        self.process_button.setEnabled(False)
        
        # Clear log
        self.log_text.clear()
        self.log("Starting processing...")
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Start worker thread
        self.worker_thread = WorkerThread(
            config=config,
            input_path=input_path,
            output_path=self.output_path_edit.text(),
            processing_type="batch" if self.batch_radio.isChecked() else "single"
        )
        
        self.worker_thread.update_log.connect(self.log)
        self.worker_thread.task_complete.connect(self.on_processing_complete)
        self.worker_thread.start()
    
    def on_processing_complete(self, success, message):
        """Handle completion of processing task"""
        self.is_processing = False
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Complete", message)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def save_config(self):
        """Save current settings to configuration"""
        # Update extraction config
        extraction_config = {
            "provider": self.provider_combo.currentData(),
            "model": self.extraction_model_combo.currentText(),
            "verification_strategy": self.verification_model_combo.currentText()
        }
        
        # Set verification provider/model if needed
        if self.verification_model_combo.currentText() == "specific":
            extraction_config["verification_provider"] = self.verification_provider_combo.currentData()
            extraction_config["verification_model"] = self.verification_model_name_combo.currentText()
        
        # Update extraction configuration
        self.config_manager.update_extraction_config(extraction_config)
        
        # Save to file
        if self.config_manager.save_config():
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        else:
            QMessageBox.critical(self, "Error", "Failed to save settings")
    
    def load_config(self):
        """Load settings from configuration"""
        # Load extraction config
        extraction_config = self.config_manager.get_extraction_config()
        
        # Set provider
        provider_id = extraction_config.get("provider", "openai")
        provider_index = self.provider_combo.findData(provider_id)
        if provider_index >= 0:
            self.provider_combo.setCurrentIndex(provider_index)
        
        # Set model
        model = extraction_config.get("model")
        if model and self.extraction_model_combo.findText(model) >= 0:
            self.extraction_model_combo.setCurrentText(model)
        
        # Set verification strategy
        strategy = extraction_config.get("verification_strategy", "different")
        self.verification_model_combo.setCurrentText(strategy)
        
        # Set verification provider/model if specific strategy
        if strategy == "specific":
            verification_provider = extraction_config.get("verification_provider")
            if verification_provider:
                provider_index = self.verification_provider_combo.findData(verification_provider)
                if provider_index >= 0:
                    self.verification_provider_combo.setCurrentIndex(provider_index)
            
            verification_model = extraction_config.get("verification_model")
            if verification_model and self.verification_model_name_combo.findText(verification_model) >= 0:
                self.verification_model_name_combo.setCurrentText(verification_model)
        
        # Update UI based on configuration
        self.on_verification_strategy_change()
    
    def reset_config(self):
        """Reset all settings to defaults"""
        confirm = QMessageBox.question(self, "Reset", "Reset all settings to default values?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Reset config manager to defaults
            self.config_manager = ConfigManager()
            
            # Reload UI with defaults
            self.load_config()
            
            QMessageBox.information(self, "Reset", "Settings have been reset to defaults.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Modern style
    window = RequirementsExtractorGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()