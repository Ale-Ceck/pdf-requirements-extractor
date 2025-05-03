import sys
import os
import json
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QComboBox, QRadioButton, 
                            QCheckBox, QSpinBox, QDoubleSpinBox, QTabWidget, QGroupBox, 
                            QTextEdit, QFileDialog, QMessageBox, QFrame, QScrollArea,
                            QSplitter, QProgressBar, QStackedWidget)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread, QMimeData, QUrl
from PyQt6.QtGui import QIcon, QFont, QPixmap, QDragEnterEvent, QDropEvent, QPalette, QColor

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
        self.setMinimumHeight(80)
        self.setAcceptDrops(True)
        self.highlight = False
        
        # Get GUI color palette if available
        if hasattr(parent, "colors"):
            self.colors = parent.colors
        else:
            # Default colors if parent doesn't provide a palette
            self.colors = {
                "primary": "#1976D2",
                "border": "#757575",
                "background_panel": "#F5F5F5",
                "highlight": "#E1F5FE"
            }
        
        # Styling
        self.normal_style = f"""
            DropAreaWidget {{
                border: 2px dashed {self.colors["border"]};
                border-radius: 8px;
                background-color: {self.colors["background_panel"]};
                padding: 24px;
            }}
        """
        self.highlight_style = f"""
            DropAreaWidget {{
                border: 2px dashed {self.colors["primary"]};
                border-radius: 8px;
                background-color: {self.colors["highlight"]};
                padding: 24px;
            }}
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
        
        # Set text color with better contrast
        color = QColor(self.colors["primary"]) if self.highlight else QColor(self.colors["text_secondary"] if "text_secondary" in self.colors else "#424242")
        painter.setPen(QPen(color))
        
        # Draw drop text
        font = self.font()
        font.setPointSize(13)
        font.setBold(True)
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
        self.operation_mode = "online"  # Either "online" or "offline"
        
        # Define color palette
        self.colors = {
            "primary": "#1976D2",       # Main action color (blue)
            "primary_light": "#BBDEFB", # Light variant
            "secondary": "#4CAF50",     # Success/confirmation color (green)
            "warning": "#FF9800",       # Warning color (orange)
            "error": "#f44336",         # Error color (red)
            "text_primary": "#212121",  # Main text (dark gray)
            "text_secondary": "#757575",# Secondary text (medium gray)
            "background": "#FFFFFF",    # Main background (white)
            "background_alt": "#F5F5F5",# Alternative background (light gray)
            "background_panel": "#FAFAFA", # Panel background
            "border": "#DDDDDD",        # Border color (light gray)
            "highlight": "#E1F5FE",     # Highlight color (very light blue)
        }
        
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
        
        # Operation mode selector
        mode_frame = QFrame()
        mode_frame.setFrameShape(QFrame.Shape.StyledPanel)
        mode_frame.setStyleSheet(f"QFrame {{ background-color: {self.colors['background_alt']}; border-radius: 8px; padding: 12px; }}")
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setSpacing(15)
        
        mode_label = QLabel("Operation Mode:")
        mode_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        mode_layout.addWidget(mode_label)
        
        # Online mode button
        self.online_mode_btn = QPushButton("Online (API)")
        self.online_mode_btn.setCheckable(True)
        self.online_mode_btn.setChecked(True)
        self.online_mode_btn.setMinimumWidth(160)
        self.online_mode_btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {self.colors["background_panel"]}; 
                color: {self.colors["text_primary"]}; 
                padding: 10px; 
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:checked {{ 
                background-color: {self.colors["primary"]}; 
                color: white; 
            }}
            QPushButton:hover:!checked {{ 
                background-color: {self.colors["primary"]}; 
            }}
        """)
        self.online_mode_btn.clicked.connect(lambda: self.switch_operation_mode("online"))
        mode_layout.addWidget(self.online_mode_btn)
        
        # Offline mode button
        self.offline_mode_btn = QPushButton("Offline (Local)")
        self.offline_mode_btn.setCheckable(True)
        self.offline_mode_btn.setMinimumWidth(160)
        self.offline_mode_btn.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {self.colors["background_panel"]}; 
                color: {self.colors["text_primary"]}; 
                padding: 10px; 
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:checked {{ 
                background-color: {self.colors["primary"]}; 
                color: white; 
            }}
            QPushButton:hover:!checked {{ 
                background-color: #E0E0E0; 
            }}
        """)
        self.offline_mode_btn.clicked.connect(lambda: self.switch_operation_mode("offline"))
        mode_layout.addWidget(self.offline_mode_btn)
        
        # Security notice with improved styling
        self.security_label = QLabel("Warning: Online mode sends data to external API services")
        self.security_label.setStyleSheet(f"QLabel {{ color: {self.colors['primary_light']}; font-weight: bold; padding: 4px; }}")
        mode_layout.addWidget(self.security_label)
        
        mode_layout.addStretch()
        main_layout.addWidget(mode_frame)
        
        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(line)
        
        # Create a splitter for configuration and log sections with improved appearance
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)  # Wider handle for easier grabbing
        
        # Configuration area (top part)
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 8)  # Add bottom margin
        
        # Create stacked widget for online/offline modes
        self.mode_stack = QStackedWidget()
        
        # Online mode settings page
        self.online_page = QWidget()
        online_layout = QVBoxLayout(self.online_page)
        online_layout.setContentsMargins(0, 0, 0, 0)
        
        # Online providers warning
        online_warning = QFrame()
        online_warning.setStyleSheet(f"""
            QFrame {{ 
                background-color: {self.colors["warning"]}30; 
                border: 1px solid {self.colors["warning"]}; 
                border-radius: 8px; 
                padding: 12px; 
            }}
        """)
        online_warning_layout = QHBoxLayout(online_warning)
        online_warning_layout.setSpacing(10)
        
        warning_icon = QLabel("⚠️")
        warning_icon.setFont(QFont("Arial", 14))
        online_warning_layout.addWidget(warning_icon)
        
        warning_text = QLabel("In Online mode, your content will be sent to external API servers via the internet")
        warning_text.setWordWrap(True)
        warning_text.setStyleSheet(f"QLabel {{ color: {self.colors['text_primary']}; font-weight: bold; }}")
        online_warning_layout.addWidget(warning_text)
        
        online_layout.addWidget(online_warning)
        
        # Online model selection
        self.create_online_model_section(online_layout)
        
        # Offline mode settings page
        self.offline_page = QWidget()
        offline_layout = QVBoxLayout(self.offline_page)
        offline_layout.setContentsMargins(0, 0, 0, 0)
        
        # Offline security notice
        offline_notice = QFrame()
        offline_notice.setStyleSheet(f"""
            QFrame {{ 
                background-color: {self.colors["secondary"]}20; 
                border: 1px solid {self.colors["secondary"]}; 
                border-radius: 8px; 
                padding: 12px; 
            }}
        """)
        offline_notice_layout = QHBoxLayout(offline_notice)
        offline_notice_layout.setSpacing(10)
        
        notice_icon = QLabel("🛡️")
        notice_icon.setFont(QFont("Arial", 14))
        offline_notice_layout.addWidget(notice_icon)
        
        notice_text = QLabel("Offline mode: All processing happens locally. No data is sent over the internet.")
        notice_text.setWordWrap(True)
        notice_text.setStyleSheet(f"QLabel {{ color: {self.colors['text_primary']}; font-weight: bold; }}")
        offline_notice_layout.addWidget(notice_text)
        
        offline_layout.addWidget(offline_notice)
        
        # Offline model selection
        self.create_offline_model_section(offline_layout)
        
        # Add pages to stack
        self.mode_stack.addWidget(self.online_page)
        self.mode_stack.addWidget(self.offline_page)
        
        # Add stack to config layout
        config_layout.addWidget(self.mode_stack)
        
        # File selection section (common to both modes)
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
        self.process_button.setMinimumWidth(180)
        self.process_button.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {self.colors["primary"]}; 
                color: white; 
                padding: 10px 16px; 
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{ 
                background-color: #1565C0; 
            }}
            QPushButton:disabled {{ 
                background-color: {self.colors["primary_light"]}; 
                color: #78909C;
            }}
        """)
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
        self.log_text.setStyleSheet(f"""
            QTextEdit {{ 
                background-color: {self.colors["background_alt"]}; 
                color: {self.colors["text_primary"]}; 
                border: 1px solid {self.colors["border"]}; 
                border-radius: 4px;
                font-family: monospace;
                padding: 8px;
            }}
        """)
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
        
    def create_online_model_section(self, parent_layout):
        """Create model selection section for online providers"""
        model_group = QGroupBox("Online Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Get available online providers and models
        providers_info = ModelProviderRegistry.get_provider_info()
        
        # Create model lists from online providers
        self.online_providers = []
        self.online_provider_models = {}
        
        for provider_info in providers_info:
            provider_id = provider_info["id"]
            provider_config = self.config_manager.get_provider_config(provider_id)
            
            # Only include online providers
            if provider_config.get("provider_type") == "online":
                self.online_providers.append(provider_info)
                self.online_provider_models[provider_id] = provider_info["models"]
        
        # Online provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Online Provider:"))
        self.online_provider_combo = QComboBox()
        
        for provider_info in self.online_providers:
            self.online_provider_combo.addItem(provider_info["name"], provider_info["id"])
        
        self.online_provider_combo.currentIndexChanged.connect(self.on_online_provider_change)
        provider_layout.addWidget(self.online_provider_combo)
        provider_layout.addStretch()
        model_layout.addLayout(provider_layout)
        
        # API key input
        apikey_layout = QHBoxLayout()
        apikey_layout.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter your API key or it will be loaded from environment")
        apikey_layout.addWidget(self.api_key_edit)
        model_layout.addLayout(apikey_layout)
        
        # Extraction model
        extraction_layout = QHBoxLayout()
        extraction_layout.addWidget(QLabel("Extraction Model:"))
        self.online_model_combo = QComboBox()
        extraction_layout.addWidget(self.online_model_combo)
        extraction_layout.addStretch()
        model_layout.addLayout(extraction_layout)
        
        # Verification
        verification_frame = QGroupBox("Verification Settings")
        verification_frame.setStyleSheet("QGroupBox { margin-top: 15px; }")
        verification_layout = QVBoxLayout(verification_frame)
        
        # Verification strategy
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Verification Strategy:"))
        self.online_verification_strategy_combo = QComboBox()
        self.online_verification_strategy_combo.addItems(["different", "same", "specific"])
        self.online_verification_strategy_combo.currentIndexChanged.connect(self.on_online_verification_strategy_change)
        strategy_layout.addWidget(self.online_verification_strategy_combo)
        strategy_layout.addStretch()
        verification_layout.addLayout(strategy_layout)
        
        # Verification provider (only shown when strategy is "specific")
        self.online_verification_provider_layout = QHBoxLayout()
        self.online_verification_provider_layout.addWidget(QLabel("Verification Provider:"))
        self.online_verification_provider_combo = QComboBox()
        
        for provider_info in self.online_providers:
            self.online_verification_provider_combo.addItem(provider_info["name"], provider_info["id"])
        
        self.online_verification_provider_combo.currentIndexChanged.connect(self.on_online_verification_provider_change)
        self.online_verification_provider_layout.addWidget(self.online_verification_provider_combo)
        self.online_verification_provider_layout.addStretch()
        
        self.online_verification_provider_widget = QWidget()
        self.online_verification_provider_widget.setLayout(self.online_verification_provider_layout)
        self.online_verification_provider_widget.setVisible(False)
        verification_layout.addWidget(self.online_verification_provider_widget)
        
        # Verification model (only shown when strategy is "specific")
        self.online_verification_model_layout = QHBoxLayout()
        self.online_verification_model_layout.addWidget(QLabel("Verification Model:"))
        self.online_verification_model_combo = QComboBox()
        self.online_verification_model_layout.addWidget(self.online_verification_model_combo)
        self.online_verification_model_layout.addStretch()
        
        self.online_verification_model_widget = QWidget()
        self.online_verification_model_widget.setLayout(self.online_verification_model_layout)
        self.online_verification_model_widget.setVisible(False)
        verification_layout.addWidget(self.online_verification_model_widget)
        
        model_layout.addWidget(verification_frame)
        parent_layout.addWidget(model_group)
    
    def create_offline_model_section(self, parent_layout):
        """Create model selection section for offline (local) providers"""
        model_group = QGroupBox("Local Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Offline provider status
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.Shape.StyledPanel)
        status_frame.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 5px; padding: 10px; }")
        status_layout = QVBoxLayout(status_frame)
        
        # Ollama status
        self.ollama_status_layout = QHBoxLayout()
        self.ollama_status_label = QLabel("Ollama Service:")
        
        self.ollama_status_value = QLabel("Not checked")
        self.ollama_status_value.setStyleSheet("QLabel { color: gray; }")
        
        self.ollama_status_layout.addWidget(self.ollama_status_label)
        self.ollama_status_layout.addWidget(self.ollama_status_value)
        
        self.ollama_check_button = QPushButton("Check Status")
        self.ollama_check_button.clicked.connect(self.check_ollama_status)
        self.ollama_check_button.setToolTip("Check if Ollama is running and available")
        self.ollama_status_layout.addWidget(self.ollama_check_button)
        
        self.ollama_status_layout.addStretch()
        status_layout.addLayout(self.ollama_status_layout)
        
        # Server URL
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Ollama Server URL:"))
        
        self.ollama_server_edit = QLineEdit("http://localhost:11434")
        server_layout.addWidget(self.ollama_server_edit)
        
        status_layout.addLayout(server_layout)
        model_layout.addWidget(status_frame)
        
        # Model settings
        model_frame = QFrame()
        model_frame.setFrameShape(QFrame.Shape.StyledPanel)
        model_frame.setStyleSheet("QFrame { background-color: #f8f8f8; border-radius: 5px; padding: 10px; }")
        model_settings_layout = QVBoxLayout(model_frame)
        
        # Model selection
        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Available Models:"))
        
        self.offline_model_combo = QComboBox()
        self.offline_model_combo.setMinimumWidth(250)
        model_selection_layout.addWidget(self.offline_model_combo)
        
        self.refresh_models_button = QPushButton("Refresh")
        self.refresh_models_button.setToolTip("Refresh available models")
        self.refresh_models_button.clicked.connect(self.refresh_offline_models)
        model_selection_layout.addWidget(self.refresh_models_button)
        
        model_selection_layout.addStretch()
        model_settings_layout.addLayout(model_selection_layout)
        
        # Model info
        self.model_info_label = QLabel("No model selected")
        self.model_info_label.setStyleSheet("QLabel { color: gray; }")
        self.model_info_label.setWordWrap(True)
        model_settings_layout.addWidget(self.model_info_label)
        
        model_layout.addWidget(model_frame)
        
        parent_layout.addWidget(model_group)
    
    def switch_operation_mode(self, mode):
        """Switch between online and offline operation modes"""
        if mode == self.operation_mode:
            return
            
        if mode == "online" and self.operation_mode == "offline":
            # Switching from offline to online mode - warn user about data sharing
            confirm = QMessageBox.warning(
                self,
                "Switch to Online Mode",
                "Switching to Online mode will send your data to external API services. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if confirm != QMessageBox.StandardButton.Yes:
                # Reset the button state
                self.offline_mode_btn.setChecked(True)
                self.online_mode_btn.setChecked(False)
                return
        
        # Update mode and UI
        self.operation_mode = mode
        
        # Update button states
        self.online_mode_btn.setChecked(mode == "online")
        self.offline_mode_btn.setChecked(mode == "offline")
        
        # Update security label text
        if mode == "online":
            self.security_label.setText("Warning: Online mode sends data to external API services")
            self.security_label.setStyleSheet(f"QLabel {{ color: {self.colors['error']}; font-weight: bold; padding: 4px; }}")
            self.mode_stack.setCurrentIndex(0)  # Show online page
        else:
            self.security_label.setText("Offline mode: All processing remains local")
            self.security_label.setStyleSheet(f"QLabel {{ color: {self.colors['secondary']}; font-weight: bold; padding: 4px; }}")
            self.mode_stack.setCurrentIndex(1)  # Show offline page
            
            # Check Ollama status when switching to offline mode
            self.check_ollama_status()
        
        # Update config
        self.config_manager.update_app_config({"use_offline_provider": mode == "offline"})
    
    def on_online_provider_change(self):
        """Handle change in online provider selection"""
        provider_id = self.online_provider_combo.currentData()
        if not provider_id:
            return
            
        # Update models for this provider
        self.online_model_combo.clear()
        if provider_id in self.online_provider_models:
            self.online_model_combo.addItems(self.online_provider_models[provider_id])
            
        # Set default model based on config
        provider_config = self.config_manager.get_provider_config(provider_id)
        default_model = provider_config.get("default_model", "")
        
        if default_model and self.online_model_combo.findText(default_model) >= 0:
            self.online_model_combo.setCurrentText(default_model)
    
    def on_online_verification_strategy_change(self):
        """Handle change in online verification strategy"""
        strategy = self.online_verification_strategy_combo.currentText()
        self.online_verification_provider_widget.setVisible(strategy == "specific")
        self.online_verification_model_widget.setVisible(strategy == "specific")
    
    def on_online_verification_provider_change(self):
        """Handle change in online verification provider"""
        provider_id = self.online_verification_provider_combo.currentData()
        if not provider_id:
            return
            
        # Update models for this provider
        self.online_verification_model_combo.clear()
        if provider_id in self.online_provider_models:
            self.online_verification_model_combo.addItems(self.online_provider_models[provider_id])
            
        # Set default model
        provider_config = self.config_manager.get_provider_config(provider_id)
        default_model = provider_config.get("default_model", "")
        
        if default_model and self.online_verification_model_combo.findText(default_model) >= 0:
            self.online_verification_model_combo.setCurrentText(default_model)
    
    def check_ollama_status(self):
        """Check if Ollama is running and update status"""
        try:
            import requests
            
            server_url = self.ollama_server_edit.text().strip()
            if not server_url:
                server_url = "http://localhost:11434"
                
            self.ollama_status_value.setText("Checking...")
            self.ollama_status_value.setStyleSheet("QLabel { color: gray; }")
            QApplication.processEvents()  # Allow UI to update
            
            try:
                response = requests.get(f"{server_url}/api/tags", timeout=2)
                
                if response.status_code == 200:
                    models_data = response.json().get("models", [])
                    model_count = len(models_data)
                    
                    self.ollama_status_value.setText(f"Running - {model_count} models available")
                    self.ollama_status_value.setStyleSheet("QLabel { color: green; }")
                    
                    # Update models in combo box
                    self.offline_model_combo.clear()
                    for model in models_data:
                        if "name" in model:
                            self.offline_model_combo.addItem(model["name"])
                            
                    # Enable model configuration
                    self.offline_model_combo.setEnabled(True)
                    
                    # Update config
                    ollama_config = {
                        "enabled": True,
                        "server_url": server_url
                    }
                    self.config_manager.update_provider_config("ollama", ollama_config)
                    
                else:
                    self.ollama_status_value.setText(f"Error: {response.status_code}")
                    self.ollama_status_value.setStyleSheet("QLabel { color: red; }")
                    self.offline_model_combo.setEnabled(False)
                    
            except requests.exceptions.RequestException as e:
                self.ollama_status_value.setText("Not running")
                self.ollama_status_value.setStyleSheet("QLabel { color: red; }")
                self.offline_model_combo.clear()
                self.offline_model_combo.setEnabled(False)
                
                # Show help if Ollama isn't running
                QMessageBox.information(self, 
                    "Ollama Not Running", 
                    "Ollama doesn't seem to be running. Please start Ollama and try again.\n\n"
                    "If you don't have Ollama installed, visit: https://ollama.com/download"
                )
                
        except ImportError:
            QMessageBox.warning(self, "Missing Dependencies", 
                "The 'requests' package is required for Ollama integration.\n"
                "Please install it with 'pip install requests'")
    
    def refresh_offline_models(self):
        """Refresh the list of available offline models"""
        self.check_ollama_status()
        
    def create_file_section(self, parent_layout):
        """Create file selection section with drag and drop support"""
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # Create a drop area widget
        self.drop_area = DropAreaWidget(self)
        self.drop_area.fileDropped.connect(self.handle_dropped_file)
        file_layout.addWidget(self.drop_area)
        
        # Input file/directory with improved styling
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)
        
        input_label = QLabel("Input PDF:")
        input_label.setMinimumWidth(120)
        input_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        input_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        input_layout.addWidget(input_label)
        
        self.input_path_edit = DragDropLineEdit()
        self.input_path_edit.setPlaceholderText("Drag and drop a PDF file here or use Browse button →")
        self.input_path_edit.fileDropped.connect(self.handle_dropped_file)
        self.input_path_edit.setMinimumHeight(30)
        self.input_path_edit.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {self.colors["border"]};
                border-radius: 4px;
                padding: 5px 8px;
                background-color: {self.colors["background"]};
            }}
            QLineEdit:focus {{
                border: 1px solid {self.colors["primary"]};
            }}
        """)
        input_layout.addWidget(self.input_path_edit)
        
        input_browse_button = QPushButton("Browse")
        input_browse_button.setMinimumWidth(100)
        input_browse_button.clicked.connect(self.browse_input)
        input_browse_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors["background_alt"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 4px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #E0E0E0;
                border: 1px solid {self.colors["primary_light"]};
            }}
        """)
        input_layout.addWidget(input_browse_button)
        file_layout.addLayout(input_layout)
        
        # Output file/directory with matching style
        output_layout = QHBoxLayout()
        output_layout.setSpacing(10)
        
        output_label = QLabel("Output Location:")
        output_label.setMinimumWidth(120)
        output_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        output_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        output_layout.addWidget(output_label)
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setMinimumHeight(30)
        self.output_path_edit.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {self.colors["border"]};
                border-radius: 4px;
                padding: 5px 8px;
                background-color: {self.colors["background"]};
            }}
            QLineEdit:focus {{
                border: 1px solid {self.colors["primary"]};
            }}
        """)
        output_layout.addWidget(self.output_path_edit)
        
        output_browse_button = QPushButton("Browse")
        output_browse_button.setMinimumWidth(100)
        output_browse_button.clicked.connect(self.browse_output)
        output_browse_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.colors["background_alt"]};
                border: 1px solid {self.colors["border"]};
                border-radius: 4px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #E0E0E0;
                border: 1px solid {self.colors["primary_light"]};
            }}
        """)
        output_layout.addWidget(output_browse_button)
        file_layout.addLayout(output_layout)
        
        # Processing type with improved styling
        type_layout = QHBoxLayout()
        type_layout.setSpacing(10)
        
        type_label = QLabel("Processing Type:")
        type_label.setMinimumWidth(120)
        type_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        type_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        type_layout.addWidget(type_label)
        
        radio_container = QFrame()
        radio_container.setStyleSheet(f"""
            QFrame {{
                background-color: {self.colors["background_alt"]};
                border-radius: 4px;
                padding: 2px;
            }}
        """)
        radio_layout = QHBoxLayout(radio_container)
        radio_layout.setContentsMargins(10, 5, 10, 5)
        radio_layout.setSpacing(15)
        
        self.single_radio = QRadioButton("Single File")
        self.single_radio.setChecked(True)
        self.single_radio.setStyleSheet(f"""
            QRadioButton {{
                color: {self.colors["text_primary"]};
                font-weight: bold;
            }}
        """)
        
        self.batch_radio = QRadioButton("Batch Directory")
        self.batch_radio.setStyleSheet(f"""
            QRadioButton {{
                color: {self.colors["text_primary"]};
                font-weight: bold;
            }}
        """)
        
        self.single_radio.toggled.connect(self.on_processing_type_change)
        
        radio_layout.addWidget(self.single_radio)
        radio_layout.addWidget(self.batch_radio)
        
        type_layout.addWidget(radio_container)
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
        
        # Creating configuration based on operation mode
        if self.operation_mode == "online":
            provider_id = self.online_provider_combo.currentData()
            model = self.online_model_combo.currentText()
            
            # Create extraction config
            extraction_config = {
                "provider": provider_id,
                "model": model,
                "verification_strategy": self.online_verification_strategy_combo.currentText(),
                "use_offline_provider": False
            }
            
            # Set API key if provided
            api_key = self.api_key_edit.text()
            if api_key:
                extraction_config["api_key"] = api_key
            
            # Set verification provider/model if needed
            if self.online_verification_strategy_combo.currentText() == "specific":
                extraction_config["verification_provider"] = self.online_verification_provider_combo.currentData()
                extraction_config["verification_model"] = self.online_verification_model_combo.currentText()
        
        else:  # Offline mode
            # Check if Ollama is running
            if self.ollama_status_value.text().startswith("Running"):
                # Get the selected model
                if self.offline_model_combo.currentText():
                    model = self.offline_model_combo.currentText()
                else:
                    QMessageBox.critical(self, "Error", "No Ollama model selected. Please select a model.")
                    return
                
                server_url = self.ollama_server_edit.text().strip()
                
                # Create extraction config
                extraction_config = {
                    "provider": "ollama",
                    "model": model,
                    "verification_strategy": "same",  # In offline mode, always use same provider for verification
                    "use_offline_provider": True,
                    "ollama": {
                        "enabled": True,
                        "server_url": server_url
                    }
                }
            else:
                # Ollama not running or no models available
                QMessageBox.critical(self, "Error", 
                    "Ollama service is not available. Please make sure Ollama is running and has models installed.")
                return
        
        # Get app config
        app_config = self.config_manager.get_app_config()
        
        # Combine configs
        config = app_config.copy()
        config.update(extraction_config)
        
        # Add confirmation for online mode
        if self.operation_mode == "online":
            confirm = QMessageBox.warning(
                self,
                "Online Processing",
                "Your content will be sent to external API services over the internet. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if confirm != QMessageBox.StandardButton.Yes:
                return
        
        # Start processing in a separate thread
        self.is_processing = True
        self.process_button.setEnabled(False)
        
        # Clear log
        self.log_text.clear()
        self.log(f"Starting processing in {self.operation_mode.upper()} mode...")
        
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
        # Save app config with current mode
        app_config = {
            "use_offline_provider": self.operation_mode == "offline"
        }
        self.config_manager.update_app_config(app_config)
        
        # Save provider-specific configurations based on current mode
        if self.operation_mode == "online":
            # Update online extraction config
            extraction_config = {
                "provider": self.online_provider_combo.currentData(),
                "model": self.online_model_combo.currentText(),
                "verification_strategy": self.online_verification_strategy_combo.currentText()
            }
            
            # Save API key if provided
            api_key = self.api_key_edit.text()
            if api_key:
                # Update the provider's config with the API key
                provider_id = self.online_provider_combo.currentData()
                provider_config = self.config_manager.get_provider_config(provider_id).copy()
                provider_config["api_key"] = api_key
                self.config_manager.update_provider_config(provider_id, provider_config)
            
            # Set verification provider/model if needed
            if self.online_verification_strategy_combo.currentText() == "specific":
                extraction_config["verification_provider"] = self.online_verification_provider_combo.currentData()
                extraction_config["verification_model"] = self.online_verification_model_combo.currentText()
        
        else:  # Offline mode
            # Update offline extraction config
            extraction_config = {
                "provider": "ollama",
                "model": self.offline_model_combo.currentText(),
                "verification_strategy": "same"  # In offline mode, always use same provider
            }
            
            # Save Ollama config
            ollama_config = {
                "enabled": True,
                "server_url": self.ollama_server_edit.text().strip(),
                "provider_type": "offline"
            }
            self.config_manager.update_provider_config("ollama", ollama_config)
        
        # Update extraction configuration
        self.config_manager.update_extraction_config(extraction_config)
        
        # Save to file
        if self.config_manager.save_config():
            QMessageBox.information(self, "Settings", "Settings saved successfully.")
        else:
            QMessageBox.critical(self, "Error", "Failed to save settings")
    
    def load_config(self):
        """Load settings from configuration"""
        # Load application config
        app_config = self.config_manager.get_app_config()
        extraction_config = self.config_manager.get_extraction_config()
        
        # Set operation mode based on config
        use_offline = app_config.get("use_offline_provider", False)
        self.switch_operation_mode("offline" if use_offline else "online")
        
        # Load online provider settings
        if not use_offline:
            # Set online provider
            provider_id = extraction_config.get("provider", "openai")
            provider_index = self.online_provider_combo.findData(provider_id)
            if provider_index >= 0:
                self.online_provider_combo.setCurrentIndex(provider_index)
            
            # Set online model
            model = extraction_config.get("model")
            if model and self.online_model_combo.findText(model) >= 0:
                self.online_model_combo.setCurrentText(model)
            
            # Set verification strategy
            strategy = extraction_config.get("verification_strategy", "different")
            self.online_verification_strategy_combo.setCurrentText(strategy)
            
            # Set verification provider/model if specific strategy
            if strategy == "specific":
                verification_provider = extraction_config.get("verification_provider")
                if verification_provider:
                    provider_index = self.online_verification_provider_combo.findData(verification_provider)
                    if provider_index >= 0:
                        self.online_verification_provider_combo.setCurrentIndex(provider_index)
                
                verification_model = extraction_config.get("verification_model")
                if verification_model and self.online_verification_model_combo.findText(verification_model) >= 0:
                    self.online_verification_model_combo.setCurrentText(verification_model)
            
            # Update UI based on verification strategy
            self.on_online_verification_strategy_change()
        
        # Load offline provider settings
        else:
            # Set Ollama server URL
            ollama_config = self.config_manager.get_provider_config("ollama")
            server_url = ollama_config.get("server_url", "http://localhost:11434")
            self.ollama_server_edit.setText(server_url)
            
            # Check Ollama status
            self.check_ollama_status()
            
            # Try to set model if available
            model = extraction_config.get("model")
            if model and self.offline_model_combo.findText(model) >= 0:
                self.offline_model_combo.setCurrentText(model)
    
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