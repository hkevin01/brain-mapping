"""
Main GUI Window for Brain Mapping Toolkit
========================================

PyQt6-based main interface for the brain mapping application.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QFileDialog, QTextEdit,
        QProgressBar, QMenuBar, QStatusBar, QSplitter, QGroupBox,
        QGridLayout, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QAction, QIcon, QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataLoaderThread(QThread):
    """Background thread for loading neuroimaging data."""
    
    progress_updated = pyqtSignal(int)
    data_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        """Load data in background thread."""
        try:
            from ..core.data_loader import DataLoader
            
            self.progress_updated.emit(25)
            
            # Load data
            loader = DataLoader()
            result = loader.load_image(self.file_path)
            
            self.progress_updated.emit(75)
            
            if result.get('success', False):
                self.progress_updated.emit(100)
                self.data_loaded.emit(result)
            else:
                self.error_occurred.emit(result.get('error', 'Unknown error'))
                
        except Exception as e:
            self.error_occurred.emit(str(e))


class DataPanel(QWidget):
    """Panel for data management and loading."""
    
    data_loaded = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the data panel UI."""
        layout = QVBoxLayout()
        
        # File loading section
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load Neuroimaging File")
        self.load_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_button)
        
        # File info display
        self.file_info = QTextEdit()
        self.file_info.setMaximumHeight(150)
        self.file_info.setReadOnly(True)
        file_layout.addWidget(self.file_info)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        file_layout.addWidget(self.progress_bar)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Data information section
        info_group = QGroupBox("Data Information")
        info_layout = QGridLayout()
        
        self.info_labels = {}
        info_items = [
            ("Shape:", "shape_label"),
            ("Data Type:", "dtype_label"),
            ("Min Value:", "min_label"),
            ("Max Value:", "max_label"),
            ("File Size:", "size_label")
        ]
        
        for i, (label_text, key) in enumerate(info_items):
            label = QLabel(label_text)
            value_label = QLabel("N/A")
            value_label.setStyleSheet("color: gray;")
            
            info_layout.addWidget(label, i, 0)
            info_layout.addWidget(value_label, i, 1)
            self.info_labels[key] = value_label
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        self.setLayout(layout)
    
    def load_file(self):
        """Open file dialog and load selected file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Neuroimaging File",
            "",
            "NIfTI Files (*.nii *.nii.gz);;DICOM Files (*.dcm);;All Files (*)"
        )
        
        if file_path:
            self.load_file_async(file_path)
    
    def load_file_async(self, file_path: str):
        """Load file asynchronously."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.load_button.setEnabled(False)
        
        # Start loading thread
        self.loader_thread = DataLoaderThread(file_path)
        self.loader_thread.progress_updated.connect(self.progress_bar.setValue)
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.error_occurred.connect(self.on_error)
        self.loader_thread.start()
    
    def on_data_loaded(self, data: Dict):
        """Handle successfully loaded data."""
        self.current_data = data
        self.update_file_info(data)
        self.update_data_info(data)
        
        self.progress_bar.setVisible(False)
        self.load_button.setEnabled(True)
        
        # Emit signal for other components
        self.data_loaded.emit(data)
    
    def on_error(self, error_message: str):
        """Handle loading error."""
        self.file_info.setText(f"Error loading file: {error_message}")
        self.progress_bar.setVisible(False)
        self.load_button.setEnabled(True)
    
    def update_file_info(self, data: Dict):
        """Update file information display."""
        info_text = f"File: {data.get('file_path', 'Unknown')}\n"
        info_text += f"Format: {data.get('format', 'Unknown')}\n"
        info_text += f"Loaded successfully at {data.get('load_time', 'Unknown')}"
        
        self.file_info.setText(info_text)
    
    def update_data_info(self, data: Dict):
        """Update data information labels."""
        image_data = data.get('data')
        if image_data is not None:
            self.info_labels['shape_label'].setText(str(image_data.shape))
            self.info_labels['dtype_label'].setText(str(image_data.dtype))
            self.info_labels['min_label'].setText(f"{image_data.min():.2f}")
            self.info_labels['max_label'].setText(f"{image_data.max():.2f}")
            
            # Calculate file size
            size_bytes = image_data.nbytes
            size_mb = size_bytes / (1024 * 1024)
            self.info_labels['size_label'].setText(f"{size_mb:.1f} MB")
            
            # Update styling
            for label in self.info_labels.values():
                label.setStyleSheet("color: black;")


class VisualizationPanel(QWidget):
    """Panel for 3D visualization controls."""
    
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up visualization panel UI."""
        layout = QVBoxLayout()
        
        # Visualization controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QGridLayout()
        
        # Rendering options
        controls_layout.addWidget(QLabel("Render Type:"), 0, 0)
        self.render_combo = QComboBox()
        self.render_combo.addItems(["Volume", "Surface", "Slices"])
        controls_layout.addWidget(self.render_combo, 0, 1)
        
        # Threshold control
        controls_layout.addWidget(QLabel("Threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1000.0)
        self.threshold_spin.setValue(50.0)
        controls_layout.addWidget(self.threshold_spin, 1, 1)
        
        # Opacity control
        controls_layout.addWidget(QLabel("Opacity:"), 2, 0)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setValue(0.7)
        self.opacity_spin.setSingleStep(0.1)
        controls_layout.addWidget(self.opacity_spin, 2, 1)
        
        # Slice controls for multi-planar view
        controls_layout.addWidget(QLabel("X Slice:"), 3, 0)
        self.x_slice_spin = QSpinBox()
        self.x_slice_spin.setRange(0, 255)
        controls_layout.addWidget(self.x_slice_spin, 3, 1)
        
        controls_layout.addWidget(QLabel("Y Slice:"), 4, 0)
        self.y_slice_spin = QSpinBox()
        self.y_slice_spin.setRange(0, 255)
        controls_layout.addWidget(self.y_slice_spin, 4, 1)
        
        controls_layout.addWidget(QLabel("Z Slice:"), 5, 0)
        self.z_slice_spin = QSpinBox()
        self.z_slice_spin.setRange(0, 255)
        controls_layout.addWidget(self.z_slice_spin, 5, 1)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Action buttons
        buttons_group = QGroupBox("Actions")
        buttons_layout = QVBoxLayout()
        
        self.render_button = QPushButton("Render 3D View")
        self.render_button.setEnabled(False)
        buttons_layout.addWidget(self.render_button)
        
        self.screenshot_button = QPushButton("Save Screenshot")
        self.screenshot_button.setEnabled(False)
        buttons_layout.addWidget(self.screenshot_button)
        
        self.reset_button = QPushButton("Reset View")
        self.reset_button.setEnabled(False)
        buttons_layout.addWidget(self.reset_button)
        
        buttons_group.setLayout(buttons_layout)
        layout.addWidget(buttons_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def set_data(self, data: Dict):
        """Set current data and update controls."""
        self.current_data = data
        
        if data and 'data' in data:
            image_data = data['data']
            shape = image_data.shape
            
            # Update slice ranges
            if len(shape) >= 3:
                self.x_slice_spin.setRange(0, shape[0] - 1)
                self.x_slice_spin.setValue(shape[0] // 2)
                
                self.y_slice_spin.setRange(0, shape[1] - 1)
                self.y_slice_spin.setValue(shape[1] // 2)
                
                self.z_slice_spin.setRange(0, shape[2] - 1)
                self.z_slice_spin.setValue(shape[2] // 2)
            
            # Update threshold range
            max_val = float(image_data.max())
            self.threshold_spin.setRange(0.0, max_val)
            self.threshold_spin.setValue(max_val * 0.3)
            
            # Enable buttons
            self.render_button.setEnabled(True)
            self.screenshot_button.setEnabled(True)
            self.reset_button.setEnabled(True)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Mapping Toolkit")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.data_panel = None
        self.viz_panel = None
        self.current_data = None
        
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        logger.info("Main window initialized")
    
    def setup_ui(self):
        """Set up the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for data and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # Data panel
        self.data_panel = DataPanel()
        self.data_panel.data_loaded.connect(self.on_data_loaded)
        left_layout.addWidget(self.data_panel)
        
        # Visualization controls
        self.viz_panel = VisualizationPanel()
        left_layout.addWidget(self.viz_panel)
        
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(400)
        
        # Right panel for 3D visualization (placeholder)
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        
        viz_placeholder = QLabel("3D Visualization Area")
        viz_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viz_placeholder.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                color: #666;
                font-size: 18px;
            }
        """)
        viz_layout.addWidget(viz_placeholder)
        
        viz_widget.setLayout(viz_layout)
        
        # Add to splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(viz_widget)
        main_splitter.setSizes([400, 800])
        
        # Set main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)
    
    def setup_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.data_panel.load_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        reset_action = QAction('Reset View', self)
        reset_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def on_data_loaded(self, data: Dict):
        """Handle data loaded event."""
        self.current_data = data
        self.viz_panel.set_data(data)
        
        file_name = Path(data.get('file_path', '')).name
        self.status_bar.showMessage(f"Loaded: {file_name}")
    
    def reset_view(self):
        """Reset the 3D view."""
        # Placeholder for view reset functionality
        self.status_bar.showMessage("View reset")
    
    def show_about(self):
        """Show about dialog."""
        from PyQt6.QtWidgets import QMessageBox
        
        QMessageBox.about(
            self,
            "About Brain Mapping Toolkit",
            """
            <h3>Brain Mapping Toolkit</h3>
            <p>Version 1.0.0 (Phase 1)</p>
            <p>An open-source toolkit for neuroimaging analysis and visualization.</p>
            <p>Features:</p>
            <ul>
                <li>Multi-format data support (NIfTI, DICOM)</li>
                <li>3D visualization with VTK</li>
                <li>FSL integration for preprocessing</li>
                <li>AMD ROCm and NVIDIA CUDA support</li>
            </ul>
            """
        )


def main():
    """Main application entry point."""
    if not PYQT_AVAILABLE:
        print("PyQt6 is required to run the GUI application.")
        print("Install with: pip install PyQt6")
        return
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Brain Mapping Toolkit")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Neuroscience Research")
    
    # Apply dark theme (optional)
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
    """)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
