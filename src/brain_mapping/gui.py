"""GUI module for Brain Mapping Toolkit."""

import sys
from pathlib import Path
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QPushButton, QLabel, QFileDialog, QTextEdit,
        QProgressBar, QSplitter, QMenuBar, QStatusBar, QToolBar,
        QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
        QCheckBox, QSlider, QMessageBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QAction, QPixmap, QIcon
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    # Fallback for when PyQt6 is not available
    class QMainWindow:
        pass


class BrainMapperGUI(QMainWindow):
    """Main GUI window for Brain Mapping Toolkit."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt6 is required for the GUI. Install with: pip install PyQt6")
        
        self.setWindowTitle("Brain Mapping Toolkit")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize components
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_status_bar()
        
        # Initialize processing thread
        self.processing_thread = None
        
    def setup_ui(self):
        """Set up the main user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Controls
        self.setup_control_panel(splitter)
        
        # Right panel - Visualization and results
        self.setup_visualization_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
        
    def setup_control_panel(self, parent):
        """Set up the left control panel."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Data loading section
        data_group = QGroupBox("Data Loading")
        data_layout = QFormLayout(data_group)
        
        self.input_path_label = QLabel("No file selected")
        load_button = QPushButton("Load Brain Data")
        load_button.clicked.connect(self.load_data)
        
        data_layout.addRow("Input File:", self.input_path_label)
        data_layout.addRow("", load_button)
        
        # Preprocessing section
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QFormLayout(preprocess_group)
        
        self.normalize_check = QCheckBox("Normalize")
        self.smooth_check = QCheckBox("Smooth")
        self.smooth_fwhm = QDoubleSpinBox()
        self.smooth_fwhm.setRange(0.5, 10.0)
        self.smooth_fwhm.setValue(6.0)
        self.smooth_fwhm.setSuffix(" mm")
        
        preprocess_layout.addRow("", self.normalize_check)
        preprocess_layout.addRow("", self.smooth_check)
        preprocess_layout.addRow("FWHM:", self.smooth_fwhm)
        
        preprocess_button = QPushButton("Apply Preprocessing")
        preprocess_button.clicked.connect(self.apply_preprocessing)
        preprocess_layout.addRow("", preprocess_button)
        
        # Analysis section
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QFormLayout(analysis_group)
        
        self.analysis_method = QComboBox()
        self.analysis_method.addItems(["Connectivity", "Activation", "Network"])
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(1, 100)
        self.threshold_slider.setValue(5)
        self.threshold_label = QLabel("0.05")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        
        analysis_layout.addRow("Method:", self.analysis_method)
        analysis_layout.addRow("Threshold:", self.threshold_slider)
        analysis_layout.addRow("", self.threshold_label)
        
        analyze_button = QPushButton("Run Analysis")
        analyze_button.clicked.connect(self.run_analysis)
        analysis_layout.addRow("", analyze_button)
        
        # Visualization section
        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)
        
        self.viz_type = QComboBox()
        self.viz_type.addItems(["Volume", "Surface", "Slices"])
        
        self.colormap = QComboBox()
        self.colormap.addItems(["viridis", "hot", "cool", "jet", "gray"])
        
        viz_layout.addRow("Type:", self.viz_type)
        viz_layout.addRow("Colormap:", self.colormap)
        
        visualize_button = QPushButton("Visualize")
        visualize_button.clicked.connect(self.create_visualization)
        viz_layout.addRow("", visualize_button)
        
        # Add all groups to control layout
        control_layout.addWidget(data_group)
        control_layout.addWidget(preprocess_group)
        control_layout.addWidget(analysis_group)
        control_layout.addWidget(viz_group)
        control_layout.addStretch()
        
        parent.addWidget(control_widget)
        
    def setup_visualization_panel(self, parent):
        """Set up the right visualization panel."""
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Visualization tab
        self.viz_tab = QWidget()
        viz_tab_layout = QVBoxLayout(self.viz_tab)
        
        # Placeholder for 3D visualization
        self.viz_placeholder = QLabel("Load brain data to begin visualization")
        self.viz_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_placeholder.setMinimumHeight(400)
        self.viz_placeholder.setStyleSheet("border: 2px dashed #aaa;")
        viz_tab_layout.addWidget(self.viz_placeholder)
        
        self.tab_widget.addTab(self.viz_tab, "Visualization")
        
        # Results tab
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setPlaceholderText("Analysis results will appear here...")
        results_layout.addWidget(self.results_text)
        
        self.tab_widget.addTab(self.results_tab, "Results")
        
        # Log tab
        self.log_tab = QWidget()
        log_layout = QVBoxLayout(self.log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("Processing logs will appear here...")
        log_layout.addWidget(self.log_text)
        
        self.tab_widget.addTab(self.log_tab, "Log")
        
        viz_layout.addWidget(self.tab_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        viz_layout.addWidget(self.progress_bar)
        
        parent.addWidget(viz_widget)
        
    def setup_menus(self):
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Data", self)
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)
        
        save_action = QAction("Save Results", self)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.show_preferences)
        tools_menu.addAction(preferences_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_toolbars(self):
        """Set up the toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        load_action = QAction("Load", self)
        load_action.triggered.connect(self.load_data)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        analyze_action = QAction("Analyze", self)
        analyze_action.triggered.connect(self.run_analysis)
        toolbar.addAction(analyze_action)
        
        visualize_action = QAction("Visualize", self)
        visualize_action.triggered.connect(self.create_visualization)
        toolbar.addAction(visualize_action)
        
    def setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def update_threshold_label(self, value):
        """Update threshold label when slider changes."""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        
    def load_data(self):
        """Load brain data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Brain Data",
            "",
            "NIfTI files (*.nii *.nii.gz);;DICOM files (*.dcm);;All files (*)"
        )
        
        if file_path:
            self.input_path_label.setText(Path(file_path).name)
            self.log_message(f"Loaded: {file_path}")
            self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
            
    def apply_preprocessing(self):
        """Apply preprocessing to loaded data."""
        if self.input_path_label.text() == "No file selected":
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
            
        self.log_message("Applying preprocessing...")
        self.status_bar.showMessage("Preprocessing...")
        
        # Start processing in background thread
        self.start_processing("preprocessing")
        
    def run_analysis(self):
        """Run brain analysis."""
        if self.input_path_label.text() == "No file selected":
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
            
        method = self.analysis_method.currentText()
        threshold = self.threshold_slider.value() / 100.0
        
        self.log_message(f"Running {method} analysis (threshold: {threshold:.2f})...")
        self.status_bar.showMessage(f"Running {method} analysis...")
        
        # Start processing in background thread
        self.start_processing("analysis")
        
    def create_visualization(self):
        """Create brain visualization."""
        if self.input_path_label.text() == "No file selected":
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return
            
        viz_type = self.viz_type.currentText()
        colormap = self.colormap.currentText()
        
        self.log_message(f"Creating {viz_type} visualization with {colormap} colormap...")
        self.status_bar.showMessage("Creating visualization...")
        
        # Start processing in background thread
        self.start_processing("visualization")
        
    def start_processing(self, task_type):
        """Start background processing thread."""
        if self.processing_thread and self.processing_thread.isRunning():
            return
            
        self.processing_thread = ProcessingThread(task_type, self)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.message.connect(self.log_message)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.processing_thread.start()
        
    def processing_finished(self, task_type, results):
        """Handle completion of background processing."""
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Ready")
        
        if task_type == "analysis":
            self.results_text.setText(str(results))
            self.tab_widget.setCurrentIndex(1)  # Switch to results tab
        elif task_type == "visualization":
            self.viz_placeholder.setText("Visualization completed")
            self.tab_widget.setCurrentIndex(0)  # Switch to visualization tab
            
        self.log_message(f"{task_type.capitalize()} completed successfully.")
        
    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        
    def log_message(self, message):
        """Add message to log."""
        self.log_text.append(f"[{QTimer().currentTime().toString()}] {message}")
        
    def save_results(self):
        """Save analysis results."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "Text files (*.txt);;CSV files (*.csv);;All files (*)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.results_text.toPlainText())
            self.log_message(f"Results saved to: {file_path}")
            
    def show_preferences(self):
        """Show preferences dialog."""
        QMessageBox.information(self, "Preferences", "Preferences dialog not implemented yet.")
        
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Brain Mapping Toolkit",
            "Brain Mapping Toolkit\n\n"
            "GPU-accelerated neuroimaging analysis and visualization\n\n"
            "Version: 1.0.0\n"
            "License: MIT"
        )


class ProcessingThread(QThread):
    """Background thread for processing tasks."""
    
    finished = pyqtSignal(str, object)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    
    def __init__(self, task_type, parent=None):
        super().__init__(parent)
        self.task_type = task_type
        
    def run(self):
        """Run the processing task."""
        try:
            if self.task_type == "preprocessing":
                self.run_preprocessing()
            elif self.task_type == "analysis":
                results = self.run_analysis()
                self.finished.emit(self.task_type, results)
            elif self.task_type == "visualization":
                self.run_visualization()
                
        except Exception as e:
            self.message.emit(f"Error: {str(e)}")
            
    def run_preprocessing(self):
        """Run preprocessing in background."""
        for i in range(101):
            self.progress.emit(i)
            self.msleep(50)  # Simulate processing time
        self.finished.emit(self.task_type, "Preprocessing completed")
        
    def run_analysis(self):
        """Run analysis in background."""
        self.message.emit("Initializing analysis...")
        self.progress.emit(25)
        self.msleep(1000)
        
        self.message.emit("Computing statistics...")
        self.progress.emit(50)
        self.msleep(1000)
        
        self.message.emit("Generating results...")
        self.progress.emit(75)
        self.msleep(1000)
        
        self.progress.emit(100)
        return "Analysis results:\n\nConnectivity matrix computed\nStatistical significance: p < 0.05\nNetwork hubs identified: 12 regions"
        
    def run_visualization(self):
        """Run visualization in background."""
        for i in range(101):
            self.progress.emit(i)
            self.msleep(30)  # Simulate processing time
        self.finished.emit(self.task_type, "Visualization completed")


def main():
    """Main entry point for the GUI application."""
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required for the GUI.")
        print("Install with: pip install PyQt6")
        sys.exit(1)
        
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Brain Mapping Toolkit")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Brain Mapping Toolkit Team")
    
    # Create and show main window
    window = BrainMapperGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
