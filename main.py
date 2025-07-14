#!/usr/bin/env python3
"""
Brain Mapping Toolkit - Main Application Entry Point
====================================================

Phase 1 Brain Mapping Toolkit with core functionality:
- FSL integration for preprocessing
- Quality control validation
- Glass brain projections
- Multi-planar reconstruction
- PyQt6 GUI interface
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from PyQt6.QtWidgets import QApplication
    from brain_mapping.gui.main_window import MainWindow
    PYQT_AVAILABLE = True
except ImportError:
    print("PyQt6 not available. Please install with: pip install PyQt6")
    PYQT_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('brain_mapping.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    logger.info("Starting Brain Mapping Toolkit - Phase 1")
    
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required to run the GUI application.")
        print("Install with: pip install PyQt6")
        sys.exit(1)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Brain Mapping Toolkit")
    app.setOrganizationName("BrainMapping")
    app.setApplicationVersion("1.0.0-Phase1")
    
    # Create and show main window
    try:
        main_window = MainWindow()
        main_window.show()
        
        logger.info("Brain Mapping Toolkit GUI started successfully")
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Error starting application: {e}")
        sys.exit(1)


def cli_mode():
    """Command-line interface mode for headless operations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Brain Mapping Toolkit - Phase 1 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --cli --preprocess data/input.nii.gz --output results/
  python main.py --cli --quality-check data/preprocessed.nii.gz
  python main.py --cli --glass-brain data/stats.nii.gz --output viz/glass_brain.png
        """
    )
    
    parser.add_argument('--cli', action='store_true', 
                       help='Run in command-line mode')
    parser.add_argument('--preprocess', type=str,
                       help='Run FSL preprocessing on input file')
    parser.add_argument('--quality-check', type=str,
                       help='Run quality control on input file')
    parser.add_argument('--glass-brain', type=str,
                       help='Generate glass brain projection')
    parser.add_argument('--multi-planar', type=str,
                       help='Generate multi-planar reconstruction')
    parser.add_argument('--output', type=str, default='output/',
                       help='Output directory (default: output/)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.cli:
        return False
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.preprocess:
            from brain_mapping.core.fsl_integration import FSLIntegration
            fsl = FSLIntegration()
            logger.info(f"Running FSL preprocessing on: {args.preprocess}")
            result = fsl.run_bet(args.preprocess, str(output_dir / "preprocessed.nii.gz"))
            logger.info(f"Preprocessing completed: {result}")
            
        if args.quality_check:
            from brain_mapping.core.quality_control import QualityControl
            qc = QualityControl()
            logger.info(f"Running quality control on: {args.quality_check}")
            report = qc.generate_report(args.quality_check)
            logger.info(f"Quality control completed: {report}")
            
        if args.glass_brain:
            from brain_mapping.visualization.glass_brain import quick_glass_brain
            logger.info(f"Generating glass brain projection: {args.glass_brain}")
            fig = quick_glass_brain(args.glass_brain, 
                                  output_path=output_dir / "glass_brain.png")
            logger.info("Glass brain projection completed")
            
        if args.multi_planar:
            from brain_mapping.visualization.multi_planar import quick_orthogonal_view
            logger.info(f"Generating multi-planar reconstruction: {args.multi_planar}")
            fig = quick_orthogonal_view(args.multi_planar,
                                      output_path=output_dir / "multi_planar.png")
            logger.info("Multi-planar reconstruction completed")
            
        return True
        
    except Exception as e:
        logger.error(f"CLI operation failed: {e}")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Check if CLI mode is requested
    if '--cli' in sys.argv:
        success = cli_mode()
        sys.exit(0 if success else 1)
    else:
        main()
