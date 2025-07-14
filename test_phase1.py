#!/usr/bin/env python3
"""
Phase 1 Validation Test
======================

Quick validation script to verify Phase 1 components are working correctly.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_core_modules():
    """Test core functionality modules."""
    print("Testing Core Modules...")
    
    try:
        from brain_mapping.core.data_loader import DataLoader
        loader = DataLoader()
        print("✓ DataLoader imported successfully")
        
        from brain_mapping.core.fsl_integration import FSLIntegration
        fsl = FSLIntegration()
        print("✓ FSLIntegration imported successfully")
        
        from brain_mapping.core.quality_control import QualityControl
        qc = QualityControl()
        print("✓ QualityControl imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Core modules test failed: {e}")
        return False

def test_visualization_modules():
    """Test visualization modules."""
    print("\nTesting Visualization Modules...")
    
    try:
        from brain_mapping.visualization.glass_brain import GlassBrainProjector
        glass_brain = GlassBrainProjector()
        print("✓ GlassBrainProjector imported successfully")
        
        from brain_mapping.visualization.multi_planar import MultiPlanarReconstructor
        mpr = MultiPlanarReconstructor()
        print("✓ MultiPlanarReconstructor imported successfully")
        
        from brain_mapping.visualization.interactive_atlas import InteractiveBrainAtlas
        atlas = InteractiveBrainAtlas()
        print("✓ InteractiveBrainAtlas imported successfully")
        
        from brain_mapping.visualization.renderer_3d import Visualizer
        viz = Visualizer()
        print("✓ 3D Renderer imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Visualization modules test failed: {e}")
        return False

def test_synthetic_data_processing():
    """Test with synthetic data."""
    print("\nTesting Synthetic Data Processing...")
    
    try:
        # Create synthetic 3D brain data
        synthetic_data = np.random.randn(64, 64, 32) * 100 + 1000
        print("✓ Synthetic 3D data created")
        
        # Test glass brain projection
        from brain_mapping.visualization.glass_brain import GlassBrainProjector
        projector = GlassBrainProjector()
        fig = projector.create_projection(synthetic_data, projection_type='max')
        print("✓ Glass brain projection created")
        
        # Test multi-planar reconstruction
        from brain_mapping.visualization.multi_planar import MultiPlanarReconstructor
        mpr = MultiPlanarReconstructor()
        fig = mpr.create_orthogonal_views(synthetic_data)
        print("✓ Multi-planar reconstruction created")
        
        # Test interactive atlas
        from brain_mapping.visualization.interactive_atlas import InteractiveBrainAtlas
        atlas = InteractiveBrainAtlas()
        atlas.load_atlas()  # This will create synthetic atlas
        regions = atlas.list_regions()
        print(f"✓ Interactive atlas loaded with {len(regions)} regions")
        
        return True
    except Exception as e:
        print(f"✗ Synthetic data processing failed: {e}")
        return False

def main():
    """Run Phase 1 validation tests."""
    print("=" * 50)
    print("Brain Mapping Toolkit - Phase 1 Validation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_core_modules()
    all_tests_passed &= test_visualization_modules()
    all_tests_passed &= test_synthetic_data_processing()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 Phase 1 Validation: ALL TESTS PASSED!")
        print("Phase 1 implementation is working correctly.")
    else:
        print("❌ Phase 1 Validation: SOME TESTS FAILED!")
        print("Please check the error messages above.")
    print("=" * 50)
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
