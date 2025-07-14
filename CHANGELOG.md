# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and foundation
- Core data loading capabilities for NIfTI and DICOM formats
- 3D visualization with VTK/Mayavi integration
- Statistical analysis modules for connectivity and activation mapping
- Machine learning pipeline with scikit-learn and PyTorch
- PyQt6-based GUI for interactive analysis
- Command-line interface for batch processing
- GPU acceleration support with CuPy
- Comprehensive documentation and development guides
- CI/CD pipeline with GitHub Actions
- Docker support for containerized deployment

### Features in Development
- Real-time fMRI analysis capabilities
- Cloud integration for collaborative analysis
- Advanced machine learning models for brain state prediction
- Interactive 3D brain atlas integration
- Multi-modal imaging support (fMRI, DTI, structural)

## [1.0.0] - TBD

### Added
- Initial release of Brain Mapping Toolkit
- Complete neuroimaging analysis pipeline
- Cross-platform GUI application
- Comprehensive API for developers
- Integration with existing tools (FSL, SPM, AFNI)
- Extensive documentation and tutorials

### Technical Highlights
- **Performance**: GPU-accelerated processing for large datasets
- **Usability**: Intuitive GUI with real-time visualization
- **Extensibility**: Plugin architecture for custom analysis methods
- **Compatibility**: Support for standard neuroimaging formats
- **Scalability**: Cloud-ready for distributed computing

### Supported Formats
- NIfTI (.nii, .nii.gz)
- DICOM (.dcm)
- Analyze (.img/.hdr)
- MGH/MGZ (FreeSurfer)
- CIFTI (HCP format)

### Analysis Methods
- **Connectivity Analysis**
  - Functional connectivity matrices
  - Effective connectivity (DCM)
  - Graph theory metrics
  
- **Activation Analysis**
  - General linear model (GLM)
  - Statistical parametric mapping
  - Multiple comparison correction
  
- **Machine Learning**
  - Support Vector Machines
  - Random Forest classification
  - Deep neural networks
  - Real-time decoding

### Visualization Features
- **3D Rendering**
  - Volume rendering
  - Surface-based visualization
  - Interactive slice viewing
  
- **Statistical Maps**
  - Activation overlays
  - Connectivity networks
  - Time series plots

### Integration Capabilities
- **External Tools**
  - FSL integration
  - AFNI compatibility
  - SPM pipeline support
  
- **Cloud Services**
  - AWS S3 storage
  - Google Cloud compute
  - Azure machine learning

## Development Roadmap

### Version 1.1.0 (Q2 2024)
- Enhanced real-time analysis
- Improved GPU memory management
- Additional ML algorithms
- Mobile companion app

### Version 1.2.0 (Q3 2024)
- Multi-site analysis tools
- Advanced visualization effects
- API improvements
- Performance optimizations

### Version 2.0.0 (Q4 2024)
- Complete architecture redesign
- WebGL-based visualization
- Distributed computing support
- Enterprise features

## Known Issues

### Current Limitations
- Large dataset memory usage (>32GB RAM recommended)
- GPU drivers compatibility on some systems
- Limited real-time processing on CPU-only systems

### Planned Fixes
- Memory-mapped data loading for large files
- Improved error handling for GPU operations
- CPU fallback optimizations

## Migration Notes

### From Other Tools
- **FSL Users**: Direct FEAT/MELODIC result import
- **SPM Users**: SPM.mat file compatibility
- **AFNI Users**: BRIK/HEAD format support

### API Changes
- All breaking changes will be documented here
- Deprecation warnings provided one version ahead
- Migration scripts available for major updates

## Acknowledgments

### Contributors
- Brain Mapping Toolkit Team
- Neuroimaging community feedback
- Open source library maintainers

### Inspiration
- NiPy: Python neuroimaging foundation
- BrainIAK: Shared response modeling
- Nilearn: Machine learning for neuroimaging
- VTK: 3D visualization toolkit

### Funding
- To be updated with grant information
- Community donations and sponsorships

---

For detailed technical changes, see the [commit history](https://github.com/your-org/brain-mapping-toolkit/commits/main).
