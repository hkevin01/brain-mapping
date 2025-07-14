# Brain Mapping Toolkit

An open-source, GPU-accelerated toolkit for advanced brain imaging analysis and visualization.

## Features

- **Multi-format Data Support**: DICOM, NIfTI, BIDS compliance
- **GPU Acceleration**: CUDA-powered real-time processing  
- **3D Visualization**: Interactive VTK/Mayavi rendering
- **Machine Learning**: Integrated ML pipelines with PyTorch/scikit-learn
- **Cloud Collaboration**: Secure sharing and collaborative analysis
- **Clinical Workflows**: HIPAA-compliant tools for medical applications

## Quick Start

```bash
pip install brain-mapping-toolkit
```

```python
from brain_mapping import BrainMapper, Visualizer

# Load and visualize fMRI data
mapper = BrainMapper()
data = mapper.load('fmri_data.nii.gz')
viz = Visualizer()
viz.plot_brain_3d(data, threshold=2.3)
viz.show()
```

## Documentation

- [Project Plan](docs/project-plan.md)
- [Integration Guide](docs/integration.md)
- [API Reference](https://brain-mapping-toolkit.readthedocs.io)

## Contributing

We welcome contributions! See our [contribution guidelines](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📧 Email: support@brain-mapping-toolkit.org
- 💬 Discussions: [GitHub Discussions](https://github.com/your-org/brain-mapping-toolkit/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/brain-mapping-toolkit/issues)
