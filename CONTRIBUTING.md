# Contributing to Ultra-AI

Thank you for your interest in contributing to Ultra-AI! This document provides guidelines and instructions for contributing to this revolutionary 390B parameter multimodal AI model project.

## 🌟 How to Contribute

We welcome contributions in many forms:

- 🐛 **Bug reports and fixes**
- ✨ **New features and enhancements**  
- 📚 **Documentation improvements**
- 🔬 **Research and experimental features**
- 🧪 **Tests and benchmarks**
- 🎨 **Examples and tutorials**

## 📋 Getting Started

### Prerequisites

- Python 3.9 or higher
- CUDA 12.0+ (for GPU training)
- Git and Git LFS
- 16GB+ RAM (32GB+ recommended)

### Development Setup

1. **Fork and clone the repository:**

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/ultra-ai-model.git
cd ultra-ai-model

# Add upstream remote
git remote add upstream https://github.com/ultra-ai/ultra-ai-model.git
```

2. **Create a virtual environment:**

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install development dependencies:**

```bash
# Install the package in development mode
pip install -e ".[dev,training,docs]"

# Install pre-commit hooks
pre-commit install
```

4. **Verify installation:**

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Run code formatting check
black --check src/ tests/
flake8 src/ tests/

# Run type checking
mypy src/
```

## 🔄 Development Workflow

### Branch Strategy

We use a simplified Git flow:

- `main`: Stable, production-ready code
- `develop`: Integration branch for new features  
- `feature/*`: Feature development branches
- `fix/*`: Bug fix branches
- `docs/*`: Documentation update branches

### Creating a Contribution

1. **Create a feature branch:**

```bash
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name
```

2. **Make your changes:**

   - Follow the [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit your changes:**

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add selective attention mechanism

- Implement core context aware attention
- Add linear scaling for ultra-long sequences  
- Include comprehensive tests and benchmarks
- Update documentation with usage examples

Closes #123"
```

4. **Push and create a pull request:**

```bash
git push origin feature/your-feature-name
```

Then create a pull request through the GitHub interface.

## 📝 Coding Standards

### Code Style

We use the following tools for consistent code style:

- **Black**: Code formatting (line length: 88)
- **Flake8**: Linting and style checking
- **isort**: Import sorting
- **mypy**: Type checking

### Code Quality Guidelines

1. **Type Hints**: All functions should have type hints

```python
def selective_scan(
    u: torch.Tensor, 
    delta: torch.Tensor, 
    A: torch.Tensor,
    B: torch.Tensor, 
    C: torch.Tensor,
    D: torch.Tensor
) -> torch.Tensor:
    """
    Selective scan operation with linear complexity.
    
    Args:
        u: Input sequence [batch, length, d_model]
        delta: Selection parameter [batch, length, d_model]
        A: State transition matrix [d_model, d_state]
        B: Input projection [batch, length, d_state] 
        C: Output projection [batch, length, d_state]
        D: Skip connection [d_model]
        
    Returns:
        torch.Tensor: Output sequence [batch, length, d_model]
    """
```

2. **Docstrings**: Use Google-style docstrings

```python
class UltraAIModel(nn.Module):
    """
    Revolutionary 390B parameter multimodal AI model.
    
    Combines Mamba-2 backbone, hybrid attention, MoE, and multimodal fusion
    for ultra-long context understanding and generation.
    
    Args:
        config: Model configuration object
        
    Attributes:
        config: Model configuration
        embeddings: Token embedding layer
        layers: Transformer layers
        lm_head: Language modeling head
        
    Example:
        >>> config = UltraAIConfig.load("config/base.yaml")
        >>> model = UltraAIModel(config)
        >>> outputs = model(input_ids)
    """
```

3. **Error Handling**: Use specific exceptions with clear messages

```python
if config.d_model % config.n_heads != 0:
    raise ValueError(
        f"d_model ({config.d_model}) must be divisible by "
        f"n_heads ({config.n_heads})"
    )
```

4. **Constants**: Use uppercase for constants

```python
DEFAULT_MAX_SEQ_LENGTH = 100_000_000
MAMBA_EXPAND_FACTOR = 2
MoE_TOP_K = 2
```

### File Organization

```
src/
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── ultra_ai_model.py   # Main model class
│   ├── mamba/             # Mamba-2 components
│   ├── attention/         # Attention mechanisms
│   ├── moe/              # Mixture of Experts
│   └── multimodal/       # Multimodal components
├── training/              # Training pipeline
├── utils/                # Utilities and helpers
└── config/               # Configuration files
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Benchmark critical operations
4. **Memory Tests**: Validate memory usage
5. **Distributed Tests**: Test multi-GPU functionality

### Writing Tests

```python
# tests/test_mamba_layer.py
import torch
import pytest
from src.models.mamba import Mamba2Layer, Mamba2Config

class TestMamba2Layer:
    """Test suite for Mamba-2 layer implementation."""
    
    @pytest.fixture
    def config(self):
        return Mamba2Config(
            d_model=512,
            d_state=64,
            d_conv=4,
            expand_factor=2
        )
    
    @pytest.fixture
    def layer(self, config):
        return Mamba2Layer(config)
    
    def test_forward_pass_shape(self, layer, config):
        """Test that forward pass produces correct output shape."""
        batch_size, seq_len = 2, 100
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
    
    def test_selective_scan_linearity(self, layer):
        """Test that selective scan has linear complexity."""
        # Test with different sequence lengths
        for seq_len in [1000, 10000, 100000]:
            x = torch.randn(1, seq_len, layer.config.d_model)
            
            start_time = time.time()
            output = layer(x)
            elapsed_time = time.time() - start_time
            
            # Assert linear scaling (implementation-dependent)
            assert elapsed_time < seq_len * 1e-6  # Rough linear bound
    
    @pytest.mark.gpu
    def test_cuda_compatibility(self, layer, config):
        """Test CUDA compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        layer = layer.cuda()
        x = torch.randn(2, 100, config.d_model).cuda()
        
        output = layer(x)
        
        assert output.is_cuda
        assert output.shape == x.shape
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -m "not slow" -v          # Skip slow tests
pytest tests/ -m gpu -v                  # Only GPU tests
pytest tests/test_mamba_layer.py -v      # Specific test file

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ -v --benchmark-only
```

## 📚 Documentation

### Documentation Types

1. **API Documentation**: Docstrings in code
2. **User Guides**: Step-by-step tutorials
3. **Architecture Documentation**: Technical details
4. **Examples**: Practical usage examples

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# Serve locally
python -m http.server 8000 -d _build/html/
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples for all public APIs
- Add diagrams for complex architectural concepts
- Keep examples up-to-date with the codebase

## 🔬 Research Contributions

### Experimental Features

We welcome research contributions and experimental features:

1. **Create an RFC**: For major architectural changes
2. **Implement in experimental branch**: Use `experimental/*` branches
3. **Provide benchmarks**: Compare against existing implementations
4. **Document thoroughly**: Explain the research motivation

### Research Areas

Current research focus areas:

- **Ultra-long context scaling**: Beyond 100M tokens
- **Multimodal fusion techniques**: Better cross-modal understanding
- **Efficiency improvements**: Memory and compute optimization
- **Novel attention mechanisms**: Alternatives to traditional attention
- **Training stability**: Improved training dynamics

## 📊 Performance Benchmarking

### Benchmark Requirements

All performance-critical contributions should include benchmarks:

```python
# tests/benchmarks/test_attention_performance.py
import pytest
import torch
import time
from src.models.attention import LinearAttention, FullAttention

class TestAttentionPerformance:
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("seq_len", [1000, 10000, 100000])
    def test_linear_attention_scaling(self, benchmark, seq_len):
        """Benchmark linear attention scaling."""
        model = LinearAttention(d_model=512, n_heads=8)
        x = torch.randn(1, seq_len, 512)
        
        result = benchmark(model, x)
        
        # Assert reasonable performance
        assert benchmark.stats.stats.mean < seq_len * 1e-5
```

### Performance Standards

- **Memory efficiency**: O(L) scaling for sequence length L
- **Computational efficiency**: Sub-quadratic complexity
- **GPU utilization**: >80% utilization on target hardware
- **Throughput**: Minimum tokens/second benchmarks

## 🐛 Bug Reports

### Bug Report Template

When reporting bugs, please include:

1. **Environment details**: OS, Python version, CUDA version
2. **Installation method**: pip, source, Docker
3. **Minimal reproduction**: Smallest code that reproduces the issue
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Error messages**: Full stack traces
7. **Configuration**: Relevant config files

### Example Bug Report

```markdown
## Bug Description
Mamba-2 layer produces NaN outputs with very long sequences (>1M tokens).

## Environment
- OS: Ubuntu 22.04
- Python: 3.11.5
- PyTorch: 2.1.0
- CUDA: 12.1
- Ultra-AI: main branch (commit abc123)

## Reproduction
```python
import torch
from src.models.mamba import Mamba2Layer, Mamba2Config

config = Mamba2Config(d_model=512, d_state=64)
layer = Mamba2Layer(config)
x = torch.randn(1, 1_000_000, 512)  # Very long sequence

output = layer(x)
print(torch.isnan(output).any())  # Returns True
```

## Expected Behavior
Output should be finite values, no NaN.

## Actual Behavior  
Output contains NaN values starting around token 800k.
```

## 🚀 Feature Requests

### Feature Request Template

1. **Feature description**: Clear description of the proposed feature
2. **Motivation**: Why this feature is needed
3. **Proposed implementation**: How it could be implemented
4. **Alternatives considered**: Other approaches considered
5. **Additional context**: Any other relevant information

## 🏆 Recognition

### Contributors

We recognize contributors in several ways:

- **Contributors file**: Listed in CONTRIBUTORS.md
- **Commit attribution**: Proper git commit attribution
- **Release notes**: Major contributions mentioned in releases
- **Research citations**: Research contributions cited in papers

### Levels of Contribution

- **Code Contributors**: Bug fixes, features, optimizations
- **Research Contributors**: Novel algorithms, architectural improvements
- **Documentation Contributors**: Guides, tutorials, examples
- **Community Contributors**: Issue triage, user support

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions  
- **Discord**: Real-time chat with maintainers and community
- **Email**: contact@ultra-ai.com for sensitive matters

### Questions and Support

Before asking questions:

1. Check existing documentation
2. Search GitHub issues and discussions
3. Review the FAQ
4. Try the example code

When asking questions:

- Provide context and background
- Include minimal reproduction code
- Specify your environment
- Show what you've already tried

## 📄 License and Legal

### License Agreement

By contributing to Ultra-AI, you agree that your contributions will be licensed under the Apache License 2.0.

### Copyright Assignment

- You retain copyright to your contributions
- You grant Ultra-AI Team rights to use and distribute your contributions
- Your contributions must be original work or properly attributed

### Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🎯 Contribution Checklist

Before submitting a pull request:

- [ ] Code follows project style guidelines
- [ ] Tests pass locally (`pytest tests/`)
- [ ] Code is properly formatted (`black src/ tests/`)
- [ ] Linting passes (`flake8 src/ tests/`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation is updated
- [ ] Tests are added for new functionality
- [ ] Commit messages follow conventions
- [ ] PR description is clear and complete
- [ ] No sensitive information is included

## 📈 Project Roadmap

### Current Priorities

1. **Performance optimization**: Memory and compute efficiency
2. **Multimodal improvements**: Better cross-modal understanding
3. **Training stability**: Robust distributed training
4. **Documentation**: Comprehensive guides and examples

### Future Plans

- **Ultra-long context**: Scaling to 1B+ tokens
- **Edge deployment**: Mobile and embedded optimizations
- **New modalities**: Support for more input types
- **Research features**: Cutting-edge experimental capabilities

---

Thank you for contributing to Ultra-AI! Together, we're pushing the boundaries of artificial intelligence and creating the next generation of AI models. 🚀