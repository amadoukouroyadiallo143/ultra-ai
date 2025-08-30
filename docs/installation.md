# Installation Guide

This guide provides complete installation instructions for Ultra-AI model training and deployment.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (11.0+), Windows 10+
- **Python**: 3.9 or higher
- **RAM**: 32GB+ for training, 16GB+ for inference
- **Storage**: 1TB+ for full model training

### Recommended Requirements  
- **OS**: Linux Ubuntu 22.04 LTS
- **Python**: 3.11
- **RAM**: 128GB+ for distributed training
- **GPUs**: 8x NVIDIA A100 80GB or H100 80GB
- **Storage**: 10TB+ NVMe SSD
- **Network**: InfiniBand for multi-node training

## CUDA Support

### CUDA Installation
Ultra-AI requires CUDA 12.0 or higher for GPU acceleration:

```bash
# Download and install CUDA Toolkit 12.0+
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvidia-smi
nvcc --version
```

### GPU Memory Requirements

| Model Variant | Training Memory | Inference Memory | GPUs Required |
|---------------|-----------------|------------------|---------------|
| Ultra-390B | 800GB+ | 200GB+ | 8x A100/H100 80GB |
| Ultra-52B | 200GB+ | 50GB+ | 4x A100 80GB |
| Ultra-13B | 50GB+ | 12GB+ | 1x A100 80GB |
| Ultra-3B | 12GB+ | 3GB+ | 1x RTX 4090 |
| Ultra-Edge | 4GB+ | 1GB+ | 1x RTX 3080 |

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone repository
git clone https://github.com/ultra-ai/ultra-ai-model.git
cd ultra-ai-model

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e .

# Verify installation
python -c "import ultra_ai_model; print('Installation successful!')"
```

### Method 2: Development Install

```bash
# Clone with submodules
git clone --recursive https://github.com/ultra-ai/ultra-ai-model.git
cd ultra-ai-model

# Create conda environment
conda create -n ultra-ai python=3.11
conda activate ultra-ai

# Install development dependencies
pip install -e ".[dev,training,docs]"

# Install pre-commit hooks
pre-commit install
```

### Method 3: Docker Install

```bash
# Pull Docker image
docker pull ultraai/ultra-ai-model:latest

# Run container with GPU support
docker run --gpus all -it --shm-size=32gb ultraai/ultra-ai-model:latest

# Or build from source
docker build -t ultra-ai-local .
docker run --gpus all -it ultra-ai-local
```

## Package Variants

### Core Package
```bash
pip install ultra-ai-model
```

### Training Package (Includes DeepSpeed, Flash Attention)
```bash
pip install ultra-ai-model[training]
```

### Inference Package (ONNX, TensorRT optimizations)
```bash
pip install ultra-ai-model[inference]
```

### Full Package (All features)
```bash
pip install ultra-ai-model[all]
```

## Advanced Dependencies

### Flash Attention (Recommended)
```bash
# Install Flash Attention for memory efficiency
pip install flash-attn --no-build-isolation

# Alternative: use xFormers
pip install xformers
```

### DeepSpeed (Required for distributed training)
```bash
# Install DeepSpeed
pip install deepspeed

# Verify DeepSpeed installation
ds_report
```

### Apex (Optional, for mixed precision)
```bash
# Install NVIDIA Apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Environment Setup

### Environment Variables
Create a `.env` file:

```bash
# Performance optimizations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_MEMORY_FRACTION=0.9

# Distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
```

### System Optimizations

#### Linux Kernel Parameters
```bash
# Add to /etc/sysctl.conf
vm.max_map_count=262144
fs.file-max=2097152

# Apply changes
sudo sysctl -p
```

#### Network Optimizations (Multi-node)
```bash
# InfiniBand optimization
echo 'net.core.rmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

## Verification

### Quick Test
```python
import torch
from ultra_ai_model import UltraAIModel, UltraAIConfig

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

# Load minimal config
config = UltraAIConfig(
    d_model=512,
    mamba_layers=2,
    attention_layers=1,
    num_experts=8,
    max_seq_length=1024
)

# Create model
model = UltraAIModel(config)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
input_ids = torch.randint(0, 1000, (1, 10))
with torch.no_grad():
    outputs = model(input_ids)
    print(f"Output shape: {outputs['logits'].shape}")

print("✅ Installation verified successfully!")
```

### Comprehensive Test
```bash
# Run test suite
pytest tests/ -v

# Run specific tests
pytest tests/test_model.py::test_forward_pass -v
pytest tests/test_training.py::test_distributed_setup -v

# Benchmark performance
python scripts/benchmark.py --model-size ultra-3b --batch-size 1 --seq-length 10000
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size or sequence length
export CUDA_MEMORY_FRACTION=0.8

# Enable gradient checkpointing
python scripts/train.py --gradient-checkpointing
```

#### DeepSpeed Installation Issues
```bash
# Install with specific CUDA version
DS_BUILD_OPS=1 DS_BUILD_UTILS=0 pip install deepspeed --global-option="build_ext" --global-option="-j8"

# Or use conda
conda install deepspeed -c conda-forge
```

#### Flash Attention Build Errors
```bash
# Install prerequisites
pip install packaging ninja

# Build with specific flags
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Reinstall in development mode
pip uninstall ultra-ai-model
pip install -e .
```

### Performance Issues

#### Slow Training
1. **Enable mixed precision**: `--mixed-precision`
2. **Use gradient checkpointing**: `--gradient-checkpointing`  
3. **Optimize data loading**: `--num-workers 8`
4. **Use DeepSpeed**: `--deepspeed --zero-stage 2`

#### High Memory Usage
1. **Reduce batch size**: `--batch-size 1`
2. **Enable CPU offloading**: `--cpu-offload`
3. **Use gradient accumulation**: `--gradient-accumulation-steps 64`
4. **Enable activation checkpointing**: `--checkpoint-activations`

## Next Steps

After successful installation:

1. **[Read the Training Guide](training.md)** - Learn how to train Ultra-AI models
2. **[Explore Examples](../examples/)** - Try pre-built examples
3. **[Check API Reference](api.md)** - Understand the complete API
4. **[Join the Community](https://github.com/ultra-ai/ultra-ai-model/discussions)** - Get help and share experiences

## Support

If you encounter issues:

1. **Check the [FAQ](faq.md)**
2. **Search [GitHub Issues](https://github.com/ultra-ai/ultra-ai-model/issues)**
3. **Ask in [Discussions](https://github.com/ultra-ai/ultra-ai-model/discussions)**
4. **Contact us at**: support@ultra-ai.com