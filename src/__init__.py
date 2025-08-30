"""
Ultra-AI: Revolutionary Multimodal Model with Ultra-Long Context

A groundbreaking 390B parameter multimodal AI model combining:
- Mamba-2 backbone (70%) with selective state space mechanism
- Advanced attention (20%) with linear scaling to 100M+ tokens  
- Mixture of Experts (8%) with 256 experts and top-2 routing
- Multimodal fusion (2%) supporting text, image, audio, and video

Key Features:
- 100 million token context length with O(L) complexity
- 52B active parameters from 390B total (13.3% efficiency)
- Native multimodal understanding and generation
- Distributed training with DeepSpeed, FSDP, and pipeline parallelism
- Edge deployment with quantization and optimization

Advanced Optimizations:
- Smart gradient checkpointing with adaptive memory management
- Dynamic quantization (INT8/INT4) for production deployment
- KV/State caching for 10-100x faster inference
- Beam search and nucleus sampling generation
- JIT compilation and mixed precision (AMP)
- Parallelized MoE computation with load balancing
"""

from .models import UltraAIModel
from .training import UltraAITrainer, DISTRIBUTED_AVAILABLE
from .utils import (
    UltraAIConfig, load_config, save_config, setup_logging,
    SmartCheckpointer, CheckpointConfig,
    DynamicQuantizer, QuantizationConfig,
    KVCache, MambaStateCache, InferenceCache, CacheConfig,
    GenerationConfig, BeamSearchDecoder, NucleusSampler,
    AdvancedGenerator, ConstrainedGenerator
)

# Import conditionnel du distributed trainer
if DISTRIBUTED_AVAILABLE:
    from .training import DistributedTrainer

__version__ = "1.0.0"
__author__ = "Ultra-AI Team"
__email__ = "contact@ultra-ai.com"

# Liste des exports conditionnels
_base_all = [
    "UltraAIModel",
    "UltraAITrainer", 
    "UltraAIConfig",
    "load_config",
    "save_config", 
    "setup_logging",
    # Advanced optimization classes
    "SmartCheckpointer",
    "CheckpointConfig", 
    "DynamicQuantizer",
    "QuantizationConfig",
    "KVCache",
    "MambaStateCache",
    "InferenceCache",
    "CacheConfig",
    "GenerationConfig",
    "BeamSearchDecoder",
    "NucleusSampler",
    "AdvancedGenerator",
    "ConstrainedGenerator"
]

if DISTRIBUTED_AVAILABLE:
    __all__ = _base_all + ["DistributedTrainer"]
else:
    __all__ = _base_all