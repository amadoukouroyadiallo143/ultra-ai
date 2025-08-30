import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class UltraAIConfig:
    """
    Configuration class for Ultra-AI model with all hyperparameters.
    Supports the revolutionary hybrid architecture specification.
    """
    
    # Model architecture
    model_name: str = "ultra-ai-390b"
    d_model: int = 2560
    vocab_size: int = 50432
    max_seq_length: int = 100_000_000  # 100M tokens context
    
    # Hybrid architecture ratios
    mamba_ratio: float = 0.70  # 70% Mamba-2
    attention_ratio: float = 0.20  # 20% Advanced attention
    moe_ratio: float = 0.08  # 8% MoE
    multimodal_ratio: float = 0.02  # 2% Multimodal fusion
    
    # Mamba-2 configuration
    mamba_layers: int = 56
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: str = "auto"
    
    # Attention configuration
    attention_layers: int = 8  # Integrated with Mamba (ratio 1:7)
    attention_heads: int = 16
    attention_type: str = "linear"  # linear, cca, in_attention
    
    # MoE configuration
    moe_layers: int = 4
    num_experts: int = 256
    moe_top_k: int = 2
    expert_capacity_factor: float = 1.25
    moe_balance_loss_weight: float = 0.01
    
    # Multimodal configuration
    modalities: list = None  # Will be set in __post_init__
    image_size: int = 224
    audio_sample_rate: int = 16000
    video_frames: int = 8
    max_image_tokens: int = 1024
    max_audio_tokens: int = 1500
    max_video_tokens: int = 512
    
    # Training configuration
    batch_size: int = 1  # Ultra-long sequences require small batches
    gradient_accumulation_steps: int = 64  # Effective batch size = 64
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer configuration
    optimizer: str = "adamw"  # adamw, lion, sophia, galore
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler configuration
    scheduler: str = "cosine"
    warmup_steps: int = 2000
    num_epochs: int = 3
    
    # Training efficiency
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    compile_model: bool = True
    
    # Distributed training
    parallelism_strategy: str = "deepspeed"  # data_parallel, model_parallel, pipeline_parallel, deepspeed, fsdp
    zero_stage: int = 2
    cpu_offload: bool = True
    
    # Data configuration
    train_data_path: str = "./data"
    val_data_path: str = "./data"
    tokenizer_name: str = "microsoft/DialoGPT-large"
    sequence_bucketing: bool = True
    max_tokens_per_batch: int = 1_000_000  # 1M tokens per batch
    
    # Checkpointing and logging
    output_dir: str = "./checkpoints"
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 5000
    max_checkpoints: int = 5
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "ultra-ai-model"
    monitor_memory: bool = True
    profile_training: bool = False
    
    # Quantization and optimization
    use_quantization: bool = False
    quantization_method: str = "qat"  # qat, ptq, qlora
    quantization_bits: int = 8
    enable_quantization: bool = False  # Dynamic quantization
    quantization_precision: str = "int8"  # int8, int4, fp16
    
    # Smart checkpointing
    enable_smart_checkpointing: bool = True
    adaptive_checkpointing: bool = True
    memory_threshold: float = 0.85
    mamba_checkpoint_ratio: float = 0.5
    attention_checkpoint_ratio: float = 0.3
    moe_checkpoint_ratio: float = 0.7
    
    # Advanced generation
    max_length: int = 100
    enable_beam_search: bool = True
    enable_nucleus_sampling: bool = True
    default_generation_method: str = "nucleus"  # greedy, nucleus, beam
    
    # Inference optimization
    enable_kv_cache: bool = True
    enable_mamba_cache: bool = True
    enable_moe_cache: bool = True
    cache_dtype: str = "fp16"  # fp32, fp16, bf16
    cache_memory_fraction: float = 0.8
    
    # JIT compilation
    enable_jit: bool = True
    jit_compile_forward: bool = True
    
    # Mixed precision optimization
    enable_amp: bool = True
    enable_tf32: bool = True
    
    # Continual learning
    continual_learning: bool = True
    catastrophic_forgetting_prevention: str = "ewc"  # ewc, l2, replay
    
    # Security and safety
    gradient_clipping: bool = True
    differential_privacy: bool = False
    privacy_noise_multiplier: float = 1.0
    
    # Edge deployment
    edge_optimization: bool = False
    target_device: str = "gpu"  # cpu, gpu, mobile, edge
    model_compression: str = "none"  # none, pruning, distillation, both
    
    # Token IDs for generation
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Top-k parameter for MoE experts
    expert_top_k: int = 2
    
    def __post_init__(self):
        """Initialize default values that depend on other parameters."""
        if self.modalities is None:
            self.modalities = ["text", "image", "audio", "video"]
            
        # Calculate effective model size
        self.total_parameters = self._calculate_total_parameters()
        self.active_parameters = self._calculate_active_parameters()
        
        # Validate configuration
        self._validate_config()
        
    def _calculate_total_parameters(self) -> int:
        """Calculate total model parameters including MoE."""
        # Base model parameters (without MoE)
        base_params = self.d_model * self.vocab_size  # Embedding
        base_params += self.mamba_layers * (self.d_model ** 2) * 8  # Mamba layers
        base_params += self.attention_layers * (self.d_model ** 2) * 4  # Attention layers
        
        # MoE parameters
        moe_params = self.num_experts * (self.d_model ** 2) * 8 * self.moe_layers
        
        total = base_params + moe_params
        return int(total)  # ~390B parameters
        
    def _calculate_active_parameters(self) -> int:
        """Calculate active parameters during inference."""
        # Base model (always active)
        active = self.d_model * self.vocab_size
        active += self.mamba_layers * (self.d_model ** 2) * 8
        active += self.attention_layers * (self.d_model ** 2) * 4
        
        # Only top-k experts are active
        active_experts = self.moe_top_k * (self.d_model ** 2) * 8 * self.moe_layers
        
        total = active + active_experts
        return int(total)  # ~52B active parameters
        
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check architecture ratios sum to 1.0
        total_ratio = (self.mamba_ratio + self.attention_ratio + 
                      self.moe_ratio + self.multimodal_ratio)
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Architecture ratios sum to {total_ratio:.3f}, not 1.0")
            
        # Check sequence length feasibility
        if self.max_seq_length > 100_000_000:
            logger.warning(f"Sequence length {self.max_seq_length:,} may exceed memory limits")
            
        # Check batch size with sequence length
        total_tokens = self.batch_size * self.max_seq_length
        if total_tokens > 100_000_000:  # 100M tokens per forward pass
            logger.warning(f"Total tokens per batch {total_tokens:,} may cause OOM")
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UltraAIConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
        
    def save(self, path: Union[str, Path], format: str = "yaml"):
        """Save configuration to file."""
        path = Path(path)
        config_dict = self.to_dict()
        
        if format.lower() == "yaml":
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Saved configuration to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'UltraAIConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        logger.info(f"Loaded configuration from {path}")
        return cls.from_dict(config_dict)
        
    def get_model_size_info(self) -> Dict[str, Any]:
        """Get detailed model size information."""
        return {
            "total_parameters": f"{self.total_parameters / 1e9:.1f}B",
            "active_parameters": f"{self.active_parameters / 1e9:.1f}B",
            "parameter_efficiency": f"{self.active_parameters / self.total_parameters * 100:.1f}%",
            "architecture_breakdown": {
                "mamba_layers": f"{self.mamba_ratio * 100:.1f}%",
                "attention_layers": f"{self.attention_ratio * 100:.1f}%",
                "moe_layers": f"{self.moe_ratio * 100:.1f}%",
                "multimodal_fusion": f"{self.multimodal_ratio * 100:.1f}%",
            },
            "context_capacity": f"{self.max_seq_length:,} tokens",
            "modalities": self.modalities,
        }
        
    def get_training_info(self) -> Dict[str, Any]:
        """Get training configuration summary."""
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        tokens_per_batch = effective_batch_size * self.max_seq_length
        
        return {
            "effective_batch_size": effective_batch_size,
            "tokens_per_batch": f"{tokens_per_batch:,}",
            "optimizer": self.optimizer,
            "learning_rate": f"{self.learning_rate:.2e}",
            "parallelism": self.parallelism_strategy,
            "zero_stage": self.zero_stage if self.parallelism_strategy == "deepspeed" else "N/A",
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
        }
        
    def estimate_memory_usage(self, precision: str = "fp16") -> Dict[str, str]:
        """Estimate memory usage for training and inference."""
        bytes_per_param = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
        param_bytes = bytes_per_param.get(precision, 2)
        
        # Model parameters
        model_memory = self.active_parameters * param_bytes
        
        # Gradients (same size as parameters in training)
        gradient_memory = model_memory
        
        # Optimizer states (AdamW: 2x parameters for momentum and variance)
        optimizer_memory = model_memory * 2
        
        # Activations (depends on sequence length and batch size)
        activation_memory = (self.batch_size * self.max_seq_length * 
                           self.d_model * param_bytes * 10)  # Rough estimate
        
        total_training = model_memory + gradient_memory + optimizer_memory + activation_memory
        total_inference = model_memory + activation_memory * 0.1  # Much less activation memory
        
        return {
            "model_parameters": f"{model_memory / 1e9:.1f} GB",
            "gradients": f"{gradient_memory / 1e9:.1f} GB",
            "optimizer_states": f"{optimizer_memory / 1e9:.1f} GB",
            "activations": f"{activation_memory / 1e9:.1f} GB",
            "total_training": f"{total_training / 1e9:.1f} GB",
            "total_inference": f"{total_inference / 1e9:.1f} GB",
            "precision": precision,
        }


# Predefined configurations
ULTRA_AI_CONFIGS = {
    "ultra_390b": UltraAIConfig(
        # Enable all advanced optimizations for flagship model
        enable_smart_checkpointing=True,
        adaptive_checkpointing=True,
        memory_threshold=0.85,
        mamba_checkpoint_ratio=0.5,
        attention_checkpoint_ratio=0.3,
        moe_checkpoint_ratio=0.7,
        enable_kv_cache=True,
        enable_mamba_cache=True,
        enable_moe_cache=True,
        cache_dtype="fp16",
        enable_jit=True,
        enable_amp=True,
        enable_tf32=True,
    ),  # Default 390B parameter model
    
    "ultra_52b_active": UltraAIConfig(
        model_name="ultra-ai-52b-active",
        d_model=2048,
        mamba_layers=48,
        attention_layers=6,
        num_experts=128,
        max_seq_length=50_000_000,  # 50M tokens
        # Advanced optimizations for large model
        enable_smart_checkpointing=True,
        adaptive_checkpointing=True,
        memory_threshold=0.8,
        mamba_checkpoint_ratio=0.6,
        attention_checkpoint_ratio=0.4,
        moe_checkpoint_ratio=0.8,
        enable_kv_cache=True,
        enable_mamba_cache=True,
        enable_moe_cache=True,
        cache_dtype="fp16",
        enable_jit=True,
        enable_amp=True,
        enable_tf32=True,
    ),
    
    "ultra_13b": UltraAIConfig(
        model_name="ultra-ai-13b",
        d_model=1536,
        mamba_layers=32,
        attention_layers=4,
        num_experts=64,
        max_seq_length=10_000_000,  # 10M tokens
        # Optimizations for 13B model
        enable_smart_checkpointing=True,
        adaptive_checkpointing=True,
        mamba_checkpoint_ratio=0.5,
        attention_checkpoint_ratio=0.3,
        moe_checkpoint_ratio=0.7,
        enable_kv_cache=True,
        enable_mamba_cache=True,
        enable_moe_cache=True,
        cache_dtype="fp16",
        enable_jit=True,
        enable_amp=True,
        enable_tf32=True,
    ),
    
    "ultra_3b": UltraAIConfig(
        model_name="ultra-ai-3b",
        d_model=1024,
        mamba_layers=24,
        attention_layers=3,
        num_experts=32,
        max_seq_length=1_000_000,  # 1M tokens
        # Optimizations for 3B model
        enable_smart_checkpointing=True,
        adaptive_checkpointing=True,
        mamba_checkpoint_ratio=0.5,
        attention_checkpoint_ratio=0.3,
        moe_checkpoint_ratio=0.7,
        enable_kv_cache=True,
        enable_mamba_cache=True,
        enable_moe_cache=True,
        cache_dtype="fp16",
        enable_jit=True,
        enable_amp=True,
        enable_tf32=True,
    ),
    
    "ultra_edge": UltraAIConfig(
        model_name="ultra-ai-edge",
        d_model=512,
        mamba_layers=12,
        attention_layers=2,
        num_experts=8,
        max_seq_length=512, 
        batch_size=8,  # Augmenté de 1 à 8
        gradient_accumulation_steps=8,  # Batch effectif = 64
        edge_optimization=True,
        use_quantization=True,
        quantization_bits=8,
        mixed_precision=True,
        # Advanced optimizations for edge deployment
        enable_quantization=True,
        quantization_precision="int8",
        enable_smart_checkpointing=True,
        adaptive_checkpointing=True,
        mamba_checkpoint_ratio=0.7,
        attention_checkpoint_ratio=0.5,
        moe_checkpoint_ratio=0.8,
        enable_kv_cache=True,
        enable_mamba_cache=True,
        enable_moe_cache=True,
        cache_dtype="fp16",
        enable_jit=True,
        enable_amp=True,
        enable_tf32=True,
    ),
}


def load_config(config_path: Optional[str] = None, config_name: Optional[str] = None) -> UltraAIConfig:
    """
    Load configuration from file or predefined config.
    
    Args:
        config_path: Path to configuration file
        config_name: Name of predefined configuration
        
    Returns:
        UltraAIConfig instance
    """
    if config_path:
        return UltraAIConfig.load(config_path)
    elif config_name:
        if config_name not in ULTRA_AI_CONFIGS:
            available = list(ULTRA_AI_CONFIGS.keys())
            raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
        return ULTRA_AI_CONFIGS[config_name]
    else:
        return UltraAIConfig()  # Default configuration


def save_config(config: UltraAIConfig, path: str, format: str = "yaml"):
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        path: Output file path
        format: File format (yaml or json)
    """
    config.save(path, format)


def create_config_from_args(args) -> UltraAIConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        UltraAIConfig instance
    """
    # Start with default or named config
    if hasattr(args, 'config_name') and args.config_name:
        config = load_config(config_name=args.config_name)
    elif hasattr(args, 'config_path') and args.config_path:
        config = load_config(config_path=args.config_path)
    else:
        config = UltraAIConfig()
        
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
            
    return config


def print_config_summary(config: UltraAIConfig):
    """Print a formatted summary of the configuration."""
    print("=" * 80)
    print(f"Ultra-AI Model Configuration: {config.model_name}")
    print("=" * 80)
    
    # Model architecture
    model_info = config.get_model_size_info()
    print("\nModel Architecture:")
    print(f"  Total Parameters: {model_info['total_parameters']}")
    print(f"  Active Parameters: {model_info['active_parameters']}")
    print(f"  Parameter Efficiency: {model_info['parameter_efficiency']}")
    print(f"  Context Length: {model_info['context_capacity']}")
    print(f"  Modalities: {', '.join(model_info['modalities'])}")
    
    # Architecture breakdown
    print("\nArchitecture Breakdown:")
    for component, percentage in model_info['architecture_breakdown'].items():
        print(f"  {component.replace('_', ' ').title()}: {percentage}")
        
    # Training configuration
    training_info = config.get_training_info()
    print("\nTraining Configuration:")
    for key, value in training_info.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
        
    # Memory estimation
    memory_info = config.estimate_memory_usage()
    print("\nMemory Usage Estimation (FP16):")
    for key, value in memory_info.items():
        if key != "precision":
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
    print("=" * 80)