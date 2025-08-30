# API Reference

Complete API documentation for Ultra-AI model classes, methods, and utilities.

## 🏗️ Core Classes

### UltraAIModel

The main Ultra-AI model class implementing the hybrid architecture.

```python
class UltraAIModel(nn.Module):
    """
    Revolutionary 390B parameter multimodal AI model.
    
    Combines Mamba-2, hybrid attention, MoE, and multimodal fusion
    for ultra-long context understanding and generation.
    """
```

#### Constructor

```python
def __init__(self, config: UltraAIConfig):
    """
    Initialize Ultra-AI model.
    
    Args:
        config (UltraAIConfig): Model configuration object
        
    Example:
        >>> config = UltraAIConfig.load("config/ultra-3b.yaml")
        >>> model = UltraAIModel(config)
    """
```

#### Methods

##### forward()

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    multimodal_inputs: Optional[Dict[str, torch.Tensor]] = None,
    return_dict: bool = True
) -> Union[CausalLMOutput, Tuple]:
    """
    Forward pass through Ultra-AI model.
    
    Args:
        input_ids (torch.Tensor): Input token IDs [batch, seq_len]
        attention_mask (torch.Tensor, optional): Attention mask [batch, seq_len]
        multimodal_inputs (dict, optional): Multimodal inputs
            - 'images': Image tensors [batch, channels, height, width]
            - 'audio': Audio tensors [batch, channels, length]
            - 'video': Video tensors [batch, frames, channels, height, width]
        return_dict (bool): Return CausalLMOutput if True, tuple otherwise
        
    Returns:
        CausalLMOutput or tuple containing:
            - logits (torch.Tensor): Output logits [batch, seq_len, vocab_size]
            - hidden_states (torch.Tensor): Hidden states [batch, seq_len, d_model]
            - aux_loss (torch.Tensor): MoE auxiliary loss
            
    Example:
        >>> outputs = model(input_ids=tokens, attention_mask=mask)
        >>> logits = outputs.logits
        >>> loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    """
```

##### generate()

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_length: int = 100,
    max_new_tokens: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    length_penalty: float = 1.0,
    do_sample: bool = True,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    **kwargs
) -> torch.Tensor:
    """
    Generate sequences using various sampling strategies.
    
    Args:
        input_ids (torch.Tensor): Input token IDs [batch, seq_len]
        max_length (int): Maximum total sequence length
        max_new_tokens (int, optional): Maximum new tokens to generate
        temperature (float): Sampling temperature (0.0 = greedy)
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p (nucleus) sampling parameter
        repetition_penalty (float): Repetition penalty factor
        length_penalty (float): Length penalty factor
        do_sample (bool): Whether to use sampling or greedy decoding
        pad_token_id (int, optional): Padding token ID
        eos_token_id (int, optional): End-of-sequence token ID
        
    Returns:
        torch.Tensor: Generated token IDs [batch, generated_length]
        
    Example:
        >>> prompt = "The future of AI is"
        >>> input_ids = tokenizer.encode(prompt, return_tensors="pt")
        >>> generated = model.generate(
        ...     input_ids,
        ...     max_new_tokens=100,
        ...     temperature=0.8,
        ...     top_p=0.9
        ... )
        >>> output = tokenizer.decode(generated[0])
    """
```

##### get_memory_usage()

```python
def get_memory_usage(self) -> Dict[str, float]:
    """
    Get model memory usage statistics.
    
    Returns:
        dict: Memory usage information
            - 'total_params': Total parameters
            - 'active_params': Currently active parameters  
            - 'memory_gb': Memory usage in GB
            - 'efficiency': Parameter efficiency ratio
            
    Example:
        >>> memory_info = model.get_memory_usage()
        >>> print(f"Model uses {memory_info['memory_gb']:.2f} GB")
    """
```

##### save_pretrained()

```python
def save_pretrained(
    self,
    save_directory: str,
    save_config: bool = True,
    save_safetensors: bool = True
):
    """
    Save model and configuration to directory.
    
    Args:
        save_directory (str): Directory to save model files
        save_config (bool): Whether to save configuration  
        save_safetensors (bool): Use safetensors format
        
    Example:
        >>> model.save_pretrained("./ultra-ai-3b-finetuned")
    """
```

##### from_pretrained()

```python
@classmethod
def from_pretrained(
    cls,
    model_name_or_path: str,
    config: Optional[UltraAIConfig] = None,
    **kwargs
) -> "UltraAIModel":
    """
    Load pre-trained model from path or Hub.
    
    Args:
        model_name_or_path (str): Model path or Hub model name
        config (UltraAIConfig, optional): Override configuration
        
    Returns:
        UltraAIModel: Loaded model instance
        
    Example:
        >>> model = UltraAIModel.from_pretrained("ultra-ai/ultra-3b")
        >>> model = UltraAIModel.from_pretrained("./my-finetuned-model")
    """
```

---

### UltraAIConfig

Configuration class for Ultra-AI models.

```python
class UltraAIConfig:
    """
    Configuration for Ultra-AI models.
    
    Contains all hyperparameters and architectural choices.
    """
```

#### Constructor

```python
def __init__(
    self,
    # Model architecture
    d_model: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    d_ff: int = 16384,
    vocab_size: int = 50000,
    max_seq_length: int = 1000000,
    
    # Mamba-2 specific
    d_state: int = 128,
    d_conv: int = 4,
    expand_factor: int = 2,
    
    # MoE specific  
    num_experts: int = 64,
    expert_top_k: int = 2,
    moe_load_balancing_loss_coef: float = 0.01,
    
    # Attention specific
    attention_dropout: float = 0.1,
    linear_attention_threshold: int = 32768,
    
    # Multimodal
    vision_patch_size: int = 16,
    audio_frame_rate: int = 16000,
    video_fps: int = 30,
    
    # Training
    dropout: float = 0.1,
    layer_norm_epsilon: float = 1e-5,
    initializer_range: float = 0.02,
    
    **kwargs
):
    """
    Initialize configuration.
    
    Args:
        d_model: Hidden dimension size
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension  
        vocab_size: Vocabulary size
        max_seq_length: Maximum sequence length
        d_state: Mamba state dimension
        d_conv: Mamba convolution kernel size
        expand_factor: Mamba expansion factor
        num_experts: Number of MoE experts
        expert_top_k: Top-k expert selection
        moe_load_balancing_loss_coef: MoE load balancing coefficient
        
    Example:
        >>> config = UltraAIConfig(
        ...     d_model=2048,
        ...     n_layers=24,
        ...     max_seq_length=500000
        ... )
    """
```

#### Class Methods

##### load()

```python
@classmethod
def load(cls, config_path: str) -> "UltraAIConfig":
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        UltraAIConfig: Loaded configuration
        
    Example:
        >>> config = UltraAIConfig.load("config/ultra-3b.yaml")
    """
```

##### get_model_size()

```python
def get_model_size(self) -> Dict[str, int]:
    """
    Estimate model size and memory requirements.
    
    Returns:
        dict: Model size information
            - 'total_params': Total parameter count
            - 'active_params': Active parameter count
            - 'memory_gb_fp32': Memory usage in FP32
            - 'memory_gb_fp16': Memory usage in FP16
            
    Example:
        >>> size_info = config.get_model_size()
        >>> print(f"Model has {size_info['total_params']:,} parameters")
    """
```

---

## 🏋️ Training Classes

### UltraAITrainer

High-level trainer for Ultra-AI models.

```python
class UltraAITrainer:
    """
    High-level trainer for Ultra-AI models with built-in optimizations.
    """
```

#### Constructor

```python
def __init__(
    self,
    model: UltraAIModel,
    config: TrainingConfig,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    tokenizer: Optional[Any] = None
):
    """
    Initialize trainer.
    
    Args:
        model: Ultra-AI model instance
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        tokenizer: Tokenizer instance
        
    Example:
        >>> trainer = UltraAITrainer(
        ...     model=model,
        ...     config=train_config,
        ...     train_dataloader=train_loader
        ... )
    """
```

#### Methods

##### train()

```python
def train(self) -> Dict[str, Any]:
    """
    Run complete training loop.
    
    Returns:
        dict: Training results and metrics
            - 'final_loss': Final training loss
            - 'total_steps': Total training steps
            - 'training_time': Total training time
            - 'throughput': Average tokens per second
            
    Example:
        >>> results = trainer.train()
        >>> print(f"Training completed in {results['training_time']:.2f}s")
    """
```

##### evaluate()

```python
def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        eval_dataloader: Evaluation data loader
        
    Returns:
        dict: Evaluation metrics
            - 'eval_loss': Average evaluation loss
            - 'perplexity': Model perplexity
            - 'accuracy': Token prediction accuracy
            
    Example:
        >>> eval_results = trainer.evaluate()
        >>> print(f"Perplexity: {eval_results['perplexity']:.2f}")
    """
```

---

### DistributedTrainer

Distributed training wrapper for Ultra-AI.

```python
class DistributedTrainer(UltraAITrainer):
    """
    Distributed training implementation with 3D parallelism support.
    """
```

#### Constructor

```python
def __init__(
    self,
    model: UltraAIModel,
    config: TrainingConfig,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    tokenizer: Optional[Any] = None,
    # Distributed-specific parameters
    data_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    tensor_parallel_size: int = 1
):
    """
    Initialize distributed trainer.
    
    Args:
        model: Ultra-AI model instance
        config: Training configuration
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        tokenizer: Tokenizer instance
        data_parallel_size: Data parallel size
        pipeline_parallel_size: Pipeline parallel size
        tensor_parallel_size: Tensor parallel size
        
    Example:
        >>> distributed_trainer = DistributedTrainer(
        ...     model=model,
        ...     config=train_config,
        ...     train_dataloader=train_loader,
        ...     data_parallel_size=8,
        ...     pipeline_parallel_size=4
        ... )
    """
```

---

## 🎨 Multimodal Classes

### MultimodalProcessor

Process multimodal inputs for Ultra-AI.

```python
class MultimodalProcessor:
    """
    Unified processor for text, image, audio, and video inputs.
    """
```

#### Methods

##### process_batch()

```python
def process_batch(
    self,
    text: Optional[List[str]] = None,
    images: Optional[List[PIL.Image.Image]] = None,
    audio: Optional[List[np.ndarray]] = None,
    videos: Optional[List[np.ndarray]] = None
) -> Dict[str, torch.Tensor]:
    """
    Process multimodal batch into model inputs.
    
    Args:
        text: List of text strings
        images: List of PIL images
        audio: List of audio arrays
        videos: List of video arrays
        
    Returns:
        dict: Processed inputs for model
            - 'input_ids': Text token IDs
            - 'multimodal_inputs': Multimodal tensor dict
            
    Example:
        >>> processor = MultimodalProcessor(config)
        >>> inputs = processor.process_batch(
        ...     text=["Describe this image"],
        ...     images=[PIL.Image.open("image.jpg")]
        ... )
        >>> outputs = model(**inputs)
    """
```

---

## 🛠️ Utility Functions

### Configuration Utilities

```python
def load_config(config_path: str) -> UltraAIConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        UltraAIConfig: Loaded configuration
    """

def save_config(config: UltraAIConfig, save_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object
        save_path: Path to save config
    """
```

### Memory Management

```python
def estimate_memory_usage(config: UltraAIConfig) -> Dict[str, float]:
    """
    Estimate memory usage for given configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        dict: Memory usage estimates
            - 'model_memory_gb': Model parameter memory
            - 'activation_memory_gb': Activation memory
            - 'optimizer_memory_gb': Optimizer state memory
            - 'total_memory_gb': Total estimated memory
    """

def optimize_memory_config(
    target_memory_gb: float,
    base_config: UltraAIConfig
) -> UltraAIConfig:
    """
    Optimize configuration for target memory usage.
    
    Args:
        target_memory_gb: Target memory in GB
        base_config: Base configuration
        
    Returns:
        UltraAIConfig: Optimized configuration
    """
```

### Model Analysis

```python
def analyze_model(model: UltraAIModel) -> Dict[str, Any]:
    """
    Comprehensive model analysis.
    
    Args:
        model: Ultra-AI model instance
        
    Returns:
        dict: Analysis results
            - 'parameter_count': Total parameters
            - 'active_parameters': Active parameters
            - 'component_breakdown': Per-component analysis
            - 'memory_usage': Memory usage statistics
            - 'efficiency_metrics': Model efficiency metrics
    """

def profile_forward_pass(
    model: UltraAIModel,
    input_shape: Tuple[int, int],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Profile model forward pass performance.
    
    Args:
        model: Ultra-AI model instance
        input_shape: Input tensor shape (batch_size, seq_len)
        device: Device to run profiling on
        
    Returns:
        dict: Profiling results
            - 'forward_time_ms': Forward pass time
            - 'memory_usage_gb': Peak memory usage
            - 'flops': Floating point operations
            - 'throughput_tokens_per_sec': Token processing throughput
    """
```

### Logging and Monitoring

```python
def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_wandb: bool = False,
    use_tensorboard: bool = False,
    experiment_name: Optional[str] = None
) -> logging.Logger:
    """
    Setup comprehensive logging for Ultra-AI training.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        use_wandb: Enable Weights & Biases logging
        use_tensorboard: Enable TensorBoard logging
        experiment_name: Experiment name for tracking
        
    Returns:
        logging.Logger: Configured logger instance
    """

class MetricsTracker:
    """
    Track and analyze training metrics.
    """
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at given step."""
        
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot training metrics."""
```

---

## 📊 Data Processing

### MultimodalDataLoader

```python
class MultimodalDataLoader:
    """
    Efficient data loader for multimodal training data.
    """
    
    def __init__(
        self,
        text_datasets: List[Dataset],
        multimodal_datasets: Optional[List[Dataset]] = None,
        batch_size: int = 8,
        sequence_length: int = 1024,
        num_workers: int = 4,
        shuffle: bool = True
    ):
        """
        Initialize multimodal data loader.
        
        Args:
            text_datasets: List of text datasets
            multimodal_datasets: List of multimodal datasets
            batch_size: Batch size
            sequence_length: Maximum sequence length
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle data
        """
```

---

## 🔧 Advanced Features

### Quantization

```python
def quantize_model(
    model: UltraAIModel,
    quantization_type: str = "int8",
    calibration_dataset: Optional[DataLoader] = None
) -> UltraAIModel:
    """
    Quantize Ultra-AI model for efficient deployment.
    
    Args:
        model: Ultra-AI model to quantize
        quantization_type: Type of quantization (int8, int4, fp16)
        calibration_dataset: Dataset for calibration
        
    Returns:
        UltraAIModel: Quantized model
    """
```

### Knowledge Distillation

```python
def distill_model(
    teacher_model: UltraAIModel,
    student_config: UltraAIConfig,
    train_dataloader: DataLoader,
    temperature: float = 3.0,
    alpha: float = 0.7
) -> UltraAIModel:
    """
    Distill large Ultra-AI model into smaller student model.
    
    Args:
        teacher_model: Large teacher model
        student_config: Configuration for student model  
        train_dataloader: Training data
        temperature: Distillation temperature
        alpha: Weight for distillation loss
        
    Returns:
        UltraAIModel: Trained student model
    """
```

---

## 🎯 Examples

### Basic Usage

```python
from ultra_ai_model import UltraAIModel, UltraAIConfig

# Load model
config = UltraAIConfig.load("config/ultra-3b.yaml")
model = UltraAIModel(config)

# Generate text
prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

output_text = tokenizer.decode(generated[0])
print(output_text)
```

### Training Example

```python
from ultra_ai_model import UltraAIModel, UltraAITrainer, TrainingConfig

# Setup training
config = UltraAIConfig.load("config/ultra-3b.yaml")
model = UltraAIModel(config)

train_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=8,
    max_steps=10000,
    gradient_checkpointing=True
)

# Create trainer
trainer = UltraAITrainer(
    model=model,
    config=train_config,
    train_dataloader=train_loader
)

# Train model
results = trainer.train()
print(f"Training completed! Final loss: {results['final_loss']:.4f}")
```

### Multimodal Example

```python
from ultra_ai_model import UltraAIModel, MultimodalProcessor
import PIL.Image

# Setup multimodal processing
processor = MultimodalProcessor(config)
model = UltraAIModel.from_pretrained("ultra-ai/ultra-3b-multimodal")

# Process multimodal input
image = PIL.Image.open("example.jpg")
text = "What do you see in this image?"

inputs = processor.process_batch(
    text=[text],
    images=[image]
)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0])
print(response)
```

This API reference provides comprehensive documentation for all Ultra-AI classes, methods, and utilities, enabling developers to effectively use and extend the Ultra-AI model framework.