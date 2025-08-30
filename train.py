#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTRA-AI TRAINING SCRIPT - VERSION PUISSANTE
========================================
Script d'entraînement ultra-optimisé exploitant toutes les capacités:
- Architecture hybride Mamba-2 + Attention + MoE + Multimodal
- Optimisations mémoire avancées et quantification dynamique  
- Monitoring en temps réel et checkpointing intelligent
- Support multi-GPU et distributed training
- Gestion automatique des erreurs et recovery
"""

import os
import sys
import argparse
import logging
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import signal
import traceback

# Core imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# ML Framework imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available - logging to file only")

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("WARNING: accelerate not available - single GPU only")

# DeepSpeed import (Windows compatible)
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("WARNING: DeepSpeed not available (normal on Windows) - using PyTorch native distributed")

# Add project source to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Ultra-AI imports
from src.models.ultra_ai_model import UltraAIModel
from src.utils.config import UltraAIConfig, load_config
from src.utils.logger import setup_logging
from src.utils.monitoring import MetricsTracker, PerformanceMonitor
from src.utils.checkpointing import CheckpointManager
from src.utils.memory import MemoryManager
from src.utils.quantization import DynamicQuantizer
from src.training.trainer import UltraAITrainer
try:
    from src.training.distributed import DistributedTrainer
    DISTRIBUTED_TRAINER_AVAILABLE = True
except ImportError:
    DISTRIBUTED_TRAINER_AVAILABLE = False
    print("WARNING: Distributed trainer not available - single GPU only")
    DistributedTrainer = None
from src.training.data_loader import UltraDataset, MultimodalDataLoader, ProcessedDataset
from src.training.optimizer import get_optimizer
from src.training.loss import UltraAILoss

logger = logging.getLogger(__name__)

class UltraPowerTrainer:
    """
    Trainer ultra-puissant avec toutes les optimisations avancées.
    Gère automatiquement la complexité de l'entraînement distribuée.
    """
    
    def __init__(self, config_path: str, args: argparse.Namespace):
        self.args = args
        self.start_time = time.time()
        
        # Load configuration
        self.config = self._load_and_validate_config(config_path)
        
        # Initialize systems
        self.device = self._setup_device()
        self.accelerator = self._setup_accelerator()
        self.logger = self._setup_logging()
        
        # Core components
        self.model = None
        self.trainer = None
        self.checkpoint_manager = None
        self.metrics_tracker = None
        self.memory_manager = None
        
        # Training state
        self.training_state = {
            'epoch': 0,
            'step': 0,
            'best_loss': float('inf'),
            'training_time': 0,
            'last_checkpoint': None
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("🚀 Ultra-AI Power Trainer Initialized!")
        
    def _load_and_validate_config(self, config_path: str) -> UltraAIConfig:
        """Load and validate training configuration."""
        try:
            # Check if it's a predefined config name
            if hasattr(self.args, 'config_name') and self.args.config_name:
                config = load_config(config_name=self.args.config_name)
            else:
                config = load_config(config_path)
            
            # Auto-adjust for local hardware
            if self.args.auto_config:
                config = self._auto_configure_for_hardware(config)
                
            # Validate configuration
            self._validate_config(config)
            
            return config
            
        except Exception as e:
            print(f"❌ Configuration Error: {e}")
            sys.exit(1)
    
    def _auto_configure_for_hardware(self, config: UltraAIConfig) -> UltraAIConfig:
        """Auto-adjust configuration for available hardware."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_name = torch.cuda.get_device_name(0)
            
            print(f"🔧 Auto-configuring for {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Adjust batch size based on GPU memory
            if gpu_memory < 12:  # <12GB
                config.batch_size = min(config.batch_size, 1)
                config.gradient_accumulation_steps = max(config.gradient_accumulation_steps, 64)
                config.max_seq_length = min(config.max_seq_length, 1024)
            elif gpu_memory < 24:  # 12-24GB
                config.batch_size = min(config.batch_size, 2)
                config.gradient_accumulation_steps = max(config.gradient_accumulation_steps, 32)
                config.max_seq_length = min(config.max_seq_length, 2048)
            else:  # 24GB+
                config.batch_size = min(config.batch_size, 4)
                config.gradient_accumulation_steps = max(config.gradient_accumulation_steps, 16)
                
        return config
    
    def _validate_config(self, config: UltraAIConfig):
        """Validate configuration for consistency."""
        issues = []
        
        # Check architecture ratios
        total_ratio = config.mamba_ratio + config.attention_ratio + config.moe_ratio + config.multimodal_ratio
        if abs(total_ratio - 1.0) > 0.01:
            issues.append(f"Architecture ratios sum to {total_ratio:.3f}, should be 1.0")
            
        # Check data paths - look for data directory or manifest file
        if hasattr(config, 'train_data_path'):
            data_path = Path(config.train_data_path)
            if not data_path.exists():
                # Try fallback paths
                fallback_paths = [Path("./data"), Path("data")]
                data_path_exists = any(p.exists() for p in fallback_paths)
                if not data_path_exists:
                    issues.append(f"Training data path not found: {config.train_data_path} (also tried ./data and data)")
            
        if issues:
            raise ValueError("Configuration issues:\n" + "\n".join(f"  • {issue}" for issue in issues))
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device configuration."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.local_rank}" if hasattr(self.args, 'local_rank') else "cuda:0")
            torch.cuda.set_device(device)
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            print(f"🔥 GPU Ready: {torch.cuda.get_device_name(device)} ({torch.cuda.get_device_properties(device).total_memory // (1024**3)}GB)")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU - Performance will be limited")
            
        return device
    
    def _setup_accelerator(self) -> Optional[Accelerator]:
        """Setup Accelerate for distributed training."""
        if self.args.use_accelerate and ACCELERATE_AVAILABLE:
            try:
                accelerator = Accelerator(
                    gradient_accumulation_steps=getattr(self.config, 'gradient_accumulation_steps', 32),
                    mixed_precision=getattr(self.config, 'mixed_precision', 'fp16'),
                    log_with="wandb" if self.args.wandb else None,
                )
                print("🚀 Accelerator initialized for distributed training")
                return accelerator
            except Exception as e:
                print(f"⚠️ Accelerator setup failed: {e}")
                return None
        return None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        log_level = "DEBUG" if self.args.debug else "INFO"
        logger = setup_logging(
            log_level=log_level,
            log_dir=self.args.output_dir / "logs",
            enable_file_logging=True,
            enable_console_logging=True
        )
        
        # Log system info
        logger.info("="*60)
        logger.info("ULTRA-AI POWER TRAINER")
        logger.info("="*60)
        logger.info(f"🏗️ Architecture: {self.config.model_name}")
        logger.info(f"📊 Model Size: {self.config.d_model}d, {self.config.vocab_size} vocab")
        logger.info(f"🔄 Max Context: {self.config.max_seq_length:,} tokens")
        logger.info(f"⚡ Device: {self.device}")
        logger.info(f"🎯 Batch Size: {self.config.batch_size} (effective: {self.config.batch_size * getattr(self.config, 'gradient_accumulation_steps', 32)})")
        
        return logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self._save_emergency_checkpoint()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _initialize_model_and_components(self):
        """Initialize model and all training components."""
        self.logger.info("🏗️ Initializing Ultra-AI Model...")
        
        # Create model with all optimizations
        self.model = UltraAIModel(
            config=self.config,
            enable_optimizations=True
        )
        
        # Model info
        memory_info = self.model.get_memory_footprint()
        self.logger.info(f"📊 Model Parameters: {memory_info['total_parameters']:,}")
        self.logger.info(f"💾 Model Memory: {memory_info['model_size_mb']:.1f} MB")
        self.logger.info(f"⚡ Active Parameters: {memory_info['active_parameters']:,}")
        
        # Initialize training components
        self._initialize_training_components()
        
        # Apply model optimizations
        self._apply_model_optimizations()
        
        # Move to device
        if not self.accelerator:
            self.model = self.model.to(self.device)
        
        self.logger.info("✅ Model and components initialized!")
    
    def _initialize_training_components(self):
        """Initialize all training components."""
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(
            window_size=100,
            save_frequency=1000
        )
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(
            monitor_interval=1.0,
            gpu_monitoring=torch.cuda.is_available(),
            memory_monitoring=True
        )
        
        # Memory management
        self.memory_manager = MemoryManager(
            enable_auto_cleanup=True,
            cleanup_threshold=0.8 if self.args.aggressive_memory else 0.85,
            aggressive_cleanup=self.args.aggressive_memory
        )
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.args.output_dir / "checkpoints"),
            max_checkpoints=5,
            async_save=True
        )
    
    def _apply_model_optimizations(self):
        """Apply advanced model optimizations."""
        self.logger.info("🔧 Applying model optimizations...")
        
        # Gradient checkpointing
        if getattr(self.config, 'gradient_checkpointing', True):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                self.logger.info("✅ Gradient checkpointing enabled")
        
        # Model compilation (PyTorch 2.0+)
        if self.args.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(
                    self.model, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                self.logger.info("✅ Model compiled with PyTorch 2.0")
            except Exception as e:
                self.logger.warning(f"⚠️ Model compilation failed: {e}")
        
        # Quantization
        if self.args.quantize:
            try:
                quantizer = DynamicQuantizer()
                self.model = quantizer.quantize_model(self.model)
                self.logger.info("✅ Dynamic quantization applied")
            except Exception as e:
                self.logger.warning(f"⚠️ Quantization failed: {e}")
    
    def _create_datasets_and_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create optimized datasets and data loaders."""
        self.logger.info("📚 Creating datasets and data loaders...")
        
        try:
            # Load processed data manifest
            data_path = Path(self.config.train_data_path)
            if not data_path.exists():
                # Essayer d'abord les données préprocessées
                processed_path = Path("./data/processed")
                if processed_path.exists():
                    data_path = processed_path
                else:
                    data_path = Path("./data")  # Fallback to default data directory
            
            manifest_path = data_path / "training_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data_manifest = json.load(f)
            
            # Détecter si on a des données préprocessées
            processed_path = Path("./data/processed")
            if processed_path.exists() and (processed_path / "training_manifest.json").exists():
                # Utiliser les données préprocessées
                self.logger.info("Using preprocessed data from data/processed/")
                dataset = ProcessedDataset(processed_data_path=str(processed_path))
            else:
                # Fallback vers UltraDataset pour données brutes
                self.logger.info("Using raw data - consider running preprocess_hf_data.py first")
                dataset = UltraDataset(
                    data_path=str(data_path),
                    tokenizer_name=getattr(self.config, 'tokenizer_name', 'microsoft/DialoGPT-large'),
                    max_seq_length=self.config.max_seq_length,
                    modalities=getattr(self.config, 'modalities', ['text']),
                )
            
            # Create data loader
            train_loader = MultimodalDataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            
            # Validation loader (optional) - pour simplifier, pas de validation pour l'instant
            val_loader = None
            # Note: Pour la validation, on pourrait implémenter un split des données preprocessées
            
            self.logger.info(f"✅ Dataset created: {len(dataset):,} samples")
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"❌ Dataset creation failed: {e}")
            raise
    
    def _create_trainer(self, train_loader: DataLoader, val_loader: Optional[DataLoader]):
        """Create the appropriate trainer (single GPU or distributed)."""
        trainer_kwargs = {
            'model': self.model,
            'train_dataloader': train_loader,
            'val_dataloader': val_loader,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'device': self.device,
            'output_dir': self.args.output_dir,
            'logging_steps': self.args.logging_steps,
            'eval_steps': self.args.eval_steps,
            'save_steps': self.args.save_steps,
            'max_grad_norm': getattr(self.config, 'max_grad_norm', 1.0),
            'mixed_precision': getattr(self.config, 'mixed_precision', 'fp16') == 'fp16',
            'gradient_checkpointing': getattr(self.config, 'gradient_checkpointing', True),
            'use_wandb': self.args.wandb,
        }
        
        # Choose trainer type
        if self.args.distributed and DISTRIBUTED_TRAINER_AVAILABLE and (torch.cuda.device_count() > 1 and not self.accelerator):
            self.logger.info("🌐 Creating Distributed Trainer")
            self.trainer = DistributedTrainer(
                parallelism_strategy=getattr(self.config, 'parallelism_strategy', 'data_parallel'),
                zero_stage=getattr(self.config, 'zero_stage', 2),
                **trainer_kwargs
            )
        else:
            self.logger.info("🎯 Creating Single GPU Trainer")
            self.trainer = UltraAITrainer(**trainer_kwargs)
        
        # Integrate with accelerator if available
        if self.accelerator:
            self.model, self.trainer.optimizer, train_loader, val_loader = self.accelerator.prepare(
                self.model, self.trainer.optimizer, train_loader, val_loader
            )
    
    def _setup_monitoring(self):
        """Setup advanced monitoring and logging."""
        if self.args.wandb and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=getattr(self.config, 'wandb_project', 'ultra-ai-training'),
                    name=f"ultra-ai-{self.args.run_name or time.strftime('%Y%m%d_%H%M%S')}",
                    config=self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                    tags=["ultra-ai", "hybrid-architecture"],
                )
                self.logger.info("📈 Weights & Biases initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ W&B setup failed: {e}")
    
    def _training_loop(self):
        """Main training loop with advanced error handling."""
        self.logger.info("🚀 Starting training loop...")
        
        try:
            # Training preparation
            self.performance_monitor.start_monitoring()
            
            # Resume from checkpoint if specified
            if self.args.resume_from:
                self._load_checkpoint(self.args.resume_from)
            
            # Main training
            start_time = time.time()
            
            for epoch in range(self.training_state['epoch'], self.args.num_epochs):
                self.training_state['epoch'] = epoch
                
                # Affichage début d'époque avec informations détaillées
                print(f"\n🌟 {'='*60} 🌟")
                print(f"📚 STARTING EPOCH {epoch + 1}/{self.args.num_epochs}")
                print(f"🗂️  Dataset: {len(self.trainer.train_dataloader.dataset):,} samples")
                print(f"📦 Batch Size: {self.config.batch_size} | Steps: {len(self.trainer.train_dataloader)}")
                print(f"🌟 {'='*60} 🌟\n")
                
                self.logger.info(f"📚 Epoch {epoch + 1}/{self.args.num_epochs}")
                
                # Epoch training
                epoch_metrics = self._train_epoch()
                
                # Validation
                if self.trainer.val_dataloader and (epoch + 1) % self.args.eval_epochs == 0:
                    val_metrics = self._validate_epoch()
                    epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                
                # Update best metrics
                if epoch_metrics.get('train_loss', float('inf')) < self.training_state['best_loss']:
                    self.training_state['best_loss'] = epoch_metrics['train_loss']
                    self._save_best_checkpoint()
                
                # Log epoch metrics
                self._log_epoch_metrics(epoch, epoch_metrics)
                
                # Memory cleanup
                self.memory_manager.cleanup()
                
            total_time = time.time() - start_time
            self.training_state['training_time'] = total_time
            
            self.logger.info(f"🎉 Training completed! Total time: {total_time/3600:.2f}h")
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Training interrupted by user")
            self._save_emergency_checkpoint()
            raise
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            self.logger.error(traceback.format_exc())
            self._save_emergency_checkpoint()
            raise
        finally:
            self._cleanup_training()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch with comprehensive monitoring."""
        self.model.train()
        epoch_metrics = {'train_loss': 0.0, 'train_steps': 0}
        
        # Progress tracking
        total_steps = len(self.trainer.train_dataloader)
        
        for step, batch in enumerate(self.trainer.train_dataloader):
            try:
                # Training step
                step_metrics = self._training_step(batch, step)
                
                # Update metrics
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                epoch_metrics['train_steps'] += 1
                
                # Logging and checkpointing
                if (step + 1) % self.args.logging_steps == 0:
                    self._log_step_metrics(step, step_metrics)
                
                if (step + 1) % self.args.save_steps == 0:
                    self._save_checkpoint()
                
                # Memory management
                if (step + 1) % 100 == 0:
                    self.memory_manager.optimize()
                
            except Exception as e:
                self.logger.error(f"❌ Training step {step} failed: {e}")
                if self.args.continue_on_error:
                    continue
                else:
                    raise
        
        # Average metrics
        for k in epoch_metrics:
            if k != 'train_steps':
                epoch_metrics[k] /= epoch_metrics['train_steps']
        
        return epoch_metrics
    
    def _training_step(self, batch, step: int) -> Dict[str, float]:
        """Single optimized training step."""
        step_start = time.time()
        
        # Move batch to device
        if not self.accelerator:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass
        if self.accelerator:
            with self.accelerator.autocast():
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
        else:
            if hasattr(self.trainer, 'scaler') and self.trainer.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = self._compute_loss(outputs, batch)
            else:
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs, batch)
        
        # Backward pass
        if self.accelerator:
            self.accelerator.backward(loss)
        else:
            if hasattr(self.trainer, 'scaler') and self.trainer.scaler:
                self.trainer.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Optimizer step
        if (step + 1) % getattr(self.config, 'gradient_accumulation_steps', 1) == 0:
            if self.accelerator:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.trainer.optimizer.step()
                self.trainer.scheduler.step()
                self.trainer.optimizer.zero_grad()
            else:
                if hasattr(self.trainer, 'scaler') and self.trainer.scaler:
                    self.trainer.scaler.unscale_(self.trainer.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.trainer.scaler.step(self.trainer.optimizer)
                    self.trainer.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.trainer.optimizer.step()
                    
                if hasattr(self.trainer, 'scheduler'):
                    self.trainer.scheduler.step()
                self.trainer.optimizer.zero_grad()
        
        # Metrics avec affichage performance
        step_time = time.time() - step_start
        tokens_per_second = batch['input_ids'].numel() / step_time if step_time > 0 else 0
        
        metrics = {
            'train_loss': loss.item(),
            'step_time': step_time,
            'tokens_per_sec': tokens_per_second,
            'learning_rate': self.trainer.optimizer.param_groups[0]['lr'] if hasattr(self.trainer, 'optimizer') else 0,
            'memory_used_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }
        
        # Affichage progression détaillée
        if step % self.args.logging_steps == 0:
            self._log_training_progress(step, metrics, outputs)
        
        # Add model-specific metrics
        if hasattr(outputs, 'router_outputs') and outputs['router_outputs']:
            metrics['moe_loss'] = sum(r.get('loss', 0) for r in outputs['router_outputs']) / len(outputs['router_outputs'])
        
        return metrics
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy loss for language modeling."""
        logits = outputs.get('logits')
        labels = batch.get('labels')
        
        # Try to use input_ids as labels (common for language modeling)
        if labels is None:
            labels = batch.get('input_ids')
        
        if logits is None:
            raise ValueError("Model outputs do not contain 'logits'")
        if labels is None:
            raise ValueError(f"Batch does not contain 'labels' or 'input_ids'. Available keys: {list(batch.keys())}")
        
        # Reshape for cross entropy: (batch_size * seq_len, vocab_size)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss = loss_fct(shift_logits, shift_labels)
        return loss
    
    def _log_training_progress(self, step: int, metrics: Dict[str, float], outputs: Dict[str, torch.Tensor]):
        """Affiche la progression détaillée de l'entraînement."""
        # Calculer le pourcentage d'avancement
        total_steps = len(self.trainer.train_dataloader)
        progress_pct = (step / total_steps) * 100
        
        # Temps écoulé et estimation temps restant
        elapsed_time = time.time() - self.start_time
        if step > 0:
            estimated_total_time = (elapsed_time / step) * total_steps
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0
        
        # Affichage formaté avec émojis et couleurs
        print(f"\n{'='*80}")
        print(f"🚀 ULTRA-AI TRAINING - Step {step}/{total_steps} ({progress_pct:.1f}%)")
        print(f"{'='*80}")
        
        # Métriques principales
        print(f"📊 Loss: {metrics['train_loss']:.6f}")
        print(f"⏱️  Step Time: {metrics['step_time']:.3f}s")
        print(f"🚄 Tokens/sec: {metrics['tokens_per_sec']:.0f}")
        print(f"📚 Learning Rate: {metrics['learning_rate']:.2e}")
        
        if torch.cuda.is_available():
            print(f"💾 GPU Memory: {metrics['memory_used_mb']:.1f} MB")
        
        # Temps
        print(f"⏰ Elapsed: {elapsed_time/60:.1f}min | ETA: {remaining_time/60:.1f}min")
        
        # Métriques modèle spécifiques
        if 'moe_loss' in metrics:
            print(f"🔀 MoE Loss: {metrics['moe_loss']:.6f}")
        
        # Informations sur les tokens
        if 'input_ids' in outputs or hasattr(self.trainer, 'train_dataloader'):
            print(f"🎯 Model: Ultra-AI Edge | Parameters: 101M | Active: 13.5M")
        
        print(f"{'='*80}\n")
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate model performance."""
        self.model.eval()
        val_metrics = {'val_loss': 0.0, 'val_steps': 0}
        
        with torch.no_grad():
            for batch in self.trainer.val_dataloader:
                if not self.accelerator:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.get('loss', outputs.get('logits', None))
                
                val_metrics['val_loss'] += loss.item()
                val_metrics['val_steps'] += 1
        
        # Average validation metrics
        for k in val_metrics:
            if k != 'val_steps':
                val_metrics[k] /= val_metrics['val_steps']
        
        return val_metrics
    
    def _log_step_metrics(self, step: int, metrics: Dict[str, float]):
        """Log step-level metrics."""
        self.logger.info(
            f"Step {step:6d} | "
            f"Loss: {metrics.get('train_loss', 0):.4f} | "
            f"LR: {metrics.get('learning_rate', 0):.2e} | "
            f"Time: {metrics.get('step_time', 0):.3f}s"
        )
        
        if self.args.wandb:
            wandb.log(metrics, step=step)
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics."""
        self.logger.info(f"Epoch {epoch + 1} Summary:")
        for k, v in metrics.items():
            if isinstance(v, float):
                self.logger.info(f"  {k}: {v:.6f}")
        
        if self.args.wandb:
            wandb.log(metrics, step=epoch)
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_data = {
            'epoch': self.training_state['epoch'],
            'step': self.training_state['step'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict() if hasattr(self.trainer, 'optimizer') else {},
            'scheduler_state_dict': self.trainer.scheduler.state_dict() if hasattr(self.trainer, 'scheduler') else {},
            'training_state': self.training_state,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(checkpoint_data)
        self.training_state['last_checkpoint'] = str(checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_best_checkpoint(self):
        """Save best model checkpoint."""
        best_path = self.args.output_dir / "best_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'training_state': self.training_state,
        }, best_path)
        self.logger.info(f"🏆 Best model saved: {best_path}")
    
    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint on interruption."""
        try:
            emergency_path = self.args.output_dir / "emergency_checkpoint.pt"
            torch.save({
                'model_state_dict': self.model.state_dict() if self.model else {},
                'training_state': self.training_state,
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            }, emergency_path)
            self.logger.info(f"🚨 Emergency checkpoint saved: {emergency_path}")
        except Exception as e:
            self.logger.error(f"❌ Emergency save failed: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if hasattr(self.trainer, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if hasattr(self.trainer, 'scheduler') and 'scheduler_state_dict' in checkpoint:
                self.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.training_state.update(checkpoint.get('training_state', {}))
            
            self.logger.info(f"📂 Checkpoint loaded: {checkpoint_path}")
            self.logger.info(f"   Resuming from epoch {self.training_state['epoch']}, step {self.training_state['step']}")
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint loading failed: {e}")
            raise
    
    def _cleanup_training(self):
        """Cleanup training resources."""
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        if self.memory_manager:
            self.memory_manager.cleanup()
        
        if self.args.wandb:
            wandb.finish()
        
        self.logger.info("🧹 Training cleanup completed")
    
    def train(self):
        """Main training entry point."""
        try:
            # Initialize all components
            self._initialize_model_and_components()
            
            # Create datasets and loaders
            train_loader, val_loader = self._create_datasets_and_loaders()
            
            # Create trainer
            self._create_trainer(train_loader, val_loader)
            
            # Setup monitoring
            self._setup_monitoring()
            
            # Run training
            self._training_loop()
            
            # Final model save
            final_path = self.args.output_dir / "final_model.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
                'training_stats': {
                    'total_time': time.time() - self.start_time,
                    'epochs': self.training_state['epoch'],
                    'best_loss': self.training_state['best_loss'],
                }
            }, final_path)
            
            self.logger.info(f"🎊 Training completed successfully!")
            self.logger.info(f"📁 Final model: {final_path}")
            
        except Exception as e:
            self.logger.error(f"💥 Training failed with error: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self._cleanup_training()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Ultra-AI Power Training Script")
    
    # Core arguments
    parser.add_argument("--config", type=str, default="configs/stage1_text.yaml", help="Configuration file")
    parser.add_argument("--config-name", type=str, choices=["ultra_390b", "ultra_52b_active", "ultra_13b", "ultra_3b", "ultra_edge"], help="Predefined configuration name")
    parser.add_argument("--output-dir", type=str, default="./training_output", help="Output directory")
    parser.add_argument("--run-name", type=str, help="Run name for logging")
    
    # Training parameters
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")
    parser.add_argument("--auto-config", action="store_true", help="Auto-configure for hardware")
    
    # Logging and monitoring
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save every N steps")
    parser.add_argument("--eval-epochs", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Optimization flags
    parser.add_argument("--compile-model", action="store_true", help="Use PyTorch 2.0 compilation")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization")
    parser.add_argument("--aggressive-memory", action="store_true", help="Aggressive memory optimization")
    
    # Distributed training
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
    parser.add_argument("--use-accelerate", action="store_true", help="Use HuggingFace Accelerate")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    
    # System
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue training on step errors")
    
    args = parser.parse_args()
    
    # Setup output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run training
    try:
        trainer = UltraPowerTrainer(args.config, args)
        trainer.train()
        
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        
    except KeyboardInterrupt:
        print("⏹️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Enable multiprocessing support
    mp.set_start_method('spawn', force=True)
    main()