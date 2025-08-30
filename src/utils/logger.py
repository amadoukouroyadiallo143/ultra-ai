import logging
import logging.handlers
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time
from datetime import datetime
import torch
import numpy as np


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
    log_dir: str = "./logs",
) -> logging.Logger:
    """
    Setup comprehensive logging for Ultra-AI training.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file name
        log_format: Custom log format
        enable_file_logging: Enable file logging
        enable_console_logging: Enable console logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        log_dir: Directory to store log files
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Default log format
    if log_format is None:
        log_format = (
            "%(asctime)s | %(name)s | %(levelname)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if enable_file_logging:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"ultra_ai_{timestamp}.log"
            
        log_file_path = log_dir / log_file
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    # Create ultra-ai specific logger
    ultra_logger = logging.getLogger("ultra_ai")
    
    # Add training metrics handler
    metrics_handler = TrainingMetricsHandler(log_dir / "training_metrics.json")
    ultra_logger.addHandler(metrics_handler)
    
    ultra_logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    
    return ultra_logger


class TrainingMetricsHandler(logging.Handler):
    """
    Custom logging handler for training metrics.
    Captures and stores structured training data.
    """
    
    def __init__(self, metrics_file: str):
        super().__init__()
        self.metrics_file = Path(metrics_file)
        self.metrics_data = []
        self.last_flush = time.time()
        self.flush_interval = 30  # Flush every 30 seconds
        
    def emit(self, record: logging.LogRecord):
        """Emit a log record."""
        try:
            # Only capture training-related logs
            if hasattr(record, 'metrics') and record.metrics:
                metrics_entry = {
                    'timestamp': record.created,
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'metrics': record.metrics,
                }
                
                if hasattr(record, 'step'):
                    metrics_entry['step'] = record.step
                if hasattr(record, 'epoch'):
                    metrics_entry['epoch'] = record.epoch
                    
                self.metrics_data.append(metrics_entry)
                
                # Periodic flush
                current_time = time.time()
                if current_time - self.last_flush > self.flush_interval:
                    self.flush_metrics()
                    self.last_flush = current_time
                    
        except Exception:
            self.handleError(record)
            
    def flush_metrics(self):
        """Flush metrics data to file."""
        if self.metrics_data:
            try:
                # Load existing data
                existing_data = []
                if self.metrics_file.exists():
                    with open(self.metrics_file, 'r') as f:
                        existing_data = json.load(f)
                        
                # Append new data
                existing_data.extend(self.metrics_data)
                
                # Save to file
                with open(self.metrics_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                    
                self.metrics_data.clear()
                
            except Exception as e:
                print(f"Error flushing metrics: {e}")
                
    def close(self):
        """Close handler and flush remaining data."""
        self.flush_metrics()
        super().close()


class DistributedLogger:
    """
    Distributed logging coordinator for multi-process training.
    Ensures proper log coordination across processes.
    """
    
    def __init__(self, rank: int, world_size: int, log_dir: str = "./logs"):
        self.rank = rank
        self.world_size = world_size
        self.log_dir = Path(log_dir)
        self.is_main_process = rank == 0
        
        # Setup process-specific logging
        self.logger = self._setup_distributed_logging()
        
    def _setup_distributed_logging(self) -> logging.Logger:
        """Setup logging for distributed process."""
        # Create rank-specific log file
        log_file = f"ultra_ai_rank_{self.rank}.log"
        
        # Only main process logs to console
        enable_console = self.is_main_process
        
        logger = setup_logging(
            log_file=log_file,
            log_dir=str(self.log_dir),
            enable_console_logging=enable_console,
        )
        
        logger.info(f"Distributed logger initialized - Rank: {self.rank}/{self.world_size}")
        
        return logger
        
    def log_metrics(self, metrics: Dict[str, Any], step: int, epoch: int = 0):
        """Log metrics with distributed coordination."""
        if self.is_main_process:
            # Create log record with metrics
            record = logging.LogRecord(
                name="ultra_ai.metrics",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Training metrics",
                args=(),
                exc_info=None,
            )
            
            # Add custom attributes
            record.metrics = metrics
            record.step = step
            record.epoch = epoch
            
            self.logger.handle(record)
            
    def aggregate_logs(self) -> Optional[Dict[str, Any]]:
        """Aggregate logs from all processes (main process only)."""
        if not self.is_main_process:
            return None
            
        aggregated_data = {}
        
        for rank in range(self.world_size):
            rank_log_file = self.log_dir / f"ultra_ai_rank_{rank}.log"
            rank_metrics_file = self.log_dir / f"training_metrics_rank_{rank}.json"
            
            if rank_metrics_file.exists():
                try:
                    with open(rank_metrics_file, 'r') as f:
                        rank_data = json.load(f)
                    aggregated_data[f"rank_{rank}"] = rank_data
                except Exception as e:
                    self.logger.warning(f"Failed to load metrics from rank {rank}: {e}")
                    
        return aggregated_data


class WandbLogger:
    """
    Weights & Biases integration for Ultra-AI training.
    """
    
    def __init__(
        self,
        project: str = "ultra-ai-model",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        
        if not self.enabled:
            return
            
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize run
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                tags=tags,
                reinit=True,
            )
            
            logging.getLogger("ultra_ai").info(f"Weights & Biases initialized: {self.run.url}")
            
        except ImportError:
            logging.getLogger("ultra_ai").warning(
                "wandb not available. Install with: pip install wandb"
            )
            self.enabled = False
        except Exception as e:
            logging.getLogger("ultra_ai").error(f"Failed to initialize wandb: {e}")
            self.enabled = False
            
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb."""
        if not self.enabled:
            return
            
        try:
            self.wandb.log(metrics, step=step)
        except Exception as e:
            logging.getLogger("ultra_ai").warning(f"Failed to log to wandb: {e}")
            
    def log_model(self, model_path: str, name: str = "ultra_ai_model"):
        """Log model artifact to wandb."""
        if not self.enabled:
            return
            
        try:
            artifact = self.wandb.Artifact(name=name, type="model")
            artifact.add_dir(model_path)
            self.run.log_artifact(artifact)
        except Exception as e:
            logging.getLogger("ultra_ai").warning(f"Failed to log model to wandb: {e}")
            
    def finish(self):
        """Finish wandb run."""
        if self.enabled and hasattr(self, 'run'):
            self.run.finish()


class TensorBoardLogger:
    """
    TensorBoard integration for Ultra-AI training.
    """
    
    def __init__(self, log_dir: str = "./tensorboard_logs", enabled: bool = True):
        self.enabled = enabled
        
        if not self.enabled:
            return
            
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            logging.getLogger("ultra_ai").info(f"TensorBoard initialized: {log_dir}")
            
        except ImportError:
            logging.getLogger("ultra_ai").warning(
                "tensorboard not available. Install with: pip install tensorboard"
            )
            self.enabled = False
            
    def log_scalar(self, tag: str, scalar_value: float, global_step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, scalar_value, global_step)
            
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
            
    def log_histogram(self, tag: str, values, global_step: int):
        """Log histogram of values."""
        if self.enabled:
            self.writer.add_histogram(tag, values, global_step)
            
    def log_graph(self, model, input_to_model):
        """Log model graph."""
        if self.enabled:
            self.writer.add_graph(model, input_to_model)
            
    def close(self):
        """Close TensorBoard writer."""
        if self.enabled:
            self.writer.close()


class MetricsLogger:
    """
    Unified metrics logging with multiple backends.
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        enable_wandb: bool = False,
        enable_tensorboard: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.file_logger = logging.getLogger("ultra_ai.metrics")
        
        self.wandb_logger = WandbLogger(
            config=wandb_config,
            enabled=enable_wandb,
        ) if enable_wandb else None
        
        self.tensorboard_logger = TensorBoardLogger(
            log_dir=str(self.log_dir / "tensorboard"),
            enabled=enable_tensorboard,
        ) if enable_tensorboard else None
        
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        epoch: int = 0,
        prefix: str = "",
    ):
        """Log metrics to all enabled backends."""
        # Prepare metrics with prefix
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics
            
        # File logging
        self.file_logger.info(
            f"Step {step}, Epoch {epoch}: {json.dumps(prefixed_metrics, indent=2)}"
        )
        
        # WandB logging
        if self.wandb_logger:
            self.wandb_logger.log(prefixed_metrics, step=step)
            
        # TensorBoard logging
        if self.tensorboard_logger:
            for key, value in prefixed_metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_logger.log_scalar(key, value, step)
                    
    def log_model_metrics(self, model: torch.nn.Module, step: int):
        """Log model-specific metrics."""
        model_metrics = {}
        
        # Parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_metrics.update({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/parameter_ratio": trainable_params / total_params if total_params > 0 else 0,
        })
        
        # Gradient statistics
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
                
        if grad_norms:
            model_metrics.update({
                "model/grad_norm_mean": np.mean(grad_norms),
                "model/grad_norm_max": np.max(grad_norms),
                "model/grad_norm_std": np.std(grad_norms),
            })
            
        self.log_metrics(model_metrics, step, prefix="")
        
    def close(self):
        """Close all loggers."""
        if self.wandb_logger:
            self.wandb_logger.finish()
            
        if self.tensorboard_logger:
            self.tensorboard_logger.close()