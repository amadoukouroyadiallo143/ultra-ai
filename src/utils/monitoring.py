import torch
import psutil
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Comprehensive metrics tracker for training and evaluation.
    Tracks loss components, performance metrics, and system resources.
    """
    
    def __init__(self, window_size: int = 100, save_frequency: int = 1000):
        self.window_size = window_size
        self.save_frequency = save_frequency
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.windows = defaultdict(lambda: deque(maxlen=window_size))
        
        # Step tracking
        self.step = 0
        self.epoch = 0
        
        # Timing
        self.start_time = time.time()
        self.step_times = deque(maxlen=window_size)
        
        # Best metrics tracking
        self.best_metrics = {}
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Update metrics with new values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (auto-incremented if not provided)
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
            
        current_time = time.time()
        
        # Update all metrics
        for name, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            elif isinstance(value, (list, tuple)):
                value = np.mean(value)
                
            self.metrics[name].append((self.step, value))
            self.windows[name].append(value)
            
            # Track best metrics
            if name.endswith('_loss'):
                if name not in self.best_metrics or value < self.best_metrics[name]:
                    self.best_metrics[name] = value
            elif name.endswith(('_acc', '_accuracy', '_f1', '_precision', '_recall')):
                if name not in self.best_metrics or value > self.best_metrics[name]:
                    self.best_metrics[name] = value
                    
        # Update timing
        if hasattr(self, 'last_update_time'):
            step_time = current_time - self.last_update_time
            self.step_times.append(step_time)
            
        self.last_update_time = current_time
        
        # Periodic saving
        if self.step % self.save_frequency == 0:
            self._save_metrics()
            
    def get_current_metrics(self) -> Dict[str, float]:
        """Get the most recent values for all metrics."""
        current = {}
        for name, values in self.metrics.items():
            if values:
                current[name] = values[-1][1]
        return current
        
    def get_windowed_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics over the current window for all metrics."""
        windowed = {}
        for name, values in self.windows.items():
            if values:
                windowed[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'current': values[-1] if values else 0.0,
                }
        return windowed
        
    def get_throughput_metrics(self) -> Dict[str, float]:
        """Get throughput and timing metrics."""
        metrics = {}
        
        if self.step_times:
            avg_step_time = np.mean(self.step_times)
            metrics['avg_step_time'] = avg_step_time
            metrics['steps_per_second'] = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
            
        total_time = time.time() - self.start_time
        metrics['total_training_time'] = total_time
        metrics['avg_steps_per_second'] = self.step / total_time if total_time > 0 else 0.0
        
        return metrics
        
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best achieved values for each metric."""
        return self.best_metrics.copy()
        
    def reset_window(self):
        """Reset the sliding windows."""
        self.windows.clear()
        self.step_times.clear()
        
    def _save_metrics(self):
        """Save metrics to disk for persistence."""
        # This could be expanded to save to various formats
        logger.debug(f"Metrics checkpoint at step {self.step}")


class PerformanceMonitor:
    """
    Real-time performance monitoring for system resources and model performance.
    """
    
    def __init__(
        self,
        monitor_interval: float = 1.0,
        gpu_monitoring: bool = True,
        memory_monitoring: bool = True,
        disk_monitoring: bool = True,
    ):
        self.monitor_interval = monitor_interval
        self.gpu_monitoring = gpu_monitoring and torch.cuda.is_available()
        self.memory_monitoring = memory_monitoring
        self.disk_monitoring = disk_monitoring
        
        # Monitoring data
        self.system_metrics = defaultdict(deque)
        self.max_memory_usage = 0.0
        self.peak_gpu_memory = 0.0
        
        # Threading
        self.monitoring_thread = None
        self.should_monitor = False
        
        # Performance alerts
        self.alerts = []
        self.memory_threshold = 0.9  # Alert at 90% memory usage
        self.gpu_memory_threshold = 0.95  # Alert at 95% GPU memory
        
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return
            
        self.should_monitor = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started performance monitoring")
        
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.should_monitor = False
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Stopped performance monitoring")
        
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.should_monitor:
            try:
                metrics = self._collect_system_metrics()
                
                # Store metrics
                timestamp = time.time()
                for name, value in metrics.items():
                    self.system_metrics[name].append((timestamp, value))
                    
                    # Keep only recent data (last 1000 samples)
                    if len(self.system_metrics[name]) > 1000:
                        self.system_metrics[name].popleft()
                        
                # Check for alerts
                self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.monitor_interval)
            
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        
        # CPU and memory
        if self.memory_monitoring:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            metrics['cpu_percent'] = cpu_percent
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / 1e9
            metrics['memory_available_gb'] = memory.available / 1e9
            
            self.max_memory_usage = max(self.max_memory_usage, memory.percent)
            
        # GPU metrics
        if self.gpu_monitoring:
            try:
                for i in range(torch.cuda.device_count()):
                    # Memory usage
                    memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(i) / 1e9
                    memory_free = torch.cuda.get_device_properties(i).total_memory / 1e9 - memory_reserved
                    
                    metrics[f'gpu_{i}_memory_allocated_gb'] = memory_allocated
                    metrics[f'gpu_{i}_memory_reserved_gb'] = memory_reserved
                    metrics[f'gpu_{i}_memory_free_gb'] = memory_free
                    
                    memory_usage_percent = memory_reserved / (torch.cuda.get_device_properties(i).total_memory / 1e9)
                    metrics[f'gpu_{i}_memory_percent'] = memory_usage_percent * 100
                    
                    self.peak_gpu_memory = max(self.peak_gpu_memory, memory_reserved)
                    
                    # GPU utilization (if nvidia-ml-py is available)
                    try:
                        import pynvml
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        metrics[f'gpu_{i}_utilization_percent'] = utilization.gpu
                        
                        # Temperature
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        metrics[f'gpu_{i}_temperature_c'] = temp
                        
                    except ImportError:
                        pass  # nvidia-ml-py not available
                    except Exception as e:
                        logger.debug(f"GPU monitoring error: {e}")
                        
            except Exception as e:
                logger.debug(f"GPU metrics collection error: {e}")
                
        # Disk I/O
        if self.disk_monitoring:
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics['disk_read_mb_per_sec'] = disk_io.read_bytes / 1e6
                    metrics['disk_write_mb_per_sec'] = disk_io.write_bytes / 1e6
                    
                # Disk usage for current directory
                disk_usage = psutil.disk_usage('.')
                metrics['disk_usage_percent'] = (disk_usage.used / disk_usage.total) * 100
                
            except Exception as e:
                logger.debug(f"Disk metrics collection error: {e}")
                
        return metrics
        
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check for performance alerts."""
        current_time = time.time()
        
        # Memory alerts
        if 'memory_percent' in metrics:
            if metrics['memory_percent'] > self.memory_threshold * 100:
                alert = {
                    'timestamp': current_time,
                    'type': 'high_memory',
                    'value': metrics['memory_percent'],
                    'threshold': self.memory_threshold * 100,
                    'message': f"High memory usage: {metrics['memory_percent']:.1f}%"
                }
                self.alerts.append(alert)
                logger.warning(alert['message'])
                
        # GPU memory alerts
        for key, value in metrics.items():
            if key.endswith('_memory_percent') and 'gpu_' in key:
                if value > self.gpu_memory_threshold * 100:
                    alert = {
                        'timestamp': current_time,
                        'type': 'high_gpu_memory',
                        'device': key,
                        'value': value,
                        'threshold': self.gpu_memory_threshold * 100,
                        'message': f"High GPU memory usage on {key}: {value:.1f}%"
                    }
                    self.alerts.append(alert)
                    logger.warning(alert['message'])
                    
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
    def get_current_system_state(self) -> Dict[str, Any]:
        """Get current system resource state."""
        metrics = self._collect_system_metrics()
        
        return {
            'current_metrics': metrics,
            'peak_memory_usage': self.max_memory_usage,
            'peak_gpu_memory_gb': self.peak_gpu_memory,
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'monitoring_active': self.should_monitor,
        }
        
    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system monitoring summary."""
        current_state = self.get_current_system_state()
        
        # Calculate averages over recent history
        recent_averages = {}
        for name, history in self.system_metrics.items():
            if history:
                recent_values = [value for _, value in list(history)[-60:]]  # Last 60 samples
                if recent_values:
                    recent_averages[f"{name}_avg_1min"] = np.mean(recent_values)
                    recent_averages[f"{name}_max_1min"] = np.max(recent_values)
                    
        return {
            **current_state,
            'recent_averages': recent_averages,
            'total_alerts': len(self.alerts),
        }
        
    def start_epoch(self):
        """Mark the start of a training epoch."""
        self.epoch_start_time = time.time()
        
    def end_epoch(self) -> Dict[str, float]:
        """Mark the end of a training epoch and return timing info."""
        if hasattr(self, 'epoch_start_time'):
            epoch_time = time.time() - self.epoch_start_time
            return {'epoch_time': epoch_time}
        return {}
        
    def get_memory_profile(self) -> Dict[str, Any]:
        """Get detailed memory profiling information."""
        profile = {
            'system_memory': {},
            'gpu_memory': {},
            'process_memory': {},
        }
        
        # System memory
        memory = psutil.virtual_memory()
        profile['system_memory'] = {
            'total_gb': memory.total / 1e9,
            'available_gb': memory.available / 1e9,
            'used_gb': memory.used / 1e9,
            'percent_used': memory.percent,
            'peak_percent': self.max_memory_usage,
        }
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                profile['gpu_memory'][f'gpu_{i}'] = {
                    'total_gb': device_props.total_memory / 1e9,
                    'allocated_gb': torch.cuda.memory_allocated(i) / 1e9,
                    'reserved_gb': torch.cuda.memory_reserved(i) / 1e9,
                    'free_gb': (device_props.total_memory - torch.cuda.memory_reserved(i)) / 1e9,
                    'peak_gb': torch.cuda.max_memory_allocated(i) / 1e9,
                }
                
        # Process memory
        process = psutil.Process()
        mem_info = process.memory_info()
        profile['process_memory'] = {
            'rss_gb': mem_info.rss / 1e9,  # Resident Set Size
            'vms_gb': mem_info.vms / 1e9,  # Virtual Memory Size
            'percent': process.memory_percent(),
        }
        
        return profile
        
    def save_monitoring_data(self, path: str):
        """Save monitoring data to file."""
        data = {
            'system_metrics': {
                name: list(history) for name, history in self.system_metrics.items()
            },
            'alerts': self.alerts,
            'summary': self.get_system_summary(),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved monitoring data to {path}")


class ModelProfiler:
    """
    Model-specific profiling for performance analysis and optimization.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.forward_times = []
        self.backward_times = []
        self.layer_times = defaultdict(list)
        
    def profile_forward_pass(self, *args, **kwargs):
        """Profile a forward pass and return timing information."""
        start_time = time.time()
        
        # Enable profiling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            output = self.model(*args, **kwargs)
            
        end_time = time.time()
        forward_time = end_time - start_time
        self.forward_times.append(forward_time)
        
        # Get profiling results
        profiling_results = {
            'forward_time': forward_time,
            'profiler_trace': prof.key_averages().table(sort_by="cuda_time_total", row_limit=10),
            'memory_stats': torch.cuda.memory_stats() if torch.cuda.is_available() else {},
        }
        
        return output, profiling_results
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        if self.forward_times:
            summary['forward_pass'] = {
                'avg_time': np.mean(self.forward_times),
                'std_time': np.std(self.forward_times),
                'min_time': np.min(self.forward_times),
                'max_time': np.max(self.forward_times),
                'total_samples': len(self.forward_times),
            }
            
        if self.backward_times:
            summary['backward_pass'] = {
                'avg_time': np.mean(self.backward_times),
                'std_time': np.std(self.backward_times),
                'min_time': np.min(self.backward_times),
                'max_time': np.max(self.backward_times),
                'total_samples': len(self.backward_times),
            }
            
        # Model size information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary['model_info'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1e6,  # Assuming fp32
        }
        
        return summary