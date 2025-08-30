import torch
import gc
import logging
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Advanced memory management for ultra-large model training.
    Handles GPU memory optimization, garbage collection, and memory profiling.
    """
    
    def __init__(
        self,
        enable_auto_cleanup: bool = True,
        cleanup_threshold: float = 0.85,  # Cleanup when 85% memory used
        aggressive_cleanup: bool = False,
        monitor_memory: bool = True,
    ):
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_threshold = cleanup_threshold
        self.aggressive_cleanup = aggressive_cleanup
        self.monitor_memory = monitor_memory
        
        # Memory tracking
        self.memory_history = defaultdict(list)
        self.peak_memory = 0.0
        self.cleanup_count = 0
        
        # Auto cleanup thread
        self.cleanup_thread = None
        self.should_monitor = False
        
        if self.monitor_memory:
            self.start_monitoring()
            
    def cleanup(self, force: bool = False):
        """
        Perform memory cleanup with garbage collection.
        
        Args:
            force: Force cleanup regardless of thresholds
        """
        if not force and not self._should_cleanup():
            return
            
        start_time = time.time()
        
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # Aggressive cleanup if enabled
        if self.aggressive_cleanup or force:
            self._aggressive_cleanup()
            
        # Get final memory usage
        final_memory = self.get_memory_usage()
        cleanup_time = time.time() - start_time
        
        self.cleanup_count += 1
        
        logger.info(
            f"Memory cleanup #{self.cleanup_count}: "
            f"freed {initial_memory.get('gpu_allocated', 0) - final_memory.get('gpu_allocated', 0):.1f}MB GPU, "
            f"collected {collected} objects in {cleanup_time:.2f}s"
        )
        
    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed based on memory usage."""
        if not self.enable_auto_cleanup:
            return False
            
        memory_usage = self.get_memory_usage()
        
        # Check GPU memory
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                gpu_usage = self.get_gpu_memory_usage(device_id)
                if gpu_usage.get('usage_percent', 0) > self.cleanup_threshold:
                    return True
                    
        # Check system memory
        if memory_usage.get('system_percent', 0) > self.cleanup_threshold * 100:
            return True
            
        return False
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get comprehensive memory usage information."""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info.update({
            'system_total': system_memory.total / 1e6,  # MB
            'system_available': system_memory.available / 1e6,
            'system_used': system_memory.used / 1e6,
            'system_percent': system_memory.percent,
        })
        
        # GPU memory
        if torch.cuda.is_available():
            total_gpu_allocated = 0
            total_gpu_reserved = 0
            
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id) / 1e6
                reserved = torch.cuda.memory_reserved(device_id) / 1e6
                
                memory_info[f'gpu_{device_id}_allocated'] = allocated
                memory_info[f'gpu_{device_id}_reserved'] = reserved
                
                total_gpu_allocated += allocated
                total_gpu_reserved += reserved
                
            memory_info['gpu_allocated'] = total_gpu_allocated
            memory_info['gpu_reserved'] = total_gpu_reserved
            
        return memory_info
        
    def get_gpu_memory_usage(self, device_id: int) -> Dict[str, float]:
        """Get detailed GPU memory usage for specific device."""
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            return {}
            
        props = torch.cuda.get_device_properties(device_id)
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        
        return {
            'device_id': device_id,
            'device_name': props.name,
            'total_memory': props.total_memory / 1e6,  # MB
            'allocated': allocated / 1e6,
            'reserved': reserved / 1e6,
            'free': (props.total_memory - reserved) / 1e6,
            'usage_percent': (reserved / props.total_memory) * 100,
        }
        
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.cleanup_thread is not None and self.cleanup_thread.is_alive():
            return
            
        self.should_monitor = True
        self.cleanup_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Started memory monitoring")
        
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.should_monitor:
            try:
                if self._should_cleanup():
                    self.cleanup()
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
            time.sleep(5.0)
            
    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        for _ in range(3):
            gc.collect()
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()