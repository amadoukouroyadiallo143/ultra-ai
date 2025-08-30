"""
Système de quantization avancé pour Ultra-AI
- Quantization dynamique INT8/INT4
- Calibration automatique
- Optimisations spécialisées par composant
- Support GPU et CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import math
from collections import defaultdict
import warnings


@dataclass
class QuantizationConfig:
    """Configuration de quantization."""
    # Types de quantization
    weights_dtype: torch.dtype = torch.int8
    activations_dtype: torch.dtype = torch.int8
    
    # Stratégies par composant
    quantize_embeddings: bool = True
    quantize_attention: bool = True 
    quantize_mamba: bool = True
    quantize_moe: bool = True
    quantize_lm_head: bool = True
    
    # Paramètres de calibration
    calibration_samples: int = 128
    percentile: float = 99.99  # Pour déterminer les bornes
    
    # Optimisations
    enable_mixed_precision: bool = True
    fuse_operations: bool = True
    optimize_for_inference: bool = True
    
    # Seuils de qualité
    max_perplexity_degradation: float = 1.05  # 5% max
    min_accuracy_retention: float = 0.95  # 95% min


class DynamicQuantizer:
    """Quantizer dynamique avec calibration automatique."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data = defaultdict(list)
        self.quantization_params = {}
        self.is_calibrated = False
        
    def calibrate(self, model: nn.Module, calibration_loader):
        """Calibrer le quantizer avec des données réelles."""
        print("Démarrage de la calibration quantization...")
        
        model.eval()
        self.calibration_data.clear()
        
        # Hook pour capturer les activations
        hooks = []
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.calibration_data[name].append(output.detach().cpu())
            return hook
        
        # Enregistrer les hooks
        for name, module in model.named_modules():
            if self._should_quantize_module(module, name):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
        
        # Collecter les données
        samples_processed = 0
        with torch.no_grad():
            for batch in calibration_loader:
                if samples_processed >= self.config.calibration_samples:
                    break
                    
                if isinstance(batch, dict):
                    _ = model(**batch)
                else:
                    _ = model(batch)
                    
                samples_processed += batch.size(0) if hasattr(batch, 'size') else 1
        
        # Nettoyer les hooks
        for hook in hooks:
            hook.remove()
        
        # Calculer les paramètres de quantization
        self._compute_quantization_params()
        self.is_calibrated = True
        
        print(f"Calibration terminée avec {samples_processed} échantillons")
    
    def _should_quantize_module(self, module: nn.Module, name: str) -> bool:
        """Déterminer si un module doit être quantizé."""
        if isinstance(module, nn.Linear):
            if 'embed' in name.lower() and not self.config.quantize_embeddings:
                return False
            if 'attention' in name.lower() and not self.config.quantize_attention:
                return False
            if 'mamba' in name.lower() and not self.config.quantize_mamba:
                return False
            if 'moe' in name.lower() and not self.config.quantize_moe:
                return False
            if 'lm_head' in name.lower() and not self.config.quantize_lm_head:
                return False
            return True
        return False
    
    def _compute_quantization_params(self):
        """Calculer les paramètres de quantization à partir des données."""
        print("Calcul des paramètres de quantization...")
        
        for name, activations in self.calibration_data.items():
            if not activations:
                continue
                
            # Concaténer toutes les activations
            all_acts = torch.cat(activations, dim=0)
            
            # Calculer les statistiques robustes
            abs_acts = torch.abs(all_acts)
            max_val = torch.quantile(abs_acts, self.config.percentile / 100.0)
            
            # Paramètres pour INT8 (-128 à 127)
            if self.config.activations_dtype == torch.int8:
                scale = max_val / 127.0
                zero_point = 0  # Quantization symétrique
            else:  # INT4 (-8 à 7)
                scale = max_val / 7.0
                zero_point = 0
            
            self.quantization_params[name] = {
                'scale': scale.item(),
                'zero_point': zero_point,
                'max_val': max_val.item()
            }
        
        print(f"Paramètres calculés pour {len(self.quantization_params)} modules")


class QuantizedLinear(nn.Module):
    """Couche linéaire quantizée optimisée."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_dtype: torch.dtype = torch.int8,
                 activation_dtype: torch.dtype = torch.int8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_dtype = weight_dtype
        self.activation_dtype = activation_dtype
        
        # Poids quantizés
        self.register_buffer('quantized_weight', torch.zeros(out_features, in_features, dtype=weight_dtype))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0))
        
        # Bias en float pour maintenir la précision
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)
        
        # Paramètres d'activation
        self.register_buffer('activation_scale', torch.tensor(1.0))
        self.register_buffer('activation_zero_point', torch.tensor(0))
        
        # État de quantization
        self.is_quantized = False
    
    @classmethod
    def from_float(cls, float_module: nn.Linear, weight_dtype: torch.dtype = torch.int8,
                   activation_dtype: torch.dtype = torch.int8):
        """Convertir un module float en module quantizé."""
        quantized = cls(
            float_module.in_features, 
            float_module.out_features,
            float_module.bias is not None,
            weight_dtype,
            activation_dtype
        )
        
        # Quantizer les poids
        quantized._quantize_weights(float_module.weight.data)
        
        if float_module.bias is not None:
            quantized.bias.data.copy_(float_module.bias.data)
        
        return quantized
    
    def _quantize_weights(self, weight: torch.Tensor):
        """Quantizer les poids."""
        abs_max = torch.abs(weight).max()
        
        if self.weight_dtype == torch.int8:
            scale = abs_max / 127.0
            quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
        else:  # INT4
            scale = abs_max / 7.0  
            quantized = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
        
        self.quantized_weight.copy_(quantized)
        self.weight_scale.copy_(scale)
        self.is_quantized = True
    
    def set_activation_params(self, scale: float, zero_point: int = 0):
        """Définir les paramètres de quantization des activations."""
        self.activation_scale.copy_(torch.tensor(scale))
        self.activation_zero_point.copy_(torch.tensor(zero_point))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass quantizé."""
        if not self.is_quantized:
            # Fallback vers float
            weight = self.quantized_weight.float() * self.weight_scale
            return F.linear(x, weight, self.bias)
        
        # Forward pass optimisé INT8
        if x.dtype != torch.int8 and self.activation_scale != 1.0:
            # Quantizer l'entrée
            x_quantized = self._quantize_activation(x)
        else:
            x_quantized = x
        
        # Multiplication quantizée optimisée
        if torch.cuda.is_available() and x.is_cuda:
            output = self._cuda_quantized_matmul(x_quantized)
        else:
            output = self._cpu_quantized_matmul(x_quantized)
        
        return output
    
    def _quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Quantizer une activation."""
        if self.activation_dtype == torch.int8:
            quantized = torch.round(x / self.activation_scale).clamp(-128, 127)
        else:  # INT4
            quantized = torch.round(x / self.activation_scale).clamp(-8, 7)
        
        return quantized.to(self.activation_dtype)
    
    def _cuda_quantized_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Multiplication matricielle quantizée CUDA."""
        # Utiliser CUTLASS ou cuBLAS pour INT8
        try:
            # Déquantizer pour la multiplication (temporaire)
            weight_float = self.quantized_weight.float() * self.weight_scale
            x_float = x.float() * self.activation_scale
            
            output = F.linear(x_float, weight_float, self.bias)
            return output
        except Exception:
            return self._cpu_quantized_matmul(x)
    
    def _cpu_quantized_matmul(self, x: torch.Tensor) -> torch.Tensor:
        """Multiplication matricielle quantizée CPU."""
        # Déquantizer et calculer
        weight_float = self.quantized_weight.float() * self.weight_scale
        x_float = x.float() * self.activation_scale
        
        output = F.linear(x_float, weight_float, self.bias)
        return output
    
    def get_memory_savings(self) -> float:
        """Calculer les économies de mémoire."""
        if not self.is_quantized:
            return 0.0
            
        original_size = self.in_features * self.out_features * 4  # float32
        quantized_size = self.in_features * self.out_features * 1  # int8
        
        return 1.0 - (quantized_size / original_size)


class AdaptiveQuantizer:
    """Quantizer adaptatif qui ajuste la précision selon l'importance."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.sensitivity_scores = {}
        
    def compute_sensitivity(self, model: nn.Module, validation_loader):
        """Calculer la sensibilité de chaque couche à la quantization."""
        print("Calcul de sensibilité des couches...")
        
        # Baseline avec modèle float
        baseline_loss = self._compute_loss(model, validation_loader)
        
        # Tester chaque couche individuellement
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"Test sensibilité: {name}")
                
                # Sauvegarder les poids originaux
                original_weight = module.weight.data.clone()
                
                # Quantizer temporairement cette couche
                self._quantize_layer_temporarily(module)
                
                # Mesurer la dégradation
                quantized_loss = self._compute_loss(model, validation_loader)
                sensitivity = (quantized_loss - baseline_loss) / baseline_loss
                
                self.sensitivity_scores[name] = sensitivity
                
                # Restaurer les poids originaux
                module.weight.data.copy_(original_weight)
                
                print(f"  Sensibilité: {sensitivity:.4f}")
    
    def _quantize_layer_temporarily(self, layer: nn.Linear):
        """Quantizer temporairement une couche."""
        weight = layer.weight.data
        abs_max = torch.abs(weight).max()
        scale = abs_max / 127.0
        
        quantized = torch.round(weight / scale).clamp(-128, 127)
        dequantized = quantized * scale
        
        layer.weight.data.copy_(dequantized)
    
    def _compute_loss(self, model: nn.Module, validation_loader) -> float:
        """Calculer la loss sur un échantillon de validation."""
        model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(validation_loader):
                if i >= 10:  # Limiter pour vitesse
                    break
                    
                try:
                    if isinstance(batch, dict):
                        outputs = model(**batch)
                    else:
                        outputs = model(batch)
                    
                    # Calculer une loss approximative
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    
                    # Loss approximative (pas besoin des vraies labels)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                         torch.zeros(logits.view(-1).size(0), 
                                                   dtype=torch.long, device=logits.device))
                    
                    total_loss += loss.item()
                    num_samples += 1
                except Exception:
                    continue
        
        return total_loss / max(num_samples, 1)
    
    def get_quantization_strategy(self) -> Dict[str, torch.dtype]:
        """Obtenir la stratégie de quantization basée sur la sensibilité."""
        strategy = {}
        
        if not self.sensitivity_scores:
            # Stratégie par défaut
            return {name: torch.int8 for name in self.sensitivity_scores}
        
        # Trier par sensibilité
        sorted_layers = sorted(self.sensitivity_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Assigner les précisions
        for i, (name, sensitivity) in enumerate(sorted_layers):
            if sensitivity > 0.02:  # Très sensible
                strategy[name] = torch.float16
            elif sensitivity > 0.01:  # Moyennement sensible  
                strategy[name] = torch.int8
            else:  # Peu sensible
                strategy[name] = torch.int8  # Ou INT4 si supporté
        
        return strategy


def quantize_model(model: nn.Module, config: QuantizationConfig,
                   calibration_loader=None) -> nn.Module:
    """Quantizer un modèle complet."""
    print("Démarrage de la quantization du modèle...")
    
    # Créer le quantizer
    quantizer = DynamicQuantizer(config)
    
    # Calibration si données disponibles
    if calibration_loader is not None:
        quantizer.calibrate(model, calibration_loader)
    
    # Remplacer les couches linéaires
    quantized_modules = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if quantizer._should_quantize_module(module, name):
                print(f"Quantization: {name}")
                
                # Créer la version quantizée
                quantized_module = QuantizedLinear.from_float(
                    module, config.weights_dtype, config.activations_dtype
                )
                
                # Définir les paramètres d'activation si calibré
                if quantizer.is_calibrated and name in quantizer.quantization_params:
                    params = quantizer.quantization_params[name]
                    quantized_module.set_activation_params(
                        params['scale'], params['zero_point']
                    )
                
                quantized_modules[name] = quantized_module
    
    # Remplacer dans le modèle
    for name, quantized_module in quantized_modules.items():
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            
        setattr(parent, child_name, quantized_module)
    
    print(f"Quantization terminée: {len(quantized_modules)} modules quantizés")
    
    # Statistiques
    total_params = sum(p.numel() for p in model.parameters())
    quantized_params = sum(
        m.quantized_weight.numel() for m in quantized_modules.values()
    )
    
    memory_savings = 1.0 - (quantized_params / total_params) * 0.25  # INT8 = 1/4 de float32
    print(f"Économies mémoire estimées: {memory_savings:.1%}")
    
    return model


# Utilitaires
def benchmark_quantization(original_model: nn.Module, quantized_model: nn.Module,
                          test_loader) -> Dict[str, float]:
    """Comparer les performances avant/après quantization."""
    print("Benchmark quantization...")
    
    results = {}
    
    # Test de vitesse
    import time
    
    def measure_speed(model, loader, name):
        model.eval()
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 10:
                    break
                    
                start = time.time()
                _ = model(batch)
                times.append(time.time() - start)
        
        return np.mean(times)
    
    original_time = measure_speed(original_model, test_loader, "original")
    quantized_time = measure_speed(quantized_model, test_loader, "quantized")
    
    results['speedup'] = original_time / quantized_time
    results['original_time'] = original_time
    results['quantized_time'] = quantized_time
    
    # Utilisation mémoire
    def get_model_size(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)  # MB
    
    results['original_size_mb'] = get_model_size(original_model)
    results['quantized_size_mb'] = get_model_size(quantized_model)
    results['compression_ratio'] = results['original_size_mb'] / results['quantized_size_mb']
    
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Compression: {results['compression_ratio']:.2f}x")
    print(f"Taille originale: {results['original_size_mb']:.1f}MB")
    print(f"Taille quantizée: {results['quantized_size_mb']:.1f}MB")
    
    return results