"""
Kernels CUDA optimisés pour les opérations Mamba-2
Implémentation haute performance pour selective scan et opérations critiques
"""

import torch
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
import os
from typing import Optional, Tuple
import warnings

# Code CUDA pour selective scan optimisé
CUDA_SELECTIVE_SCAN = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

__global__ void selective_scan_fwd_kernel(
    const float* __restrict__ u,      // [batch, seqlen, d_inner]
    const float* __restrict__ delta,  // [batch, seqlen, d_inner] 
    const float* __restrict__ A,      // [d_inner, d_state]
    const float* __restrict__ B,      // [batch, seqlen, d_state]
    const float* __restrict__ C,      // [batch, seqlen, d_state]
    const float* __restrict__ D,      // [d_inner]
    float* __restrict__ y,            // [batch, seqlen, d_inner]
    float* __restrict__ x,            // [batch, d_inner, d_state] - état interne
    int batch_size,
    int seqlen, 
    int d_inner,
    int d_state
) {
    int batch_id = blockIdx.x;
    int inner_id = threadIdx.x;
    
    if (batch_id >= batch_size || inner_id >= d_inner) return;
    
    // État local pour ce thread
    extern __shared__ float shared_state[];
    float* local_x = &shared_state[threadIdx.x * d_state];
    
    // Initialiser l'état
    for (int s = 0; s < d_state; s++) {
        local_x[s] = 0.0f;
    }
    
    // Scan séquentiel optimisé
    for (int t = 0; t < seqlen; t++) {
        int u_idx = batch_id * seqlen * d_inner + t * d_inner + inner_id;
        int delta_idx = u_idx;
        int B_idx = batch_id * seqlen * d_state + t * d_state;
        int C_idx = B_idx;
        
        float u_val = u[u_idx];
        float delta_val = delta[delta_idx];
        float D_val = D[inner_id];
        
        // Mise à jour de l'état: x = deltaA * x + deltaB * u
        float y_val = 0.0f;
        for (int s = 0; s < d_state; s++) {
            float A_val = A[inner_id * d_state + s];
            float B_val = B[B_idx + s];
            float C_val = C[C_idx + s];
            
            // Discrétisation: deltaA = exp(delta * A)
            float deltaA = expf(delta_val * A_val);
            float deltaB_u = delta_val * B_val * u_val;
            
            // Mise à jour de l'état
            local_x[s] = deltaA * local_x[s] + deltaB_u;
            
            // Sortie: y += C * x
            y_val += C_val * local_x[s];
        }
        
        // Ajouter skip connection
        y_val += D_val * u_val;
        
        // Écrire la sortie
        int y_idx = batch_id * seqlen * d_inner + t * d_inner + inner_id;
        y[y_idx] = y_val;
    }
    
    // Sauvegarder l'état final
    for (int s = 0; s < d_state; s++) {
        int x_idx = batch_id * d_inner * d_state + inner_id * d_state + s;
        x[x_idx] = local_x[s];
    }
}

torch::Tensor selective_scan_cuda(
    torch::Tensor u,
    torch::Tensor delta, 
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor D
) {
    auto batch_size = u.size(0);
    auto seqlen = u.size(1);
    auto d_inner = u.size(2);
    auto d_state = A.size(1);
    
    auto y = torch::zeros_like(u);
    auto x = torch::zeros({batch_size, d_inner, d_state}, u.options());
    
    dim3 grid(batch_size);
    dim3 block(min(d_inner, 1024));
    int shared_mem = block.x * d_state * sizeof(float);
    
    selective_scan_fwd_kernel<<<grid, block, shared_mem>>>(
        u.data_ptr<float>(),
        delta.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        D.data_ptr<float>(),
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        batch_size, seqlen, d_inner, d_state
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("selective_scan_cuda", &selective_scan_cuda, "Selective scan CUDA");
}
"""

class OptimizedMambaKernels:
    """Gestionnaire des kernels CUDA optimisés pour Mamba-2."""
    
    _cuda_kernels = None
    _is_available = None
    
    @classmethod
    def is_available(cls) -> bool:
        """Vérifier si les kernels CUDA sont disponibles."""
        if cls._is_available is not None:
            return cls._is_available
            
        try:
            if not torch.cuda.is_available():
                cls._is_available = False
                return False
                
            # Tenter de compiler les kernels
            cls._get_kernels()
            cls._is_available = True
            return True
        except Exception:
            cls._is_available = False
            return False
    
    @classmethod
    def _get_kernels(cls):
        """Compiler et charger les kernels CUDA."""
        if cls._cuda_kernels is not None:
            return cls._cuda_kernels
            
        try:
            # Créer un fichier temporaire pour le code CUDA
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
                f.write(CUDA_SELECTIVE_SCAN)
                cuda_file = f.name
            
            # Compiler les kernels
            cls._cuda_kernels = load(
                name="selective_scan_cuda",
                sources=[cuda_file],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3', '--use_fast_math', '-lcublas'],
                verbose=False
            )
            
            # Nettoyer le fichier temporaire  
            os.unlink(cuda_file)
            
        except Exception as e:
            warnings.warn(f"Impossible de charger les kernels CUDA: {e}")
            cls._cuda_kernels = None
            
        return cls._cuda_kernels
    
    @classmethod
    def selective_scan(cls, u, delta, A, B, C, D):
        """Selective scan optimisé avec kernels CUDA."""
        if not cls.is_available():
            # Fallback vers implémentation PyTorch
            return cls._selective_scan_pytorch(u, delta, A, B, C, D)
            
        try:
            kernels = cls._get_kernels()
            if kernels is not None:
                return kernels.selective_scan_cuda(u, delta, A, B, C, D)
        except Exception:
            pass
            
        # Fallback
        return cls._selective_scan_pytorch(u, delta, A, B, C, D)
    
    @classmethod
    def _selective_scan_pytorch(cls, u, delta, A, B, C, D):
        """Implémentation PyTorch optimisée comme fallback."""
        batch_size, seqlen, d_inner = u.shape
        d_state = A.shape[-1]
        
        # Vectorisation pour améliorer les performances
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        
        # Scan parallèle approximatif (plus rapide)
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []
        
        # Traiter par chunks pour réduire la mémoire
        chunk_size = min(64, seqlen)
        for i in range(0, seqlen, chunk_size):
            end_idx = min(i + chunk_size, seqlen)
            chunk_len = end_idx - i
            
            # Scan pour ce chunk
            chunk_outputs = []
            for t in range(chunk_len):
                abs_t = i + t
                x = deltaA[:, abs_t] * x + deltaB_u[:, abs_t]
                y = torch.einsum('bdn,bn->bd', x, C[:, abs_t])
                chunk_outputs.append(y)
            
            outputs.extend(chunk_outputs)
        
        y = torch.stack(outputs, dim=1)
        
        # Skip connection
        y = y + u * D.unsqueeze(0).unsqueeze(0)
        
        return y


def optimized_einsum(equation: str, *tensors) -> torch.Tensor:
    """Einsum optimisé avec détection automatique du meilleur backend."""
    
    # Pour les opérations simples, utiliser des opérations dédiées
    if equation == "bld,dn->bldn":  # delta * A
        return tensors[0].unsqueeze(-1) * tensors[1].unsqueeze(0).unsqueeze(0)
    elif equation == "bld,bln,bld->bldn":  # delta * B * u  
        return (tensors[0].unsqueeze(-1) * tensors[1].unsqueeze(2) * 
                tensors[2].unsqueeze(-1))
    elif equation == "bdn,bn->bd":  # x * C
        return torch.sum(tensors[0] * tensors[1].unsqueeze(1), dim=-1)
    
    # Utiliser torch.einsum standard pour les autres cas
    return torch.einsum(equation, *tensors)


class FusedMambaOps:
    """Opérations Mamba fusionnées pour réduire les kernel launches."""
    
    @staticmethod
    def fused_mamba_step(x, conv_weight, conv_bias, x_proj_weight, dt_proj_weight, 
                        dt_proj_bias, A_log, D, dt_rank, d_state, d_inner):
        """Étape Mamba fusionnée : conv1d + projection + selective scan."""
        
        # Fusion conv1d + activation
        x_conv = F.conv1d(x.transpose(1, 2), conv_weight, conv_bias, 
                         padding=conv_weight.shape[-1] - 1, groups=d_inner)
        x_conv = x_conv[..., :x.shape[1]].transpose(1, 2)  # Trim to original length
        x_silu = F.silu(x_conv)
        
        # Projection fusionnée
        x_proj = F.linear(x_silu, x_proj_weight)
        dt, B, C = x_proj.split([dt_rank, d_state, d_state], dim=-1)
        
        # dt projection avec activation
        dt = F.softplus(F.linear(dt, dt_proj_weight, dt_proj_bias))
        
        # Selective scan
        A = -torch.exp(A_log.float())
        y = OptimizedMambaKernels.selective_scan(x_silu, dt, A, B, C, D.float())
        
        return y