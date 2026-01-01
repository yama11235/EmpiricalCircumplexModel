# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# The text below is the original header from the nanoGPT library
"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from transformers import TrainerCallback


def justnorm(tensor: torch.Tensor, dim: int, eps: float = 1e-12) -> torch.Tensor:
    """Return the L2-normalized tensor along the specified dimension."""
    denom = tensor.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return tensor / denom


def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
    # Split the sinusoidal_pos into sin and cos parts
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # Apply the rotary embeddings to the query and key
    q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
    k_rot = torch.stack((-k[..., 1::2], k[..., ::2]), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    k_rot = torch.reshape(k_rot, k.shape[:-1] + (k.shape[-1]//2, 2)) * torch.stack((cos, sin), dim=-1)
    q_rot = torch.reshape(q_rot, q.shape)
    k_rot = torch.reshape(k_rot, k.shape)
    return q_rot, k_rot

def get_sinusoidal_embeddings( n_positions, dim):
    """Generate sinusoidal positional embeddings."""
    position = torch.arange(n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    sinusoidal_emb = torch.zeros((n_positions, dim))
    sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
    return sinusoidal_emb

def _is_ngpt_block(module: nn.Module) -> bool:
    """
    Determine if it is an nGPT Block by "use_nGPT=1 and has necessary Linear".
    Checks based on attributes rather than class name, so it supports GPTClassifier/nGPTClassifier.
    """
    cfg = getattr(module, "config", None)
    if cfg is None or getattr(cfg, "use_nGPT", 0) != 1:
        return False
    needed = ["query", "key", "value", "att_c_proj", "c_fc", "mlp_c_proj"]
    return all(hasattr(module, name) for name in needed)


def _normalize_single_ngpt_block(block: nn.Module) -> None:
    """
    Original nGPT train.py's normalize_matrices(),
    applied to "only one Block".
    """
    with torch.no_grad():
        # Assumes Linear shape is (out_features, in_features)
        # -> Normalize row-wise (dim=1) / column-wise (dim=0) following the original code
        block.query.weight.data.copy_(justnorm(block.query.weight.data, dim=1))
        block.key.weight.data.copy_(justnorm(block.key.weight.data, dim=1))
        block.value.weight.data.copy_(justnorm(block.value.weight.data, dim=1))
        block.c_fc.weight.data.copy_(justnorm(block.c_fc.weight.data, dim=1))

        block.att_c_proj.weight.data.copy_(justnorm(block.att_c_proj.weight.data, dim=0))
        block.mlp_c_proj.weight.data.copy_(justnorm(block.mlp_c_proj.weight.data, dim=0))

class NGPTWeightNormCallback(TrainerCallback):
    """
    Callback to call model.normalize_ngpt_matrices() after optimizer.step().
    Implements pseudo-Riemannian optimization (projection onto sphere).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def on_train_begin(self, args, state, control, **kwargs):
        # Normalize once at the beginning as well
        if not self.enabled:
            return control
        model = kwargs.get("model", None)
        normalize = getattr(model, "normalize_ngpt_matrices", None)
        if callable(normalize):
            normalize()
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        # Hook called immediately after optimizer.step() (before zero_grad)
        if not self.enabled:
            return control
        model = kwargs.get("model", None)
        normalize = getattr(model, "normalize_ngpt_matrices", None)
        if callable(normalize):
            normalize()
        return control

class Block(nn.Module):

    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        
        self.c_fc    = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.bfloat16)
        self.silu    = nn.SiLU()
        self.mlp_c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16)

        if (config.use_nGPT == 0):
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if (config.use_nGPT == 1):
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.sqk_init_value = 1.0       
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))

    
    @staticmethod
    def justnorm(x: torch.Tensor) -> torch.Tensor:
        return justnorm(x, dim=-1)

    def forward(self, h, attention_mask=None):
        B, T, C = h.size()

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_att(h)
        
        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head) 
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)

        sinusoidal_pos = get_sinusoidal_embeddings(T, self.config.n_embd // self.config.n_head).to(device=q.device)
        q, k = apply_rotary_position_embeddings(sinusoidal_pos, q.transpose(1, 2), k.transpose(1, 2))
        # q = q.transpose(2, 1)
        # k = k.transpose(2, 1)
        v = v.transpose(1, 2)

        if (self.config.use_nGPT == 1):
            # Shape for (1, Head, 1, Dim) -> (Batch, Head, Seq, Dim)
            head_dim = self.config.n_embd // self.config.n_head
            sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling)).view(1, self.config.n_head, 1, head_dim)
            # sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling)).view(1, 1, self.config.n_head, self.config.n_embd // self.config.n_head)
            q = sqk * self.justnorm(q)  
            k = sqk * self.justnorm(k)  

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if (self.config.use_nGPT == 0): softmax_scale = 1.0 / sqrt_head_dim 
        if (self.config.use_nGPT == 1): softmax_scale = sqrt_head_dim 
        
        # y = flash_attn_func(q.to(dtype=torch.bfloat16), k.to(dtype=torch.bfloat16), v.to(dtype=torch.bfloat16), dropout_p=0.0, softmax_scale=softmax_scale, causal=False, window_size=(-1, -1), alibi_slopes=None, deterministic=True)
        
        # Create mask
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: (Batch, Seq) -> 1=valid, 0=padding
            # PyTorch SDPA expects: (Batch, 1, 1, Seq) or broadcastable
            # Boolean mask: True = valid (attend), False = invalid (ignore)
            attn_mask = attention_mask.view(B, 1, 1, T).bool()

        if self.config.use_nGPT == 1:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, 
                scale=softmax_scale,
                is_causal=False
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, 
                is_causal=False
            )
        
        y = y.to(dtype=q.dtype)
        # (Batch, Head, Seq, Dim) -> (Batch, Seq, Head, Dim) -> (Batch, Seq, Hidden)
        y = y.transpose(1, 2).contiguous().view(B, T, self.config.n_embd)
        
        h_att = self.att_c_proj(y)

        if (self.config.use_nGPT == 0):
            h = h + h_att
        if (self.config.use_nGPT == 1):
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)
            
            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_att)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        hin = h
        if (self.config.use_nGPT == 0):
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if (self.config.use_nGPT == 1):
            suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.config.n_embd ** 0.5))) 
            uv = suv * uv  
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if (self.config.use_nGPT == 0):
            h = h + h_mlp
        if (self.config.use_nGPT == 1):
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h) # normally, normalization is not needed
            B_norm = self.justnorm(h_mlp)
                
            #res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        return h

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False 

class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight
