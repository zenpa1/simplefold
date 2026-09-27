#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import math
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# >>> MODIFICATION START: AdaLNEnergy
class AdaLNEnergy(nn.Module):
    """Layer normalization conditioned on noise/time and optional energy."""

    def __init__(self, hidden_size, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.condition = nn.Linear(cond_dim, hidden_size)
        self.energy = nn.Linear(1, hidden_size)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, cond, free_energy=None):
        conditioning = self.condition(cond)
        if free_energy is not None:
            if free_energy.ndim == 1:
                free_energy = free_energy.unsqueeze(-1)
            conditioning = conditioning + self.energy(free_energy)

        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


# <<< MODIFICATION END: AdaLNEnergy


#################################################################################
#                            Attention Layers                                  #
#################################################################################

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_bias=True,
        qk_norm=True,
        pos_embedder=None,
        linear_target: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = linear_target(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.pos_embedder = pos_embedder

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        pos = kwargs.get("pos")

        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientSelfAttentionLayer(SelfAttentionLayer):
    """Started from https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        attn_mask = kwargs.get("attention_mask")
        pos = kwargs.get("pos")

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = qkv.unbind(0)

        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=q.dtype)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        q, k = self.q_norm(q), self.k_norm(k)
        x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


#################################################################################
#                              FeedForward Layer                                #
#################################################################################


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)
        torch.nn.init.xavier_uniform_(self.w3.weight)
        if self.w1.bias is not None:
            torch.nn.init.constant_(self.w1.bias, 0)
        if self.w2.bias is not None:
            torch.nn.init.constant_(self.w2.bias, 0)
        if self.w3.bias is not None:
            torch.nn.init.constant_(self.w3.bias, 0)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# >>> MODIFICATION START: TopologyConditionedMoE
class TopologyConditionedMoE(nn.Module):
    """Sparse feed-forward experts selected by token and topology features."""

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=4,
        top_k=2,
        multiple_of=256,
        topology_dim=None,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.num_experts = num_experts
        self.top_k = top_k
        topology_dim = dim if topology_dim is None else topology_dim
        self.topology_projection = nn.Linear(topology_dim, dim, bias=False)
        self.router = nn.Linear(2 * dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(dim, hidden_dim, multiple_of=multiple_of)
            for _ in range(num_experts)
        ])
        self.last_aux_loss = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.router.weight)
        torch.nn.init.xavier_uniform_(self.topology_projection.weight)

    def forward(self, x, topology=None):
        batch_size, seq_len, dim = x.shape
        x_flat = x.reshape(-1, dim)

        if topology is None:
            topology_flat = torch.zeros_like(x_flat)
        else:
            if topology.ndim == 2:
                if topology.shape[0] != batch_size:
                    raise ValueError("topology batch dimension must match x")
                topology = topology.unsqueeze(1).expand(-1, seq_len, -1)
            if topology.shape[:2] != (batch_size, seq_len):
                raise ValueError("topology must have shape (batch, seq_len, features)")
            topology_flat = self.topology_projection(topology.reshape(-1, topology.shape[-1]))

        router_probs = F.softmax(self.router(torch.cat([x_flat, topology_flat], dim=-1)), dim=-1)
        top_k_probs, top_k_idx = router_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out_flat = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            mask = (top_k_idx == expert_id).any(dim=-1)
            if not mask.any():
                continue
            expert_output = expert(x_flat[mask])
            weight = (top_k_probs[mask] * (top_k_idx[mask] == expert_id)).sum(dim=-1, keepdim=True)
            out_flat[mask] += weight * expert_output

        self.last_aux_loss = self._load_balance_loss(router_probs, top_k_idx)
        return out_flat.reshape(batch_size, seq_len, dim)

    def _load_balance_loss(self, router_probs, top_k_idx):
        top1_idx = top_k_idx[:, 0]
        tokens_per_expert = F.one_hot(top1_idx, self.num_experts).float().mean(dim=0)
        prob_per_expert = router_probs.mean(dim=0)
        return self.num_experts * (tokens_per_expert * prob_per_expert).sum()


# <<< MODIFICATION END: TopologyConditionedMoE

#################################################################################
#                               Utility Layers                                  #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#ADDED NEW ENERGY EMBEDDER

class EnergyEmbedder(nn.Module): # NEW
    """
    Embeds scalar free-energy values into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.constant_(self.mlp[2].weight, 0)   # was normal_(..., std=0.02)
        nn.init.constant_(self.mlp[2].bias, 0) 

    def forward(self, energy):
        # reuse TimestepEmbedder's sinusoidal scheme — same shape contract (B,) -> (B, hidden_size)
        e_freq = TimestepEmbedder.timestep_embedding(energy, self.frequency_embedding_size)
        e_emb = self.mlp(e_freq)
        return e_emb

# >>> MODIFICATION START: GatedSparseAttention
class GatedSparseAttention(SelfAttentionLayer):
    """Block-local self-attention with an input-conditioned output gate."""

    def __init__(self, *args, block_size=64, sparsity_window=None, **kwargs):
        super().__init__(*args, **kwargs)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.sparsity_window = sparsity_window
        self.feature_gate = nn.Linear(self.qkv.in_features, self.qkv.in_features)

    def forward(self, x, **kwargs):
        batch_size, seq_len, hidden_size = x.shape
        pos = kwargs.get("pos")
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, hidden_size // self.num_heads
        )
        q, k, v = rearrange(qkv, "b n t h d -> t b h n d").unbind(0)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)
        q, k = self.q_norm(q), self.k_norm(k)

        block_count = (seq_len + self.block_size - 1) // self.block_size
        padded_len = block_count * self.block_size
        pad_len = padded_len - seq_len
        if pad_len:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        block_shape = (batch_size * block_count, self.num_heads, self.block_size, -1)
        q = q.reshape(*block_shape)
        k = k.reshape(*block_shape)
        v = v.reshape(*block_shape)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.reshape(batch_size, self.num_heads, padded_len, -1)
        out = out[:, :, :seq_len].transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        gate = torch.sigmoid(self.feature_gate(x))
        return self.proj_drop(self.proj(out * gate))


# <<< MODIFICATION END: GatedSparseAttention

class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            )
        self.dropout_prob = dropout_prob
        self.null_token = nn.Parameter(torch.randn(input_dim), requires_grad=True)

    def token_drop(self, cond, force_drop_ids=None):
        """
        cond: (B, N, D)
        Drops conditions to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids
        cond[drop_ids] = self.null_token[None, None, :]
        return cond

    def forward(self, cond, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        embeddings = self.proj(cond)
        return embeddings


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, c_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 2 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
