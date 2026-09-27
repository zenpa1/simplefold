#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from torch import nn
from timm.models.vision_transformer import Mlp
# >>> MODIFICATION START: SimpleFoldBlock
from model.torch.layers import (
    AdaLNEnergy,
    TopologyConditionedMoE,
    modulate,
    SwiGLUFeedForward,
)
# <<< MODIFICATION END: SimpleFoldBlock


# >>> MODIFICATION START: SimpleFoldBlock
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=True,
        feed_forward_layer=None,
        energy_cond_dim=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if feed_forward_layer is not None:
            self.mlp = feed_forward_layer()
        elif use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.adaLN_energy = (
            AdaLNEnergy(hidden_size, energy_cond_dim)
            if energy_cond_dim is not None
            else None
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

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        if self.adaLN_energy is not None:
            nn.init.zeros_(self.adaLN_energy.modulation[-1].weight)
            nn.init.zeros_(self.adaLN_energy.modulation[-1].bias)

    def forward(
        self,
        latents,
        c,
        **kwargs,
    ):
        if self.adaLN_energy is not None:
            energy = kwargs.pop("energy", None)
            topology = kwargs.pop("topology", None)
            attn_input = self.adaLN_energy(latents, c, free_energy=energy)
            latents = latents + self.attn(attn_input, **kwargs)
            mlp_input = self.adaLN_energy(latents, c, free_energy=energy)
            if isinstance(self.mlp, TopologyConditionedMoE):
                mlp_output = self.mlp(mlp_input, topology=topology)
            else:
                mlp_output = self.mlp(mlp_input)
            return latents + mlp_output

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        _latents = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs
        )
        latents = latents + gate_msa.unsqueeze(1) * _latents
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class TransformerBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        self_attention_layer,
        hidden_size,
        mlp_ratio=4.0,
        use_swiglu=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = self_attention_layer()
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            self.mlp = SwiGLUFeedForward(hidden_size, mlp_hidden_dim)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

    def forward(
        self,
        latents,
        **kwargs,
    ):
        _latents = self.attn(self.norm1(latents), **kwargs)
        latents = latents + _latents
        latents = latents + self.mlp(self.norm2(latents))
        return latents


class HomogenTrunk(nn.Module):
    def __init__(self, block, depth):
        super().__init__()
        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    def forward(self, latents, c, **kwargs):
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents


    # <<< MODIFICATION END: SimpleFoldBlock