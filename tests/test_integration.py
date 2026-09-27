# >>> MODIFICATION START: SparseUpcycling
import pytest
import torch
from torch import nn

from simplefold.model.torch.architecture import sparse_upcycle_weights
from simplefold.model.torch.blocks import DiTBlock
from simplefold.model.torch.layers import GatedSparseAttention, TopologyConditionedMoE


def test_modified_block_and_sparse_upcycling():
    batch_size, seq_len, hidden_size = 2, 16, 64

    def attention_factory():
        return GatedSparseAttention(hidden_size, num_heads=8, block_size=8)

    def moe_factory():
        return TopologyConditionedMoE(
            dim=hidden_size,
            hidden_dim=hidden_size * 4,
            num_experts=4,
            top_k=2,
            topology_dim=hidden_size,
        )

    sparse_block = DiTBlock(
        self_attention_layer=attention_factory,
        hidden_size=hidden_size,
        feed_forward_layer=moe_factory,
        energy_cond_dim=hidden_size,
    )
    dense_block = DiTBlock(
        self_attention_layer=attention_factory,
        hidden_size=hidden_size,
        use_swiglu=True,
    )

    copied = sparse_upcycle_weights(
        nn.ModuleDict({"block": dense_block}),
        nn.ModuleDict({"block": sparse_block}),
    )
    assert copied == 4
    assert all(
        torch.equal(sparse_block.mlp.experts[0].state_dict()[key], value)
        for key, value in dense_block.mlp.state_dict().items()
    )

    output = sparse_block(
        torch.randn(batch_size, seq_len, hidden_size),
        torch.randn(batch_size, hidden_size),
        energy=torch.randn(batch_size),
        topology=torch.randn(batch_size, seq_len, hidden_size),
    )
    assert output.shape == (batch_size, seq_len, hidden_size)


# <<< MODIFICATION END: SparseUpcycling