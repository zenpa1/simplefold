# >>> MODIFICATION START: TopologyConditionedMoE
import pytest
import torch

from simplefold.model.torch.layers import TopologyConditionedMoE


def test_topology_conditioned_moe_shape_and_auxiliary_loss():
    batch_size, seq_len, dim, topology_dim = 2, 16, 256, 8
    module = TopologyConditionedMoE(
        dim=dim,
        hidden_dim=dim * 4,
        num_experts=4,
        top_k=2,
        topology_dim=topology_dim,
    )
    tokens = torch.randn(batch_size, seq_len, dim)
    topology = torch.randn(batch_size, seq_len, topology_dim)

    output = module(tokens, topology=topology)

    assert output.shape == tokens.shape
    assert module.last_aux_loss is not None
    assert module.last_aux_loss.ndim == 0
    assert torch.isfinite(module.last_aux_loss)


# <<< MODIFICATION END: TopologyConditionedMoE