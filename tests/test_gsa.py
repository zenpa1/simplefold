# >>> MODIFICATION START: GatedSparseAttention
import pytest
import torch

from simplefold.model.torch.layers import GatedSparseAttention


@pytest.mark.parametrize("seq_len", [128, 1024])
def test_gated_sparse_attention_shape_and_blocks(seq_len):
    batch_size, hidden_size, num_heads, block_size = 2, 256, 8, 64
    module = GatedSparseAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_norm=True,
        block_size=block_size,
    )
    inputs = torch.randn(batch_size, seq_len, hidden_size)

    output = module(inputs)

    assert output.shape == inputs.shape
    assert seq_len % block_size == 0
    assert module.block_size == block_size


# <<< MODIFICATION END: GatedSparseAttention