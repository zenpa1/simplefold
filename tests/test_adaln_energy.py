# >>> MODIFICATION START: AdaLNEnergy
import pytest
import torch

from simplefold.model.torch.layers import AdaLNEnergy


@pytest.mark.parametrize("free_energy_shape", [None, (4,)])
def test_adaln_energy_forward_shape(free_energy_shape):
    batch_size, seq_len, hidden_size, cond_dim = 4, 12, 256, 128
    module = AdaLNEnergy(hidden_size=hidden_size, cond_dim=cond_dim)
    inputs = torch.randn(batch_size, seq_len, hidden_size)
    conditioning = torch.randn(batch_size, cond_dim)
    free_energy = (
        None
        if free_energy_shape is None
        else torch.randn(*free_energy_shape)
    )

    output = module(inputs, conditioning, free_energy=free_energy)

    assert output.shape == inputs.shape


# <<< MODIFICATION END: AdaLNEnergy