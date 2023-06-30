import numpy as np
import torch

def RateEncoder(input, min_val, max_val, spike_time):
    rate = float(input - min_val) / (max_val - min_val)
    temp = np.random.rand(spike_time)
    res = np.zeros(spike_time)
    res[temp <= rate] = 1
    idx = np.where(res == 1)[0]
    return res


def FixedRateEncoder(input, min_val, max_val, spike_time):
    rate = float(input - min_val) / (max_val - min_val)
    rate = int(rate * 100)
    res = np.zeros(spike_time)
    res[:rate] = 1
    return res

def bernoulli(datum: torch.Tensor, time: int, dt: float = 1, device="cpu",  **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Bernoulli-distributed spike trains based on input intensity. Inputs must
    be non-negative. Spikes correspond to successful Bernoulli trials, with success
    probability equal to (normalized in [0, 1]) input value.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get("max_prob", 1.0)

    assert 0 <= max_prob <= 1, "Maximum firing probability must be in range [0, 1]"
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()

    if time is not None:
        time = int(time / dt)

    # Normalize inputs and rescale (spike probability proportional to input intensity).
    if datum.max() > 1.0:
        datum /= datum.max()

    # Make spike data from Bernoulli sampling.
    if time is None:
        spikes = torch.bernoulli(max_prob * datum).to(device)
        spikes = spikes.view(*shape)
    else:
        spikes = torch.bernoulli(max_prob * datum.repeat([time, 1]))
        spikes = spikes.view(time, *shape)

    return spikes.byte()

