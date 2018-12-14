import numpy as np
import pytest
from scipy import stats
from ..models import maxwell_boltzmann
from ..models import harmonic_oscillator
from ..md import sample


def test_sample():
    beta, mass = 2.0, [1.0]
    potential, force = harmonic_oscillator(1.0)
    positions, momenta = sample(
        'langevin', force, [[0.0]], maxwell_boltzmann(mass, 1, beta), mass,
        beta=beta, maxiter=1000, timestep=0.01, subsample=150)
    positions_ref = np.random.normal(scale=1.0 / np.sqrt(beta), size=1000).reshape(-1, 1, 1)
    _, pvalue = stats.ks_2samp(potential(positions), potential(positions_ref))
    assert pvalue > 0.01