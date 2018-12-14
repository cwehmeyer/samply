import numpy as np
import pytest
from scipy import stats
from ..models import harmonic_oscillator
from ..mc import uniform_update
from ..mc import sample


def test_sample():
    beta = 2.0
    potential, _ = harmonic_oscillator(1.0)
    positions = sample(
        'metropolis', potential, [[0.0]], beta=beta, maxiter=1000,
        update=uniform_update(stepsize=0.2), subsample=150)
    positions_ref = np.random.normal(scale=1.0 / np.sqrt(beta), size=1000).reshape(-1, 1, 1)
    _, pvalue = stats.ks_2samp(potential(positions), potential(positions_ref))
    assert pvalue > 0.01
