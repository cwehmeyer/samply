import numpy as np
import pytest
from scipy import stats
from ..models import maxwell_boltzmann
from ..models import kinetic_energy
from ..models import harmonic_oscillator


def test_maxwell_boltzmann():
    momentum = np.linalg.norm(maxwell_boltzmann(np.ones(1000), 3, beta=1.0), axis=-1)
    kde = stats.gaussian_kde(momentum)
    p = np.linspace(momentum.min(), momentum.max(), 100)
    model = p**2 * np.exp(-p**2 / 2)
    model /= np.sum(model * (p[1] - p[0]))
    np.testing.assert_allclose(kde(p), model, atol=0.1, rtol=0.1)


def test_kinetic_energy():
    mass = np.ones(1000)
    np.testing.assert_allclose(
        kinetic_energy(np.ones((1000, 3)), mass),
        1500)
    np.testing.assert_allclose(
        kinetic_energy(2 * np.ones((1000, 3)), mass),
        3000)
    np.testing.assert_allclose(
        kinetic_energy(np.ones((5, 1000, 3)), 4 * mass),
        [375] * 5)


def test_harmonic_oscillator():
    x = np.linspace(-5, 5, 101).reshape(-1, 1, 1)
    for k in [0.1, 1.0, 10.0]:
        potential, force = harmonic_oscillator(k)
        np.testing.assert_allclose(potential(x), 0.5 * k * np.sum(x**2, axis=(-2, -1)))
        np.testing.assert_allclose(force(x), -k * x)
