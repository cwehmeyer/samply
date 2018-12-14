import numpy as np
from functools import partial
from .box import distance_calculator


def maxwell_boltzmann(mass, dimension, beta=1.0):
    """Samples momenta from the Maxwell-Boltzmann distribution

    Arguments:
        mass (array-like of float): mass of the particles
        dimension (int): number of spacial dimensions
        beta (float): inverse thermal energy (Boltzmann constant times temperature)
    """
    sigma = np.sqrt(np.asarray(mass) / beta)[:, None]
    return np.random.normal(size=(len(mass), dimension)) * sigma


def kinetic_energy(momentum, mass):
    """Compute the kinetic energy of moving particles

    Arguments:
        momentum (array-like of float): momentum of the particles
        mass (array-like of float): mass of the particles
    """
    if momentum.ndim == 3:
        mass = mass[None, :]
    return 0.5 * (momentum / mass[..., None]).sum(axis=(-2, -1))


def harmonic_oscillator(spring_constant):
    """Creates a harmonic oscillator potential and force field

    Arguments:
        spring_constant (float): spring constant of the harmonic oscillator
    """
    def _potential(factor, position):
        """A harmonic oscillator potential"""
        return factor * np.power(position, 2).sum(axis=(-2, -1))
    def _force(factor, position):
        """A harmonic oscillator force field"""
        return -position * factor
    potential = partial(_potential, 0.5 * spring_constant)
    force = partial(_force, spring_constant)
    return potential, force


def lennard_jones(epsilon, sigma, distance_vectors=None):
    """Creates a single species Lennard-Jones potential and force field

    Arguments:
        epsilon (float): depth of the potential well
        sigma (float): distance of zero crossing
        distance_vectors (function): computes distance vectors from positions
    """
    def _potential(epsilon, sigma, distance_vectors, position):
        """A Lennard-Jones particles potential"""
        if position.ndim == 3:
            return np.array([
                _potential(epsilon, sigma, distance_vectors, p)
                for p in position])
        distance = np.linalg.norm(distance_vectors(position), axis=-1)
        idx = np.triu_indices_from(distance, k=1)
        fraction = np.power(sigma / distance[idx], 6)
        return 4.0 * (epsilon * (fraction * (fraction - 1.0))).sum()
    def _force(epsilon, sigma, distance_vectors, position):
        """A Lennard-Jones particles force field"""
        if position.ndim == 3:
            return np.array([
                _force(epsilon, sigma, distance_vectors, p)
                for p in position])
        distance_vec = distance_vectors(position)
        distance_sqr = np.sum(distance_vec**2, axis=-1)
        fraction = np.zeros_like(distance_sqr)
        mask = ~np.eye(len(fraction), dtype=bool)
        fraction[mask] = np.power(sigma**2 / distance_sqr[mask], 3)
        fraction[mask] = (fraction[mask] * (2.0 * fraction[mask] - 1.0)) / distance_sqr[mask]
        return 24.0 * epsilon * np.sum(fraction[:, :, None] * distance_vec, axis=-2)
    if distance_vectors is None:
        distance_vectors = distance_calculator()
    potential = partial(_potential, epsilon, sigma, distance_vectors)
    force = partial(_force, epsilon, sigma, distance_vectors)
    return potential, force
