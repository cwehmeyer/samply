import numpy as np
from functools import partial


def uniform_update(stepsize=0.1, drop=None):
    """Creates a uniform sampling scheme for trial positions

    Arguments:
        stepsize (float): max step size in each position component
        drop (float): probability of not updating a position component
    """
    def _uniform_update(stepsize, position):
        """Uniform update without drop"""
        return position + np.random.uniform(
            low=-stepsize, high=stepsize, size=position.shape)
    def _uniform_update_drop(stepsize, drop, position):
        """Uniform update with drop"""
        delta = np.random.uniform(
            low=-stepsize, high=stepsize, size=position.shape)
        mask = np.random.choice(
            (1, 0), p=(1.0 - drop, drop), size=position.shape)
        return position + mask * delta
    if drop is None:
        return partial(_uniform_update, stepsize)
    return partial(_uniform_update_drop, stepsize, drop)
        

def metropolis(potential, position, beta=1.0, update=None, maxiter=None, subsample=1):
    """Generator for the Metropolis scheme

    Arguments:
        potential (function): computes the potential energy for a position
        position (array-like of float): initial position
        beta (float): inverse thermal energy (Boltzmann constant times temperature)
        update (function): creates trial positions, defaults to uniform_update
        maxiter (int): maximum number of iterations, runs indefinitely if None
        subsample (int): number of updates between yielded positions
    """
    if update is None:
        update = uniform_update()
    position = np.array(position)
    energy = potential(position)
    counter = 0
    while True:
        if maxiter is not None:
            counter += 1
            if counter > maxiter:
                break
        for _ in range(subsample):
            position_ = update(position)
            energy_ = potential(position_)
            if energy_ <= energy \
            or np.random.rand() < np.exp(beta * (energy - energy_)):
                position[:] = position_
                energy = energy_
        yield position


def sample(method, potential, position, **kwargs):
    """Sample positions using a Monte Carlo scheme

    Arguments:
        method: [metropolis]
        potential (function): computes the potential energy for a position
        position (array-like of float): initial position
        kwargs: parameters for the generators
    """
    if 'maxiter' not in kwargs or kwargs['maxiter'] == None:
        raise ValueError(f'Set maxiter to a finite number')
    if method.lower() == 'metropolis':
        generator = metropolis(potential, position, **kwargs)
    else:
        raise ValueError(f'Unkown method: {method}')
    position = np.asarray(position)
    positions = np.zeros(shape=[kwargs['maxiter'] + 1] + list(position.shape))
    positions[0, :, :] = position
    for i, position in enumerate(generator):
        positions[i + 1, :, :] = position
    return positions
