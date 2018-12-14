import numpy as np


def langevin(
        force, position, momentum, mass,
        damping=0.1, timestep=0.001, beta=1.0,
        maxiter=None, subsample=1):
    """Generator for the Langevin integration scheme

    Arguments:
        force (function): computes the force for a position
        position (array-like of float): initial position
        momentum (array-like of float): initial momentum
        mass (array-like of float): mass of the particles
        damping (float): damping factor
        timestep (float): integration time step
        beta (float): inverse thermal energy (Boltzmann constant times temperature)
        maxiter (int): maximum number of iterations, runs indefinitely if None
        subsample (int): number of updates between yielded positions
    """
    position = np.array(position)
    momentum = np.array(momentum)
    mass = np.array(mass)
    frc = force(position)
    th = 0.5 * timestep
    thm = th / mass[:, None]
    edt = np.exp(-damping * timestep)
    sqf = np.sqrt((1.0 - edt**2) * mass / beta)[:, None]
    shape = list(position.shape)
    counter = 0
    while True:
        if maxiter is not None:
            counter += 1
            if counter > maxiter:
                break
        for _ in range(subsample):
            momentum += th * frc
            position += thm * momentum
            momentum[:] = edt * momentum + sqf * np.random.randn(*shape)
            position += thm * momentum
            frc[:] = force(position)
            momentum += th * frc
        yield position, momentum


def sample(method, force, position, momentum, mass, **kwargs):
    """Sample positions and momenta using a molecular dynamics scheme

    Arguments:
        method: [langevin]
        force (function): computes the force for a position
        position (array-like of float): initial position
        momentum (array-like of float): initial momentum
        mass (array-like of float): mass of the particles
        kwargs: parameters for the generators
    """
    if 'maxiter' not in kwargs or kwargs['maxiter'] == None:
        raise ValueError(f'Set maxiter to a finite number')
    if method.lower() == 'langevin':
        generator = langevin(force, position, momentum, mass, **kwargs)
    else:
        raise ValueError(f'Unkown method: {method}')
    position = np.asarray(position)
    momentum = np.asarray(momentum)
    positions = np.zeros(shape=[kwargs['maxiter'] + 1] + list(position.shape))
    momenta = np.zeros_like(positions)
    positions[0, :, :] = position
    momenta[0, :, :] = momentum
    for i, (position, momentum) in enumerate(generator):
        positions[i + 1, :, :] = position
        momenta[i + 1, :, :] = momentum
    return positions, momenta
