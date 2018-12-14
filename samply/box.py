import numpy as np
from functools import partial


def wrap_position(position, sizes):
    """Wrap particle positions inside the simulation cell

    Arguments:
        position (array-like of float): particle positions
        sizes (array-like of float): box sizes
    """
    position = np.asarray(position)
    sizes = np.asarray(sizes)
    shift = sizes[None, :] * np.round(position / sizes[None, :])
    return position - shift


def minimum_image_box(sizes):
    """Creates a distance wrapper using the minimum image convention

    Arguments:
        sizes (array-like of float): box sizes
    """
    def _box(sizes, distance_vectors):
        """A minimum image wrapper for distances"""
        shift = sizes[None, None, :] * np.round(distance_vectors / sizes[None, None, :])
        distance_vectors -= shift
        return distance_vectors
    return partial(_box, np.array(sizes))


def distance_calculator(box=None):
    """Creates a distance vector calculator

    Arguments:
        box (function): use periodicity if not None
    """
    def _distance_vectors(box, position):
        """A distance vector calculator"""
        distance_vectors = position[..., None, :] - position[..., None, :, :]
        try:
            return box(distance_vectors)
        except TypeError:
            return distance_vectors
    return partial(_distance_vectors, box)
