import numpy as np
import pytest
from ..box import wrap_position
from ..box import minimum_image_box
from ..box import distance_calculator


def wrap_naive(position, sizes):
    p = np.array(position)
    for j, size in enumerate(sizes):
        for i in range(len(p)):
            while p[i, j] >= 0.5 * size:
                p[i, j] -= size
            while p[i, j] < -0.5 * size:
                p[i, j] += size
    return p


def test_wrap_position():
    position = np.random.uniform(low=-10, high=10, size=(100, 3))
    sizes = np.array([1.0, 1.5, 2.0])
    position_wrapped = wrap_position(position, sizes)
    for size, p_cmp in zip(sizes, position_wrapped.T):
        assert np.all(-0.5 * size <= p_cmp)
        assert np.all(p_cmp < 0.5 * size)
    np.testing.assert_allclose(
        position_wrapped,
        wrap_naive(position, sizes))


def test_minimum_image_box():
    position = np.random.uniform(low=-10, high=10, size=(100, 3))
    sizes = np.array([1.0, 1.5, 2.0])
    wrapper = minimum_image_box(sizes)
    distances = position[:, None, :] - position[None, :, :]
    distances_wrapped = wrap_position(distances.copy(), sizes)
    for size, d_cmp in zip(sizes, distances_wrapped.reshape(-1, 3).T):
        assert np.all(-0.5 * size <= d_cmp)
        assert np.all(d_cmp < 0.5 * size)
    np.testing.assert_allclose(
        distances_wrapped,
        wrap_naive(distances.reshape(-1, 3), sizes).reshape(distances.shape))


def test_distance_calculator():
    position = np.random.uniform(low=-10, high=10, size=(100, 3))
    distances_ref = position[:, None, :] - position[None, :, :]
    distances = distance_calculator(box=None)
    np.testing.assert_allclose(
        distances(position),
        distances_ref)
    sizes = np.array([1.0, 1.5, 2.0])
    distances = distance_calculator(box=minimum_image_box(sizes))
    np.testing.assert_allclose(
        distances(position),
        wrap_naive(distances_ref.reshape(-1, 3), sizes).reshape(distances_ref.shape))
