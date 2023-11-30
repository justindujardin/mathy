from typing import Callable

import numpy

from fragile.core.utils import random_state


def l2_norm(x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """Euclidean distance between two batches of points stacked across the first dimension."""
    return numpy.linalg.norm(x - y, axis=1)


def relativize(x: numpy.ndarray) -> numpy.ndarray:
    """Normalize the data using a custom smoothing technique."""
    std = x.std()
    if float(std) == 0:
        return numpy.ones(len(x), dtype=type(std))
    standard = (x - x.mean()) / std
    standard[standard > 0] = numpy.log(1.0 + standard[standard > 0]) + 1.0
    standard[standard <= 0] = numpy.exp(standard[standard <= 0])
    return standard


def get_alives_indexes(oobs: numpy.ndarray):
    """Get indexes representing random alive walkers given a vector of death conditions."""
    if numpy.all(oobs):
        return numpy.arange(len(oobs))
    ix = numpy.logical_not(oobs).flatten()
    return numpy.random.choice(numpy.arange(len(ix))[ix], size=len(ix), replace=ix.sum() < len(ix))


def calculate_virtual_reward(
    observs: numpy.ndarray,
    rewards: numpy.ndarray,
    oobs: numpy.ndarray = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: numpy.ndarray = 1.0,
    return_compas: bool = False,
    distance_function: Callable = l2_norm,
):
    """Calculate the virtual rewards given the required data."""

    compas = get_alives_indexes(oobs) if oobs is not None else numpy.arange(len(rewards))
    flattened_observs = observs.reshape(len(oobs), -1)
    other_reward = (
        other_reward.flatten() if isinstance(other_reward, numpy.ndarray) else other_reward
    )
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    rewards_norm = relativize(rewards)

    virtual_reward = distance_norm ** dist_coef * rewards_norm ** reward_coef * other_reward
    return virtual_reward.flatten() if not return_compas else virtual_reward.flatten(), compas


def calculate_clone(virtual_rewards: numpy.ndarray, oobs: numpy.ndarray, eps=1e-3):
    """Calculate the clone indexes and masks from the virtual rewards."""
    compas_ix = get_alives_indexes(oobs)
    vir_rew = virtual_rewards.flatten()
    clone_probs = (vir_rew[compas_ix] - vir_rew) / numpy.maximum(vir_rew, eps)
    will_clone = clone_probs.flatten() > numpy.random.random(len(clone_probs))
    return compas_ix, will_clone


def fai_iteration(
    observs: numpy.ndarray,
    rewards: numpy.ndarray,
    oobs: numpy.ndarray,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    eps=1e-8,
    other_reward: numpy.ndarray = 1.0,
):
    """Perform a FAI iteration."""
    virtual_reward, vr_compas = calculate_virtual_reward(
        observs,
        rewards,
        oobs,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        other_reward=other_reward,
    )
    compas_ix, will_clone = calculate_clone(virtual_rewards=virtual_reward, oobs=oobs, eps=eps)
    return compas_ix, will_clone


def cross_virtual_reward(
    host_observs: numpy.ndarray,
    host_rewards: numpy.ndarray,
    ext_observs: numpy.ndarray,
    ext_rewards: numpy.ndarray,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    return_compas: bool = False,
    distance_function: Callable = l2_norm,
):
    """Calculate the virtual rewards between two cloud of points."""
    host_observs = host_observs.reshape(len(host_rewards), -1)
    ext_observs = ext_observs.reshape(len(ext_rewards), -1)
    compas_host = random_state.permutation(numpy.arange(len(host_rewards)))
    compas_ext = random_state.permutation(numpy.arange(len(ext_rewards)))

    # TODO: check if it's better for the distances to be the same for host and ext
    h_dist = distance_function(host_observs, ext_observs[compas_host])
    e_dist = distance_function(ext_observs, host_observs[compas_ext])
    host_distance = relativize(h_dist.flatten())
    ext_distance = relativize(e_dist.flatten())

    host_rewards = relativize(host_rewards)
    ext_rewards = relativize(ext_rewards)

    host_vr = host_distance ** dist_coef * host_rewards ** reward_coef
    ext_vr = ext_distance ** dist_coef * ext_rewards ** reward_coef
    if return_compas:
        return (host_vr, compas_host), (ext_vr, compas_ext)
    return host_vr, ext_vr


def cross_clone(
    host_virtual_rewards: numpy.ndarray,
    ext_virtual_rewards: numpy.ndarray,
    host_oobs: numpy.ndarray = None,
    eps=1e-3,
):
    """Perform a clone operation between two different groups of points."""
    compas_ix = random_state.permutation(numpy.arange(len(ext_virtual_rewards)))
    host_vr = host_virtual_rewards.flatten()
    ext_vr = ext_virtual_rewards.flatten()
    clone_probs = (ext_vr[compas_ix] - host_vr) / numpy.maximum(ext_vr, eps)
    will_clone = clone_probs.flatten() > numpy.random.random(len(clone_probs))
    if host_oobs is not None:
        will_clone[host_oobs] = True
    return compas_ix, will_clone


def cross_fai_iteration(
    host_observs: numpy.ndarray,
    host_rewards: numpy.ndarray,
    ext_observs: numpy.ndarray,
    ext_rewards: numpy.ndarray,
    host_oobs: numpy.ndarray = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    distance_function: Callable = l2_norm,
    eps: float = 1e-8,
):
    """Perform a FractalAI cloning process between two clouds of points."""
    host_vr, ext_vr = cross_virtual_reward(
        host_observs=host_observs,
        host_rewards=host_rewards,
        ext_observs=ext_observs,
        ext_rewards=ext_rewards,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        distance_function=distance_function,
        return_compas=False,
    )

    compas_ix, will_clone = cross_clone(
        host_virtual_rewards=host_vr, ext_virtual_rewards=ext_vr, host_oobs=host_oobs, eps=eps
    )
    return compas_ix, will_clone
