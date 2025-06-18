#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module selects a single voxel within a neighbourhood as the centre of the PVS.
Version:    1.0.0
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np


# ***************************************************************************
# * Function
# ***************************************************************************
def summarise_neighbourhood(patch: np.ndarray) -> np.ndarray:
    r"""
    Select a single voxel within a neighbourhood as the centre of the PVS.

    Parameters
    ----------
    patch : ndarray
        3D boolean or binary patch.

    Returns
    -------
    ndarray
        3D binary patch with exactly one voxel set to 1 (if any existed).
    """
    new_patch = np.zeros_like(patch)
    # find indices where patch == 1
    feasible = np.argwhere(patch == 1)
    if feasible.size > 0:
        # choose one at random
        idx = np.random.randint(len(feasible))
        coord = tuple(feasible[idx])
        new_patch[coord] = 1
    return new_patch
