#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module generates a 3D rotation of an sMRI volume.
Version:    1.0.0
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
from typing import Sequence as sequence
from scipy.ndimage import rotate


# ***************************************************************************
# * Function
# ***************************************************************************
def apply_rotation(SI: np.ndarray,
                   theta: float,
                   dimensions: sequence
                   ) -> np.ndarray:
    r"""
    Apply gross motion (3D rotation) to an sMRI volume.

    Parameters
    ----------
    SI : ndarray
        3D sMRI signal.
    theta : float
        Rotation angle in degrees.
    dimensions : sequence of int or bool, length 3
        Unit vector indicating axis of rotation:
        [1,0,0] → x-axis; [0,1,0] → y-axis; [0,0,1] → z-axis.

    Returns
    -------
    ndarray
        Rotated 3D volume, cropped to original shape.
    """
    dims = np.asarray(dimensions, dtype=int)
    if np.array_equal(dims, [1, 0, 0]):
        axes = (1, 2)
    elif np.array_equal(dims, [0, 1, 0]):
        axes = (0, 2)
    elif np.array_equal(dims, [0, 0, 1]):
        axes = (0, 1)
    else:
        raise ValueError("`dimensions` must be a unit vector along x, y, or z axis")

    return rotate(
        SI,
        angle=theta,
        axes=axes,
        reshape=False,
        order=1,
        mode='constant',
        cval=0.0
    )
