#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module generates a PVS (perivascular space) creator function
with specified characteristics.
Version:    1.0.0
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Sequence, Callable


# ***************************************************************************
# * Function
# ***************************************************************************
def create_PVS_creator(length: float,
                       width: float,
                       NVox: Sequence[int]
                       ) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Generate a PVS (perivascular space) creator function with specified characteristics.

    Parameters
    ----------
    length : float
        PVS length.
    width : float
        PVS width.
    NVox : sequence of int
        Spatial dimensions [nx, ny, nz] of the true FOV region in voxels.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        Function that, given a 3x3 orientation matrix, returns a 3D boolean mask
        of the PVS, matching the grid defined by NVox.
    """
    NVox = np.array(NVox, dtype=int)
    half_size = NVox // 2

    # Define voxel grids (match MATLAB ndgrid order)
    vx = np.arange(-half_size[1], half_size[1] + 1)
    vy = np.arange(-half_size[0], half_size[0] + 1)
    vz = np.arange(-half_size[2], half_size[2] + 1)
    x_vox, y_vox, z_vox = np.meshgrid(vx, vy, vz, indexing='ij')

    # Binary cylinder mask
    PVS = ((x_vox**2 + y_vox**2) <= (width / 2)**2) & (np.abs(z_vox) <= length / 2)
    PVS = PVS.astype(float)

    # Create interpolant for sub-voxel rotation
    interpolant = RegularGridInterpolator(
        (vx, vy, vz),
        PVS,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # Sampling points for interpolation
    sampling_points = np.vstack([
        x_vox.ravel(),
        y_vox.ravel(),
        z_vox.ravel()
    ]).T

    def PVS_creator(orientation: np.ndarray) -> np.ndarray:
        """
        Create PVS mask given a rotation matrix.

        Parameters
        ----------
        orientation : ndarray
            3×3 rotation matrix.

        Returns
        -------
        ndarray
            3D boolean array of shape x_vox.shape indicating the rotated PVS.
        """
        # Rotate sampling points
        rot_pts = (orientation @ sampling_points.T).T
        # Interpolate and threshold
        values = interpolant(rot_pts)
        mask = values >= 0.5
        # Reshape to original grid
        return mask.reshape(x_vox.shape)

    return PVS_creator
