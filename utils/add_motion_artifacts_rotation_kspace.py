#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module adds motion artifacts to k-space data by combining segments from different k-spaces.
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
def add_motion_artifacts_rotation_kspace(k_spaces: np.ndarray,
                                         n_true_acq: np.ndarray
                                         ) -> np.ndarray:
    r"""
    Adds motion artifacts to k-space data by combining segments from different k-spaces.
    The function takes a 4D k-space array and randomly selects segments from it to create a composite k-space.
    The composite k-space is constructed by selecting segments from the provided k-spaces, simulating motion artifacts.
    The function assumes that the input k_spaces is either 3D (no motion) or 4D (multiple states for motion).

    Parameters
    ----------
    k_spaces : np.ndarray
        The input k-space data. It can be 3D (no motion) or 4D (multiple states for motion).
        The shape of the array is expected to be (Nx, Ny, Nz) for 3D or (Nx, Ny, Nz, N_states) for 4D.
    n_true_acq : np.ndarray
        The acquisition matrix size for the composite k-space. It should be a 1D array of length 3, representing (Nx, Ny, Nz).
    
    Returns
    -------
    np.ndarray
        The composite k-space data with motion artifacts added. The shape of the output array is (Nx, Ny, Nz).
    """
    
    if k_spaces.ndim <= 3: # No multiple k-spaces provided, return as is
        return k_spaces
    else:
        
        nx, ny, nz = int(n_true_acq[0]), int(n_true_acq[1]), int(n_true_acq[2])
        n_k_space_lines = nz * nx # Total lines to fill (outer loops over slices then lines)

        # Determine segment lengths randomly (as in MATLAB)
        # randi([low, high]) -> np.random.randint(low, high+1)
        k_space_lines_second_segment = np.random.randint(n_k_space_lines // 2, n_k_space_lines + 1)
        remaining_lines = n_k_space_lines - k_space_lines_second_segment
        
        if remaining_lines < 0: remaining_lines = 0 # Should not happen if logic is sound

        k_space_lines_first_segment = np.random.randint(0, (remaining_lines // 2) + 1)
        k_space_lines_third_segment = n_k_space_lines - k_space_lines_first_segment - k_space_lines_second_segment
        
        if k_space_lines_third_segment < 0: # Adjust if rounding/logic leads to negative
            k_space_lines_third_segment = 0

        composite_k_space = np.zeros((nx, ny, nz), dtype=k_spaces.dtype)
        line_count = 0

        # This means outer loop is over x-dimension (lines), inner loop over z-dimension (slices)
        for i_slice_idx in range(nz): # Loop over slices (z-dimension)
            for i_line_idx in range(nx): # Loop over lines (x-dimension)
                if line_count < k_space_lines_first_segment:
                    composite_k_space[i_line_idx, :, i_slice_idx] = k_spaces[i_line_idx, :, i_slice_idx, 0]
                elif line_count < (k_space_lines_first_segment + k_space_lines_second_segment):
                    composite_k_space[i_line_idx, :, i_slice_idx] = k_spaces[i_line_idx, :, i_slice_idx, 1]
                else:
                    if k_spaces.shape[3] > 2: # If a third motion state exists
                        composite_k_space[i_line_idx, :, i_slice_idx] = k_spaces[i_line_idx, :, i_slice_idx, 2]
                    else: # Fallback to the second state if only two are provided
                         composite_k_space[i_line_idx, :, i_slice_idx] = k_spaces[i_line_idx, :, i_slice_idx, 1]

                line_count += 1
        return composite_k_space
