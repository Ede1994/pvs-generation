#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module resamples high-resolution (HR) k-space data to low-resolution (LR) k-space.
Version:    1.0.0
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
from scipy.interpolate import interpn
from utils.createWindow3D import create_window_3d
from utils.generateKSpace import generate_k_space


# ***************************************************************************
# * Function
# ***************************************************************************
def resample_kspace(hr_si: np.ndarray,
                    fov_mm_true: np.ndarray,
                    n_true: np.ndarray,
                    fov_mm_acq: np.ndarray,
                    n_acq: np.ndarray
                    ) -> np.ndarray:
    r"""
    Resample high-resolution (HR) k-space data to low-resolution (LR) k-space.

    Parameters
    ----------
    hr_si : np.ndarray
        High-resolution image space data, can be 3D or 4D.
    fov_mm_true : np.ndarray
        Field of View in mm for the true resolution, should be a 3-element array [FOVx, FOVy, FOVz].
    n_true : np.ndarray
        Number of samples in the true resolution, should be a 3-element array [Nx, Ny, Nz].
    fov_mm_acq : np.ndarray
        Field of View in mm for the acquisition resolution, should be a 3-element array [FOVx, FOVy, FOVz].
    n_acq : np.ndarray
        Number of samples in the acquisition resolution, should be a 3-element array [Nx, Ny, Nz].
        
    Returns
    -------
    np.ndarray
        Low-resolution k-space data after resampling.
    """
    # Resamples HR image data to LR k-space.
    # Ensure inputs are numpy arrays for vectorized ops
    fov_mm_true = np.array(fov_mm_true)
    n_true = np.array(n_true)
    fov_mm_acq = np.array(fov_mm_acq)
    n_acq = np.array(n_acq)

    res_mm_true = fov_mm_true / n_true    # HR resolution
    res_mm_acq = fov_mm_acq / n_acq    # LR resolution

    k_fov_per_mm_hr = 1.0 / res_mm_true  # HR k-space FoV extent
    k_res_per_mm_hr = 1.0 / fov_mm_true  # HR k-space resolution (sampling interval)
    k_fov_per_mm_lr = 1.0 / res_mm_acq  # LR k-space FoV extent
    k_res_per_mm_lr = 1.0 / fov_mm_acq  # LR k-space resolution

    # HR k-space coordinates for interpolation source
    ky_khr_coords = np.arange(-k_fov_per_mm_hr[1]/2.0, k_fov_per_mm_hr[1]/2.0 - k_res_per_mm_hr[1] + k_res_per_mm_hr[1]/2.01, k_res_per_mm_hr[1])
    kx_khr_coords = np.arange(-k_fov_per_mm_hr[0]/2.0, k_fov_per_mm_hr[0]/2.0 - k_res_per_mm_hr[0] + k_res_per_mm_hr[0]/2.01, k_res_per_mm_hr[0])
    kz_khr_coords = np.arange(-k_fov_per_mm_hr[2]/2.0, k_fov_per_mm_hr[2]/2.0 - k_res_per_mm_hr[2] + k_res_per_mm_hr[2]/2.01, k_res_per_mm_hr[2])
    
    # LR k-space coordinates for interpolation query points
    ky_klr_coords = np.arange(-k_fov_per_mm_lr[1]/2.0, k_fov_per_mm_lr[1]/2.0 - k_res_per_mm_lr[1] + k_res_per_mm_lr[1]/2.01, k_res_per_mm_lr[1])
    kx_klr_coords = np.arange(-k_fov_per_mm_lr[0]/2.0, k_fov_per_mm_lr[0]/2.0 - k_res_per_mm_lr[0] + k_res_per_mm_lr[0]/2.01, k_res_per_mm_lr[0])
    kz_klr_coords = np.arange(-k_fov_per_mm_lr[2]/2.0, k_fov_per_mm_lr[2]/2.0 - k_res_per_mm_lr[2] + k_res_per_mm_lr[2]/2.01, k_res_per_mm_lr[2])

    # Create query points grid for interpn
    kx_klr_grid, ky_klr_grid, kz_klr_grid = np.meshgrid(kx_klr_coords, ky_klr_coords, kz_klr_coords, indexing='ij')
    lr_query_points = np.stack([kx_klr_grid.ravel(), ky_klr_grid.ravel(), kz_klr_grid.ravel()], axis=-1)

    sf = np.prod(fov_mm_true / fov_mm_acq)  # Scaling factor

    ### Windowing and RoI masking in image space
    # Create HR image space coordinates for FoV masking (is_within_LR_FoV)
    # Assuming hr_si is (Nx, Ny, Nz)
    x_hr_img_coords_vec = np.arange(-fov_mm_true[0]/2.0, fov_mm_true[0]/2.0 - res_mm_true[0] + res_mm_true[0]/2.01, res_mm_true[0])
    y_hr_img_coords_vec = np.arange(-fov_mm_true[1]/2.0, fov_mm_true[1]/2.0 - res_mm_true[1] + res_mm_true[1]/2.01, res_mm_true[1])
    z_hr_img_coords_vec = np.arange(-fov_mm_true[2]/2.0, fov_mm_true[2]/2.0 - res_mm_true[2] + res_mm_true[2]/2.01, res_mm_true[2])
    x_hr_img_grid, y_hr_img_grid, z_hr_img_grid = np.meshgrid(x_hr_img_coords_vec, y_hr_img_coords_vec, z_hr_img_coords_vec, indexing='ij')

    # LR FoV limits for masking
    x_mm_lr_lims = [-fov_mm_acq[0]/2.0, fov_mm_acq[0]/2.0 - res_mm_acq[0]]
    y_mm_lr_lims = [-fov_mm_acq[1]/2.0, fov_mm_acq[1]/2.0 - res_mm_acq[1]]
    z_mm_lr_lims = [-fov_mm_acq[2]/2.0, fov_mm_acq[2]/2.0 - res_mm_acq[2]]

    is_within_lr_fov_mask = (
        (x_hr_img_grid >= np.min(x_mm_lr_lims)) & (x_hr_img_grid <= np.max(x_mm_lr_lims)) &
        (y_hr_img_grid >= np.min(y_mm_lr_lims)) & (y_hr_img_grid <= np.max(y_mm_lr_lims)) &
        (z_hr_img_grid >= np.min(z_mm_lr_lims)) & (z_hr_img_grid <= np.max(z_mm_lr_lims))
    )
    
    is_modified_flags = fov_mm_true > fov_mm_acq # Boolean flags for filtering

    window_3d = create_window_3d(fov_mm_true, fov_mm_acq, res_mm_true, is_modified_flags)
    
    # Apply window and FoV mask
    hr_si_windowed = (hr_si * window_3d) * is_within_lr_fov_mask
    
    hr_k = generate_k_space(hr_si_windowed) # Compute k-space of windowed HR data

    lr_k_flat = interpn(
        (kx_khr_coords, ky_khr_coords, kz_khr_coords), # Points defining the hr_k grid
        hr_k,                                          # Values on the hr_k grid
        lr_query_points,                               # Points to interpolate at
        method="linear",
        bounds_error=False, fill_value=0.0 # Fill with 0 outside original k-space bounds
    )

    lr_k_reshaped = lr_k_flat.reshape((len(kx_klr_coords), len(ky_klr_coords), len(kz_klr_coords)))
    
    return sf * lr_k_reshaped
