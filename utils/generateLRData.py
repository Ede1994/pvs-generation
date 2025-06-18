
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module generates Low Resolution (LR) data from High Resolution (HR) data.
Version:    1.0.0
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
from utils.resample import resample_kspace
from utils.generateImageSpace import generate_image_space
from utils.add_noise import add_noise_kspace
from utils.add_motion_artifacts_rotation_kspace import add_motion_artifacts_rotation_kspace


# ***************************************************************************
# * Function
# ***************************************************************************
def generate_lr_data(hr_si: np.ndarray,
                     fov_mm_true: list,
                     n_true: list,
                     sd_noise: float,
                     fov_mm_acq: list,
                     n_acq: list,
                     apply_noise_flag: int,
                     apply_motion_artefacts_flag: int
                   ) -> np.ndarray:
    r"""
    Generate Low Resolution (LR) data from High Resolution (HR) data.

    Parameters:
    ----------
    hr_si : np.ndarray
        High Resolution data in image space, can be 3D or 4D.
    fov_mm_true : list
        Field of View in mm for the true resolution, should be a list of 3 elements [FOVx, FOVy, FOVz].
    n_true : list
        Number of samples in the true resolution, should be a list of 3 elements [Nx, Ny, Nz].
    sd_noise : float
        Standard deviation of the noise to be added to the Low Resolution data.
    fov_mm_acq : list
        Field of View in mm for the acquisition resolution, should be a list of 3 elements [FOVx, FOVy, FOVz].
    n_acq : list
        Number of samples in the acquisition resolution, should be a list of 3 elements [Nx, Ny, Nz].
    apply_noise_flag : int
        Flag to indicate whether to apply noise to the Low Resolution data (1 for yes, 0 for no).
    apply_motion_artefacts_flag : int
        Flag to indicate whether to apply motion artefacts to the Low Resolution data (1 for yes, 0 for no).

    Returns:
    -------
    np.ndarray
        Low Resolution data in image space after resampling, noise addition, and motion artefacts application.
    """
    # Main function to generate Low Resolution data from High Resolution data.
    # Ensure inputs are numpy arrays
    fov_mm_true_np = np.array(fov_mm_true, dtype=float)
    n_true_np = np.array(n_true, dtype=int)
    fov_mm_acq_np = np.array(fov_mm_acq, dtype=float)
    n_acq_np = np.array(n_acq, dtype=int)

    # Ensure hr_si is 3D. If not, this pipeline part needs adjustment.
    if hr_si.ndim != 3:
        # For a direct translation assuming the input HR_SI might be 4D:
        if apply_motion_artefacts_flag and hr_si.ndim == 4 and hr_si.shape[3] >= 3:
             lr_k_space_state1 = resample_kspace(hr_si[:,:,:,0], fov_mm_true_np, n_true_np, fov_mm_acq_np, n_acq_np)
             lr_k_space_state2 = resample_kspace(hr_si[:,:,:,1], fov_mm_true_np, n_true_np, fov_mm_acq_np, n_acq_np)
             lr_k_space_state3 = resample_kspace(hr_si[:,:,:,2], fov_mm_true_np, n_true_np, fov_mm_acq_np, n_acq_np)
             # Stack them along a new 4th dimension for add_motion_artifacts
             lr_k_space_multistate = np.stack([lr_k_space_state1, lr_k_space_state2, lr_k_space_state3], axis=-1)
        elif hr_si.ndim == 4: # Only one state available from 4D, or motion not applied
            lr_k_space_multistate = resample_kspace(hr_si[:,:,:,0], fov_mm_true_np, n_true_np, fov_mm_acq_np, n_acq_np)
        elif hr_si.ndim == 3:
            lr_k_space_multistate = resample_kspace(hr_si, fov_mm_true_np, n_true_np, fov_mm_acq_np, n_acq_np)
        else:
            raise ValueError("hr_si must be 3D or 4D.")

    else: # No motion artifacts flag, HR_SI is likely 3D or only first frame used.
        current_hr_si_slice = hr_si
        if hr_si.ndim == 4 :
            current_hr_si_slice = hr_si[:,:,:,0] # Use first frame if 4D but no motion flag
        lr_k_space_multistate = resample_kspace(current_hr_si_slice, fov_mm_true_np, n_true_np, fov_mm_acq_np, n_acq_np)

    # Induce motion artifacts (if lr_k_space_multistate is 4D with appropriate states)
    lr_k_space_motion = add_motion_artifacts_rotation_kspace(lr_k_space_multistate, n_acq_np)
    
    # Add noise
    if apply_noise_flag:
        lr_k_space_acquired = add_noise_kspace(lr_k_space_motion, sd_noise, n_acq_np)
    else:
        lr_k_space_acquired = lr_k_space_motion

    # Transform to image space and take magnitude
    lr_si = np.abs(generate_image_space(lr_k_space_acquired))
    return lr_si
