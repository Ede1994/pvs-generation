#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module adds noise to k-space data.
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
def add_noise_kspace(k_space: np.ndarray,
                     sd_noise: float,
                     n_acq: np.ndarray
                     ) -> np.ndarray:
    r"""
    Add noise to k-space
    Add noise to input k-space.
    We represent the signal-to-noise ratio (SNR) as the quotient
    between the mean signal value and the standard deviation of the
    background noise. The SNR of the real scans should be similar to that of
    our simulations, i.e. SNR_real = SNR_sim or mu_real/SD_real =
    mu_sim/SD_sim. Thus, the standard deviation of the noise in
    our simulations should be equal to (mu_sim*SD_real)/mu_real. First, we 
    estimated the standard deviation of the noise in real scans by computing 
    the mean signal within the normal-appearing white matter region and the
    standard deviation of the noise from background area. Second, we divide
    the estimated standard deviation by sqrt(2-pi/2) and by the square root
    of the number of points.
    
    Parameters
    ----------
    k_space : np.ndarray
        The input k-space data. It is expected to be a 3D or 4D array.
    sd_noise : float
        The standard deviation of the noise to be added to the k-space data.
        This value is adjusted based on the number of voxels in the k-space.
    n_acq : np.ndarray
        The acquisition matrix size for the k-space. It should be a 1D array of length 3, representing (Nx, Ny, Nz).

    Returns
    -------
    np.ndarray
        The k-space data with added noise. The shape of the output array is the same as the input k-space.
    """
    if sd_noise == 0:
        return k_space
    
    # Noise standard deviation for real and imaginary parts
    noise_std_per_component = sd_noise / np.sqrt(2.0)
    
    real_noise = np.random.normal(0, noise_std_per_component, k_space.shape)
    imag_noise = np.random.normal(0, noise_std_per_component, k_space.shape)
    
    complex_noise = real_noise + 1j * imag_noise
    
    return k_space + complex_noise
