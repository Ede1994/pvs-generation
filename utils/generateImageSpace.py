#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module computes the 3D Fourier transform from k-space to image space.
Version:    1.0.0
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
from numpy.fft import fftn, fftshift, ifftshift

# ***************************************************************************
# * Function
# ***************************************************************************
def generate_image_space(k_space: np.ndarray
                         ) -> np.ndarray:
    r"""
    Compute 3D Fourier transform from k-space to image space.

    Parameters:
    ----------
    k_space: np.ndarray
        3D numpy array representing k-space data.

    Returns:
    -------
    SI: np.ndarray
        Complex-valued image space representation.
    """
    return fftshift(fftn(ifftshift(k_space)))
