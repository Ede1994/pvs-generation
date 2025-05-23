import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def generate_image_space(k_space: np.ndarray
                         ) -> np.ndarray:
    r"""
    Compute 3D Fourier transform from k-space to image space.

    Parameters:
    - k_space: np.ndarray
        3D numpy array representing k-space data.

    Returns:
    - SI: np.ndarray
        Complex-valued image space representation.
    """
    return fftshift(fftn(ifftshift(k_space)))
