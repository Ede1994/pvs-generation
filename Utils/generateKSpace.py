import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift

def generate_k_space(SI: np.ndarray
                     ) -> np.ndarray:
    r"""
    Compute 3D inverse Fourier transform from image space to k-space.

    Parameters:
    - SI: np.ndarray
        3D numpy array representing the image space data.

    Returns:
    - k_space: np.ndarray
        Complex-valued k-space representation.
    """
    return fftshift(ifftn(ifftshift(SI)))
