import numpy as np

def no_filter_1d(fov_mm_hr: float,
              res_mm_hr: float
              ) -> np.ndarray:
    r"""
    Creates a constant window (no filtering) over the high-resolution FoV.
    This is used when the low-resolution FoV is zero.

    Parameters
    ----------
    fov_mm_hr : float
        High-resolution field of view in mm.
    res_mm_hr : float
        High-resolution voxel size in mm.
    
    Returns
    -------
    ndarray
        1D constant window (no filtering).
    """
    num_points = int(round(fov_mm_hr / res_mm_hr))
    if num_points <= 0: # Handle edge case
        return np.array([])

    d_coords = np.arange(-fov_mm_hr / 2.0, fov_mm_hr / 2.0 - res_mm_hr + (res_mm_hr / 2.01) , res_mm_hr) # Small epsilon to include endpoint
    return np.ones_like(d_coords, dtype=float)

def butterworth_1d(fov_mm_hr: float,
                fov_mm_lr: float,
                res_mm_hr: float
                ) -> np.ndarray:
    r"""
    Creates a 1D Butterworth window with cutoff at FoV_mm_LR/2.

    Parameters
    ----------
    fov_mm_hr : float
        High-resolution field of view in mm.
    fov_mm_lr : float
        Low-resolution field of view in mm.
    res_mm_hr : float
        High-resolution voxel size in mm.

    Returns
    -------
    ndarray
        1D Butterworth filter profile.
    """
    # Creates a 1D Butterworth filter profile.
    d0 = fov_mm_lr / 2.0  # Cutoff frequency
    d_coords = np.arange(-fov_mm_hr / 2.0, fov_mm_hr / 2.0 - res_mm_hr + (res_mm_hr / 2.01) , res_mm_hr)

    if np.isclose(d0, 0): # Cutoff is zero
        w = np.zeros_like(d_coords, dtype=float)
        if d_coords.size > 0:
             dc_idx = np.argmin(np.abs(d_coords)) # Index closest to 0 Hz
             if np.isclose(d_coords[dc_idx], 0):
                 w[dc_idx] = 1.0 # Pass DC if present
        return w

    max_abs_d = np.max(np.abs(d_coords)) if d_coords.size > 0 else 0.0

    if np.isclose(max_abs_d, 0) or (max_abs_d / d0) <= 1.000001:
        n_order = 20.0 # High order to approximate passband if ratio is ill-defined for log
    else:
        log_arg = max_abs_d / d0
        if log_arg <= 0 : # Should not happen due to previous check, but defensive
            n_order = 20.0
        else:
            n_order = np.log(1000.0) / np.log(log_arg)

    ratio_d_d0 = np.abs(d_coords / d0)
    # Handle cases where d_coords/d0 is 0 and n_order might be non-integer leading to NaN for 0**non_int
    # Power of 0 is 0 for positive exponent, 1 for exponent 0, undef for negative.
    # Here, n_order should be positive.
    denominator_term = np.power(ratio_d_d0, n_order)
    # Ensure 0^n is 0 if n > 0
    denominator_term[np.isclose(ratio_d_d0, 0) & (n_order > 0)] = 0.0

    w = 1.0 / (1.0 + denominator_term)
    return w

def create_window_3d(
    fov_mm_hr: np.ndarray,
    fov_mm_lr: np.ndarray,
    res_mm_hr: np.ndarray,
    is_modified: np.ndarray
) -> np.ndarray:
    r"""
    Builds a separable 3D window to avoid phase-warping on resampling.

    Parameters
    ----------
    fov_mm_hr : ndarray
        High-resolution field of view in mm.
    fov_mm_lr : ndarray
        Low-resolution field of view in mm.
    res_mm_hr : ndarray
        High-resolution voxel size in mm.
    is_modified : ndarray
        Boolean array indicating if the corresponding dimension is modified.

    Returns
    -------
    ndarray
        3D window array.
    """
    # Creates a 3D k-space window.
    # fov_mm_hr, fov_mm_lr, res_mm_hr, is_modified are 3-element arrays/lists.
    
    if is_modified[0]: # Corresponds to MATLAB's first dimension (often x or phase-encode)
        w1 = butterworth_1d(fov_mm_hr[0], fov_mm_lr[0], res_mm_hr[0])
    else:
        w1 = no_filter_1d(fov_mm_hr[0], res_mm_hr[0])
    
    if is_modified[1]: # Corresponds to MATLAB's second dimension (often y or frequency-encode)
        w2 = butterworth_1d(fov_mm_hr[1], fov_mm_lr[1], res_mm_hr[1])
    else:
        w2 = no_filter_1d(fov_mm_hr[1], res_mm_hr[1])
    
    # MATLAB's 3rd dim for window is always no_filter in original createLRPVSMaps.
    w3 = no_filter_1d(fov_mm_hr[2], res_mm_hr[2])

    if w1.size == 0 or w2.size == 0 or w3.size == 0:
        # If any dimension results in an empty filter (e.g., num_points was 0)
        # then the resulting 3D window will be empty or an error will occur.
        # Determine the expected NTrue from fov/res to create an empty array of correct shape.
        nx = int(round(fov_mm_hr[0] / res_mm_hr[0]))
        ny = int(round(fov_mm_hr[1] / res_mm_hr[1]))
        nz = int(round(fov_mm_hr[2] / res_mm_hr[2]))
        return np.zeros((ny, nx, nz)) # Note MATLAB's meshgrid order for W generation
    
    # W1 (y-dim), W2 (x-dim), W3 (z-dim)
    # For outer product: w1 needs to be (Ny, 1), w2 needs to be (1, Nx)
    # The resulting w_2d will be (Ny, Nx)
    w_2d = w1[:, np.newaxis] * w2[np.newaxis, :]
    
    # w_3d needs shape (Ny, Nx, Nz)
    w_3d = w_2d[:, :, np.newaxis] * w3[np.newaxis, np.newaxis, :]
    
    return w_3d

