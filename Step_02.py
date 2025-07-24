#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script generates high-resolution (HR) perivascular spaces (PVS) masks
from a tissue map, using a specified PVS volume space and dimensions.
Version:    0.0.1
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt as bwdist
from skimage.measure import label, regionprops_table
from skimage.morphology import ball
from tqdm import tqdm
from numba import njit

from utils.create_PVS_creator import create_PVS_creator
from utils.blockproc3 import blockproc3
from utils.summariseNeighbourhood import summarise_neighbourhood

# Import configuration and parameter settings
from config.conf import *
from config.params import *


# ***************************************************************************
# * Functions
# ***************************************************************************
@njit
def cross_prod_norm_numba(A, B):
    r"""
    Compute the norm of the cross product of two vectors A and B using Numba.

    Parameters
    ----------
    A : np.ndarray
        First vector (1D array).
    B : np.ndarray
        Second vector (1D array).
    
    Returns
    -------
    float
        The norm of the cross product of A and B.
    """
    return np.linalg.norm(np.cross(A, B))

@njit
def GG_numba(A, B):
    r"""
    Compute the Gram matrix for two vectors A and B using Numba.

    Parameters
    ----------
    A : np.ndarray
        First vector (1D array).
    B : np.ndarray
        Second vector (1D array).

    Returns
    -------
    np.ndarray
        The Gram matrix of A and B, which is a 3x3 matrix.
    """
    c = np.dot(A, B)
    s = cross_prod_norm_numba(A, B)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

@njit
def FFi_numba(A, B):
    r"""
    Compute the FFi matrix for two vectors A and B using Numba.

    Parameters
    ----------
    A : np.ndarray
        First vector (1D array).
    B : np.ndarray
        Second vector (1D array).

    Returns
    -------
    np.ndarray
        The FFi matrix, which is a 3x3 matrix.
    """
    v1 = A
    v2_num = B - np.dot(A, B) * A
    norm_v2 = np.linalg.norm(v2_num)
    if norm_v2 < 1e-6: # Collinear case A = +/- B
        # This fallback to eye(3) is problematic if A = -B, as U will not be a 180-deg rotation.
        return np.eye(3, dtype=np.float64)
    v2 = v2_num / norm_v2
    v3 = np.cross(B, A)
    # Ensure column_stack output is consistently float64 for numba type stability
    return np.column_stack((v1.astype(np.float64), v2.astype(np.float64), v3.astype(np.float64)))


@njit
def UU_numba(Fi, G):
    r"""
    Compute the UU matrix for two matrices Fi and G using Numba.

    Parameters
    ----------
    Fi : np.ndarray
        The FFi matrix (3x3).
    G : np.ndarray
        The Gram matrix (3x3).

    Returns
    -------
    np.ndarray
        The UU matrix, which is a 3x3 matrix.
    """
    # Fi and G are assumed to be float64 numpy arrays as output by FFi_numba and GG_numba
    try:
        # The .astype(np.float64) for Fi is technically redundant if FFi_numba ensures float64 output,
        # but it's harmless.
        inv_Fi = np.linalg.inv(Fi.astype(np.float64))
    except Exception:  # Changed from np.linalg.LinAlgError to Exception
        # This block will now be Numba-compatible.
        # It will catch LinAlgError as well as any other error during inv_Fi.
        return np.eye(3, 3, k=0, dtype=np.float64) #np.eye(3, dtype=np.float64)
    return Fi @ G @ inv_Fi


# ***************************************************************************
# * Main
# ***************************************************************************
if __name__ == "__main__":
    # loading images
    nii = nib.load(MIDA_tissue_map_fname)
    tissue_map = nii.get_fdata().astype(np.uint8)
    affine = nii.affine
    header = nii.header

    # Construct HR brain mask
    roi = np.isin(tissue_map, [3, 4, 7])
    #roi = bwdist(~roi) >= 4
    nib.save(nib.Nifti1Image(roi.astype(np.uint8), affine, header), HR_ROI_mask_fname)

    for iCase in tqdm(range(len(lengths)), desc="Cases"):
        for rep in range(NRep):
            # print(f"Processing Case {iCase+1}/{len(lengths)}, Rep {rep+1}/{NRep}...")
            
            _pvs_generator_callable = create_PVS_creator(lengths[iCase], widths[iCase], PVS_vol_space)

            # print("Generating feasible locations...")
            feasible_locations_mask = roi * (np.random.rand(*roi.shape) >= 0.5)
            block_size = tuple(np.ceil(np.array(PVS_vol_space) * 0.7).astype(int))
            
            feasible_locations_processed = blockproc3(
                feasible_locations_mask,
                block_size,
                summarise_neighbourhood,
                border=(0, 0, 0) 
            )
            feasible_voxels_coords = np.argwhere(feasible_locations_processed == 1)

            current_PVS_mask = np.zeros_like(roi, dtype=np.uint8)
            
            a_canonical = np.array([0.0, 0.0, 1.0]) # Canonical PVS orientation (along Z-axis)
            
            # Define half_size based on PVS_vol_space for patch placement
            # This defines the extent from the center voxel for a patch of size 2*half_size+1
            half_size_patch = np.floor(np.array(PVS_vol_space)/2).astype(int)

            # print(f"Generating PVS for {feasible_voxels_coords.shape[0]} feasible locations...")
            for voxelIdx in tqdm(range(feasible_voxels_coords.shape[0]), desc=f"Case {iCase+1} Rep {rep+1} Voxels", leave=False):
                vx, vy, vz = feasible_voxels_coords[voxelIdx]

                target_point = np.array([240.0, 240.0, 240.0]) # As in MATLAB code (0-indexed)
                b_vec_direction = target_point - np.array([vx, vy, vz], dtype=np.float64)
                
                norm_b_vec = np.linalg.norm(b_vec_direction)
                Rotation_Matrix_R_for_PVS_template = np.eye(3, dtype=np.float64) # Default to no rotation

                if norm_b_vec > 1e-6:
                    b_vec_normalized = b_vec_direction / norm_b_vec
                    
                    # The user's formulation U = UU(FFi(b,a), GG(b,a)) results in U*b = a.
                    # (where b is b_vec_normalized, a is a_canonical)
                    # We need R such that R*a_canonical = b_vec_normalized. So R = U.T.
                    
                    # Critical: Check for 180-degree case where FFi_numba fallback is problematic.
                    if np.dot(b_vec_normalized, a_canonical) < -0.9999: # b_vec is nearly -a_canonical
                        # FFi_numba(b,a) -> eye(3). GG_numba(b,a) -> diag specific. U_b_to_a -> diag specific.
                        # U_b_to_a.T * a_canonical will likely NOT be b_vec_normalized.
                        # This case requires a proper 180-degree rotation matrix construction.
                        # For simplicity here, we can use scipy if available, otherwise issue warning.
                        try:
                            from scipy.spatial.transform import Rotation as SciPyRotation
                            # align_vectors(vectors_from, vectors_to)
                            # We want to align a_canonical TO b_vec_normalized
                            Rotation_Matrix_R_for_PVS_template, _ = SciPyRotation.align_vectors(a_canonical.reshape(1,-1), b_vec_normalized.reshape(1,-1))
                            Rotation_Matrix_R_for_PVS_template = Rotation_Matrix_R_for_PVS_template[0]
                        except ImportError:
                            # print("Warning: scipy.spatial.transform not available for robust 180-deg rotation. Using FFi/GG/UU which may be inexact.")
                            # Fallback to original method (U.T), acknowledging potential issue for 180-deg
                            Fi_mat = FFi_numba(b_vec_normalized, a_canonical)
                            G_mat = GG_numba(b_vec_normalized, a_canonical)
                            U_b_to_a = UU_numba(Fi_mat, G_mat)
                            Rotation_Matrix_R_for_PVS_template = U_b_to_a.T
                    else: # Standard case
                        Fi_mat = FFi_numba(b_vec_normalized, a_canonical)
                        G_mat = GG_numba(b_vec_normalized, a_canonical)
                        U_b_to_a = UU_numba(Fi_mat, G_mat)
                        Rotation_Matrix_R_for_PVS_template = U_b_to_a.T
                
                # else: voxel is at target_point, use default identity rotation (PVS along a_canonical)

                pvs_template_rotated = _pvs_generator_callable(Rotation_Matrix_R_for_PVS_template)
                
                # Calculate slices for inserting the PVS patch (centered at vx,vy,vz)
                xmin = vx - half_size_patch[0]
                xmax_plus_1 = vx + half_size_patch[0] + 1
                ymin = vy - half_size_patch[1]
                ymax_plus_1 = vy + half_size_patch[1] + 1
                zmin = vz - half_size_patch[2]
                zmax_plus_1 = vz + half_size_patch[2] + 1

                full_mask_shape = current_PVS_mask.shape
                if not (xmin >= 0 and xmax_plus_1 <= full_mask_shape[0] and
                        ymin >= 0 and ymax_plus_1 <= full_mask_shape[1] and
                        zmin >= 0 and zmax_plus_1 <= full_mask_shape[2]):
                    continue

                patch_slice = (slice(xmin, xmax_plus_1),
                            slice(ymin, ymax_plus_1),
                            slice(zmin, zmax_plus_1))
                
                slice_shape_from_indices = tuple(s.stop - s.start for s in patch_slice)
                if pvs_template_rotated.shape != slice_shape_from_indices:
                    # This should not happen if half_size_patch correctly corresponds to template_shape from create_pvs_creator
                    # print(f"Shape mismatch: PVS {pvs_template_rotated.shape}, Slice {slice_shape_from_indices}. Skipping.")
                    continue
                
                total_voxels_in_pvs = np.sum(pvs_template_rotated)
                if total_voxels_in_pvs == 0:
                    continue

                mask_sub_patch = current_PVS_mask[patch_slice]
                roi_sub_patch = roi[patch_slice]

                check_intersection = np.sum(pvs_template_rotated & mask_sub_patch)
                check_within_ROI = np.sum(pvs_template_rotated & roi_sub_patch)

                if check_intersection == 0 and check_within_ROI == total_voxels_in_pvs:
                    current_PVS_mask[patch_slice] = np.logical_or(mask_sub_patch, pvs_template_rotated)
            
            current_PVS_mask[~roi] = 0

            # print("Filtering PVS by volume...")
            labeled_mask, num_features = label(current_PVS_mask, connectivity=3, return_num=True)
            mask_corrected_PVS = np.zeros_like(current_PVS_mask, dtype=bool)

            if num_features > 0:
                props = regionprops_table(labeled_mask, properties=('label', 'area'))
                min_volume_voxels = lengths[iCase] * (widths[iCase] / 2)**2 * np.pi
                
                valid_labels = [p_label for p_label, p_area in zip(props['label'], props['area']) if p_area >= min_volume_voxels]
                if valid_labels:
                    mask_corrected_PVS = np.isin(labeled_mask, valid_labels)
            
            output_fname = HR_PVS_map_output_pattern.format(iCase, rep)
            
            header_out = header.copy()
            header_out.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(mask_corrected_PVS.astype(np.uint8), affine, header_out), output_fname)
            # print(f"Saved PVS mask to {output_fname}")

    print("Processing complete.")
