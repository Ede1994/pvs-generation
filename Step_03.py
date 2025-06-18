#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script processes high-resolution (HR) tissue probability maps and computes
high-resolution (HR) signal intensity (SI) maps based on the provided parameters.
Version:    0.0.1
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt as bwdist
from scipy.ndimage import gaussian_filter as imgaussfilt3
import os
from tqdm import tqdm
from numba import njit

# Import configuration and parameter settings
from config.conf import *
from config.params import *


# ***************************************************************************
# * Functions
# ***************************************************************************
@njit
def process_probabilities_and_si_numba(
    HR_tissue_prob_input: np.ndarray,
    dGM_dist_final_in: np.ndarray,
    dGM_mask_map_in: np.ndarray,
    PVS_map_in: np.ndarray,
    SI_in: np.ndarray,
    dGM_py_idx_in: int,
    NAWM_py_idx_in: int,
    N_true_tuple_in: tuple,
    NumRegions_in: int
    ) -> np.ndarray:
    r"""
    Process probabilities and compute HR Signal Intensity (SI) map.
    This function updates the HR_tissue_prob based on dGM distance and computes
    the final HR SI map using the provided signal intensities.

    Parameters:
    ----------
    HR_tissue_prob_input : np.ndarray
        Initial probabilities for each tissue region (shape: N_true_tuple_in + (NumRegions_in,)).
    dGM_dist_final_in : np.ndarray
        Smoothed distance map for dGM regions (shape: N_true_tuple_in).
    dGM_mask_map_in : np.ndarray
        Mask for dGM regions (shape: N_true_tuple_in).
    PVS_map_in : np.ndarray
        Binary map indicating PVS regions (shape: N_true_tuple_in).
    SI_in : np.ndarray
        Signal intensities for each tissue region (shape: (NumRegions_in,)).
    dGM_py_idx_in : int
        0-based index for dGM region in HR_tissue_prob.
    NAWM_py_idx_in : int
        0-based index for NAWM region in HR_tissue_prob.
    N_true_tuple_in : tuple
        Tuple of integers representing the spatial dimensions of the HR map (e.g., (240, 240, 48)).
    NumRegions_in : int
        Total number of tissue regions (including dGM and NAWM).
    
    Returns:
    -------
    HR_SI : np.ndarray
        High Resolution Signal Intensity map (shape: N_true_tuple_in).
    """
    HR_tissue_prob = HR_tissue_prob_input.copy()

    sum_HR_tissue_prob_before_dGM_update = np.sum(HR_tissue_prob, axis=3)
    
    for i in range(HR_tissue_prob.shape[0]):
        for j in range(HR_tissue_prob.shape[1]):
            for k in range(HR_tissue_prob.shape[2]):
                HR_tissue_prob[i, j, k, dGM_py_idx_in] = \
                    dGM_dist_final_in[i, j, k] * (2.0 - sum_HR_tissue_prob_before_dGM_update[i, j, k])
    
    sum_HR_tissue_prob_after_dGM_update = np.sum(HR_tissue_prob, axis=3)
    
    for i in range(HR_tissue_prob.shape[0]):
        for j in range(HR_tissue_prob.shape[1]):
            for k in range(HR_tissue_prob.shape[2]):
                HR_tissue_prob[i, j, k, NAWM_py_idx_in] += \
                    dGM_mask_map_in[i, j, k] * (1.0 - sum_HR_tissue_prob_after_dGM_update[i, j, k])

    ### Normalize HR_tissue_prob safely
    # Calculate sum along axis 3, resulting in a 3D array.
    sum_total_prob_3d = np.sum(HR_tissue_prob, axis=3) # No keepdims

    for r_idx in range(HR_tissue_prob.shape[0]):
        for c_idx in range(HR_tissue_prob.shape[1]):
            for s_idx in range(HR_tissue_prob.shape[2]):
                current_sum = sum_total_prob_3d[r_idx, c_idx, s_idx] # Access the 3D sum array
                if current_sum != 0.0:
                    for region_k_idx in range(HR_tissue_prob.shape[3]): # NumRegions_in
                        HR_tissue_prob[r_idx, c_idx, s_idx, region_k_idx] /= current_sum
                else:
                    for region_k_idx in range(HR_tissue_prob.shape[3]):
                         HR_tissue_prob[r_idx, c_idx, s_idx, region_k_idx] = 0.0
    
    HR_SI = np.zeros((N_true_tuple_in[0], N_true_tuple_in[1], N_true_tuple_in[2]), dtype=np.float64)
    
    for iRegion_py_idx in range(NumRegions_in):
        for i in range(HR_SI.shape[0]):
            for j in range(HR_SI.shape[1]):
                for k in range(HR_SI.shape[2]):
                    HR_SI[i, j, k] += HR_tissue_prob[i, j, k, iRegion_py_idx] * SI_in[iRegion_py_idx]

    PVS_inner_dist_float = PVS_map_in.astype(np.float64)
    
    for i in range(HR_SI.shape[0]):
        for j in range(HR_SI.shape[1]):
            for k in range(HR_SI.shape[2]):
                HR_SI[i, j, k] = HR_SI[i, j, k] * (1.0 - PVS_inner_dist_float[i, j, k]) + \
                                 SI_in[NumRegions_in - 1] * PVS_inner_dist_float[i, j, k]
            
    return HR_SI


# ***************************************************************************
# * Main
# ***************************************************************************
if __name__ == "__main__":
    ### Loading images and initial setup
    nii_info = nib.load(MIDA_tissue_map_fname)
    tissue_map_orig = nii_info.get_fdata().astype(np.uint8)
    pat_tissue_map_orig = nib.load(MIDA_tissue_map_ageing_fname).get_fdata().astype(np.uint8)
    affine = nii_info.affine
    header = nii_info.header.copy()

    header.set_data_dtype(np.float64)

    dGM_py_idx = dGM_class_idx - 1
    NAWM_py_idx = NAWM_class_idx - 1

    # Pre-convert N_true to a tuple of integers for Numba
    # N_true is likely ['240', '240', '48'] from Params.py
    N_true_int_tuple = tuple(map(int, N_true))
    # Ensure SI is a NumPy array of float64 for Numba
    SI_np_array = np.array(SI, dtype=np.float64)


    ### Main processing loop for each case and repetition
    for iCase_loop_idx in tqdm(range(len(lengths)), desc="Cases"):
        for rep_loop_idx in range(NRep):
            current_iCase_for_filename = iCase_loop_idx
            current_rep_for_filename = rep_loop_idx

            input_fname = HR_PVS_map_output_pattern.format(current_iCase_for_filename, current_rep_for_filename)
            output_fname = HR_SI_output_pattern.format(current_iCase_for_filename, current_rep_for_filename)
            
            output_fname_with_ext = output_fname
            if not output_fname_with_ext.endswith('.nii.gz'):
                output_fname_with_ext = output_fname + '.nii.gz'

            if os.path.exists(output_fname_with_ext):
                continue
            
            if not os.path.exists(input_fname):
                print(f"PVS input file {input_fname} not found, skipping Case {current_iCase_for_filename} Rep {current_rep_for_filename}.")
                continue

            PVS_map = nib.load(input_fname).get_fdata() > 0 # Boolean array

            current_tissue_map = pat_tissue_map_orig if use_MIDA_ageing_mask else tissue_map_orig

            if use_MIDA_ageing_mask:
                smoothness_values = [1, 1, 1, 1, 0, 1, 0, 1, 0]
            else:
                smoothness_values = [1, 1, 1, 0, 0, 1, 0, 1, 0]
            
            ### Initial HR_tissue_prob calculation
            HR_tissue_prob = np.zeros(N_true_int_tuple + (int(NumRegions),), dtype=float)

            for iRegion_py_idx in range(NumRegions):
                tissue_label_in_map = iRegion_py_idx + 1
                mask = (current_tissue_map == tissue_label_in_map).astype(float)
                if smoothness_values[iRegion_py_idx] == 0:
                    HR_tissue_prob[:, :, :, iRegion_py_idx] = mask
                else:
                    HR_tissue_prob[:, :, :, iRegion_py_idx] = imgaussfilt3(mask, smoothness_values[iRegion_py_idx])
            
            ### dGM distance calculations
            dGM_mask_map = (current_tissue_map == dGM_class_idx).astype(float)
            NAWM_interface_mask = (current_tissue_map == NAWM_class_idx) * \
                                (bwdist( (current_tissue_map == dGM_class_idx) ) <= 1)
            dGM_dist_calc = dGM_mask_map * bwdist(NAWM_interface_mask > 0.5)

            max_val_dGM_dist = np.max(dGM_dist_calc)
            if max_val_dGM_dist > 0:
                dGM_dist_calc_norm = dGM_dist_calc / max_val_dGM_dist
            else:
                dGM_dist_calc_norm = np.zeros_like(dGM_dist_calc, dtype=float)
            dGM_dist_final = imgaussfilt3(dGM_dist_calc_norm, 1.0)

            ### Call Numba JIT-compiled function for core logic
            HR_SI = process_probabilities_and_si_numba(
                HR_tissue_prob,      # Input: initial probabilities
                dGM_dist_final,      # Input: dGM distance map
                dGM_mask_map,        # Input: dGM mask
                PVS_map,             # Input: PVS map (boolean)
                SI_np_array,         # Input: Signal intensities (NumPy array)
                dGM_py_idx,          # Input: 0-based dGM index
                NAWM_py_idx,         # Input: 0-based NAWM index
                N_true_int_tuple,    # Input: Tuple of spatial dimensions
                NumRegions           # Input: Number of regions
            )
                
            ### Save output         
            nib.save(nib.Nifti1Image(HR_SI.astype(np.float64), affine, header), output_fname)

    print("Processing complete.")

