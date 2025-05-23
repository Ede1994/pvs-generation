## Create HR SI maps
#  This script creates HR SI maps for simulation
#
# (c) Jose Bernal 2021

import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt as bwdist
from scipy.ndimage import gaussian_filter as imgaussfilt3
import os
from tqdm import tqdm

# Import configuration and parameter settings
from Conf import *
from Params import *

### Loading images and initial setup
# Load the MIDA tissue map (e.g., a standard brain segmentation)
nii_info = nib.load(MIDA_tissue_map_fname)
tissue_map_orig = nii_info.get_fdata().astype(np.uint8) # Original tissue map data, ensure uint8 type
# Load the MIDA tissue map for ageing (e.g., a segmentation representing an aged brain)
pat_tissue_map_orig = nib.load(MIDA_tissue_map_ageing_fname).get_fdata().astype(np.uint8) # Ageing tissue map data
affine = nii_info.affine # Get affine transformation from the NIFTI file
header = nii_info.header.copy() # Create a copy of the header to modify later

# Define a region of interest (ROI) including Normal Appearing White Matter (NAWM) and deep Gray Matter (dGM)
# This variable is defined but not explicitly used in the subsequent processing loop for generating HR_SI.
# included_roi = np.logical_or(tissue_map_orig == NAWM_class_idx, tissue_map_orig == dGM_class_idx)

# Set the data type in the output NIFTI header to float64 (equivalent to 'double' in MATLAB)
header.set_data_dtype(np.float64)

# Convert 1-based class indices (from Params.py, MATLAB convention) to 0-based Python indices
# These Python indices will be used for accessing elements in HR_tissue_prob and the SI array.
dGM_py_idx = dGM_class_idx - 1 # 0-based index for deep Gray Matter
NAWM_py_idx = NAWM_class_idx - 1 # 0-based index for Normal Appearing White Matter


### Main processing loop for each case and repetition
# Iterate over different cases (e.g., varying PVS characteristics defined by 'lengths')
for iCase_loop_idx in tqdm(range(len(lengths)), desc="Cases"):
    # Iterate over repetitions for each case
    for rep_loop_idx in range(NRep):
        # Use 0-based indexing for filenames, assuming PVS files are named like "..._Case_0_Rep_0"
        current_iCase_for_filename = iCase_loop_idx
        current_rep_for_filename = rep_loop_idx

        # Format input and output filenames based on case and repetition
        input_fname = HR_PVS_map_output_pattern.format(current_iCase_for_filename, current_rep_for_filename)
        output_fname = HR_SI_output_pattern.format(current_iCase_for_filename, current_rep_for_filename)
        
        # Ensure the output filename has the .nii.gz extension for existence check and saving
        output_fname_with_ext = output_fname
        if not output_fname_with_ext.endswith('.nii.gz'):
             output_fname_with_ext = output_fname + '.nii.gz'

        # Skip processing if the output file already exists
        if os.path.exists(output_fname_with_ext):
            # print(f"File {output_fname_with_ext} already exists, skipping.") # Optional: uncomment for verbosity
            continue
        
        # Skip if the required input PVS map file does not exist
        if not os.path.exists(input_fname):
            print(f"PVS input file {input_fname} not found, skipping Case {current_iCase_for_filename} Rep {current_rep_for_filename}.")
            continue

        # Load the PVS map (binary mask where PVS are True)
        PVS_map = nib.load(input_fname).get_fdata() > 0

        # Select the appropriate tissue map based on the 'use_MIDA_ageing_mask' flag
        current_tissue_map = pat_tissue_map_orig if use_MIDA_ageing_mask else tissue_map_orig

        # Define smoothness values for Gaussian filtering based on the selected tissue map type
        if use_MIDA_ageing_mask:
            smoothness_values = [1, 1, 1, 1, 0, 1, 0, 1, 0] # Smoothness for ageing map
        else:
            smoothness_values = [1, 1, 1, 0, 0, 1, 0, 1, 0] # Smoothness for standard map
        
        # Initialize a 4D array to store tissue probabilities (spatial dimensions + tissue types)
        HR_tissue_prob = np.zeros(tuple(map(int, N_true)) + (int(NumRegions),), dtype=float)

        ### Calculate initial tissue probabilities with smoothing
        # Iterate over each tissue region/type
        # iRegion_py_idx is the 0-based index for HR_tissue_prob and SI arrays.
        # tissue_label_in_map is the 1-based label value found in current_tissue_map.
        for iRegion_py_idx in range(NumRegions):
            # Convert 0-based Python index to 1-based tissue label (assumes labels are 1, 2, ..., NumRegions)
            tissue_label_in_map = iRegion_py_idx + 1
            
            # Create a binary mask for the current tissue label
            mask = (current_tissue_map == tissue_label_in_map).astype(float)
            
            # Apply Gaussian smoothing if the smoothness value for this region is not 0
            # smoothness_values is 0-indexed by iRegion_py_idx.
            if smoothness_values[iRegion_py_idx] == 0:
                HR_tissue_prob[:, :, :, iRegion_py_idx] = mask # No smoothing
            else:
                # Apply 3D Gaussian filter
                HR_tissue_prob[:, :, :, iRegion_py_idx] = imgaussfilt3(mask, smoothness_values[iRegion_py_idx])
        
        ### Calculate dGM distance and update tissue probabilities for dGM and NAWM
        # Create a mask for deep Gray Matter (dGM) using its 1-based class index
        dGM_mask_map = (current_tissue_map == dGM_class_idx).astype(float)
        # Create a mask for the interface area: NAWM voxels within 1 unit distance of dGM
        # bwdist calculates distance to the nearest True pixel.
        NAWM_interface_mask = (current_tissue_map == NAWM_class_idx) * \
                              (bwdist( (current_tissue_map == dGM_class_idx) ) <= 1) # Distance from dGM border
        
        # Calculate distances within dGM regions to this NAWM_interface_mask
        dGM_dist_calc = dGM_mask_map * bwdist(NAWM_interface_mask > 0.5) # Use >0.5 for boolean input to bwdist

        # Normalize the calculated dGM distances safely (handles division by zero)
        max_val_dGM_dist = np.max(dGM_dist_calc)
        if max_val_dGM_dist > 0:
            dGM_dist_calc_norm = dGM_dist_calc / max_val_dGM_dist
        else:
            dGM_dist_calc_norm = np.zeros_like(dGM_dist_calc, dtype=float) # All zeros if max is 0
        
        # Apply Gaussian smoothing to the normalized dGM distance map
        dGM_dist_final = imgaussfilt3(dGM_dist_calc_norm, 1.0) # Sigma for smoothing is 1.0

        # Calculate the sum of tissue probabilities along the tissue type axis (axis 3)
        # This sum is based on the state *before* dGM-specific updates.
        sum_HR_tissue_prob_before_dGM_update = np.sum(HR_tissue_prob, axis=3)

        # Update the tissue probability for dGM using the calculated dGM distance map
        # This uses the 0-based Python index for dGM.
        HR_tissue_prob[:, :, :, dGM_py_idx] = dGM_dist_final * (2.0 - sum_HR_tissue_prob_before_dGM_update)
        
        # Recalculate the sum of tissue probabilities *after* the dGM update
        sum_HR_tissue_prob_after_dGM_update = np.sum(HR_tissue_prob, axis=3)
        # Update the tissue probability for NAWM.
        # This uses the 0-based Python index for NAWM and the dGM mask based on its 1-based map label.
        HR_tissue_prob[:, :, :, NAWM_py_idx] += dGM_mask_map * (1.0 - sum_HR_tissue_prob_after_dGM_update)

        ### Normalize final tissue probabilities
        # Calculate the sum of all tissue probabilities for each voxel (should ideally be 1 after normalization)
        sum_total_prob = np.sum(HR_tissue_prob, axis=3, keepdims=True) # keepdims for broadcasting
        HR_tissue_prob_normalized = np.zeros_like(HR_tissue_prob) # Initialize normalized array
        # Perform safe division to normalize probabilities, handling cases where sum_total_prob is 0
        np.divide(HR_tissue_prob, sum_total_prob, out=HR_tissue_prob_normalized, where=sum_total_prob != 0)
        HR_tissue_prob = HR_tissue_prob_normalized # Update HR_tissue_prob with normalized values

        ### Calculate final Signal Intensity (SI) map
        # Initialize the High-Resolution Signal Intensity (HR_SI) map
        HR_SI = np.zeros(tuple(map(int, N_true)), dtype=float)
        # Calculate weighted sum of SIs based on tissue probabilities
        # SI array (from Params.py) is assumed to be 0-indexed, corresponding to iRegion_py_idx.
        for iRegion_py_idx in range(NumRegions):
            HR_SI += HR_tissue_prob[:, :, :, iRegion_py_idx] * SI[iRegion_py_idx]

        # Convert PVS map to float for arithmetic operations
        PVS_inner_dist_float = PVS_map.astype(float)
        # Adjust SI for PVS regions:
        # Non-PVS regions keep their calculated SI.
        # PVS regions get the SI value of the last defined tissue type (e.g., CSF or a PVS-specific SI).
        # SI array is 0-indexed, so SI[NumRegions-1] is the SI for the last tissue type.
        HR_SI = HR_SI * (1.0 - PVS_inner_dist_float) + SI[NumRegions - 1] * PVS_inner_dist_float
            
        # Save the resulting High-Resolution Signal Intensity map as a NIFTI file
        nib.save(nib.Nifti1Image(HR_SI.astype(np.float64), affine, header), output_fname)


