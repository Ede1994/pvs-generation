#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script processes high-resolution (HR) PVS maps and generates low-resolution (LR) PVS maps.
Version:    0.0.1
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt as bwdist
from tqdm import tqdm

from utils.generateLRData import generate_lr_data

# Import configuration and parameter settings
from config.conf import *
from config.params import *


# ***************************************************************************
# * Main
# ***************************************************************************
if __name__ == "__main__":
    # Ensure these are numpy arrays if they are used in calculations expecting them
    LBC_res_mm_np = np.array(LBC_res_mm)
    N_acq_np = np.array(N_acq)
    FOV_mm_true_np = np.array(FOV_mm_true)
    N_true_np = np.array(N_true)
    FOV_mm_acq_np = N_acq_np * LBC_res_mm_np # Calculate LR FoV

    for iCase in tqdm(range(len(lengths)), desc="Cases"):
        for rep in range(NRep):
            # Define input and output filenames
            input_fname = HR_PVS_map_output_pattern.format(str(iCase), str(rep))
            output_fname = LR_PVS_map_output_pattern.format(
                str(iCase), str(rep),
                str(LBC_res_mm[0]), # Original LBC_res_mm for filename
                str(LBC_res_mm[1]),
                str(LBC_res_mm[2])
            )

            if os.path.exists(output_fname):
                continue
            
            if not os.path.exists(input_fname):
                print(f"Input file {input_fname} not found. Skipping Case {iCase} Rep {rep}.")
                continue

            nii_hr = nib.load(input_fname)
            hr_pvs_binary_map = nii_hr.get_fdata() # This is 0 for background, 1 for PVS

            # PVS regions will have positive distance values, background will be 0.
            hr_map_dist_transform = bwdist(hr_pvs_binary_map == 0).astype(np.float32)

            # Generate Low Resolution data
            apply_noise_flag_val = 1 if apply_noise else 0 # Example if apply_noise is bool
            apply_motion_flag_val = 1 if apply_motion_artefacts else 0 # Example

            lr_map_processed = generate_lr_data(
                hr_map_dist_transform, # Input is the distance transform
                FOV_mm_true_np.tolist(),       # Pass as list if function expects list
                N_true_np.tolist(),
                SDnoise,
                FOV_mm_acq_np.tolist(),
                N_acq_np.tolist(),
                apply_noise_flag_val,        # Pass the flag value
                apply_motion_flag_val      # Pass the flag value
            )
            
            # Create new header for the LR image
            lr_header = nii_hr.header.copy()
            lr_header.set_data_dtype(np.uint8) # Final output is binary 0/1
            
            # Update spatial dimensions (matrix size)
            current_dims = list(lr_header['dim'])
            current_dims[0] = 3 # Number of spatial dimensions
            current_dims[1:4] = list(N_acq_np) # Set new spatial dimensions
            current_dims[4:] = [1] * (len(current_dims) - 4) # Set trailing dims to 1
            lr_header['dim'] = np.array(current_dims, dtype=lr_header['dim'].dtype)
            
            # Update voxel dimensions (zooms)
            lr_header.set_zooms(tuple(LBC_res_mm_np))

            # Update affine matrix
            # Simple scaling, assumes axis alignment and origin preservation (approximate)
            lr_affine = nii_hr.affine.copy()
            for i_dim in range(3):
                old_vox_size = np.linalg.norm(nii_hr.affine[:3, i_dim])
                new_vox_size = LBC_res_mm_np[i_dim]
                if old_vox_size > 1e-6 : # Avoid division by zero
                    scale = new_vox_size / old_vox_size
                    lr_affine[:3, i_dim] *= scale
                else: # Should not happen for valid affine
                    lr_affine[:3, i_dim] = 0 # Or handle error
                    lr_affine[i_dim, i_dim] = new_vox_size


            # Threshold and save LR PVS map
            lr_pvs_binary_output = (lr_map_processed < 0.5).astype(np.uint8)

            ### Save the LR PVS map
            nib.save(
                nib.Nifti1Image(lr_pvs_binary_output, lr_affine, lr_header),
                output_fname
            )
