#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script downsamples high-resolution (HR) signal intensity (SI) maps for simulation.
Version:    0.0.1
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt as bwdist
from tqdm import tqdm

from utils.applyRotation import apply_rotation
from utils.generateLRData import generate_lr_data

# Import configuration and parameter settings
from config.conf import *
from config.params import *


# ***************************************************************************
# * Main
# ***************************************************************************
if __name__ == "__main__":
    # Load HR SI file info for metadata
    input_fname = HR_SI_output_pattern.format(0, 0)
    nii_info = nib.load(input_fname)
    tissue_map = nib.load(MIDA_tissue_map_fname).get_fdata()

    # Generate random transformations
    rot_angles = np.random.uniform(-5, -1, size=(NRep, 1))
    rot_dimensions = np.random.randint(0, 2, size=(NRep, 3))

    LBC_res_mm_np = np.array(LBC_res_mm)
    N_acq_np = np.array(N_acq)
    FOV_mm_true_np = np.array(FOV_mm_true)
    N_true_np = np.array(N_true)
    FOV_mm_acq_np = N_acq_np * LBC_res_mm_np # Calculate LR FoV

    # Create LR T2-w like images with PVS
    for iCase in tqdm(range(len(lengths)), desc="Cases"):
        for rep in range(NRep):
            input_fname = HR_SI_output_pattern.format(iCase, rep)
            output_fname = LR_SI_output_pattern.format(
                iCase, rep,
                LBC_res_mm[0], LBC_res_mm[1], LBC_res_mm[2]
            )

            nii = nib.load(input_fname)
            HR_SI = nii.get_fdata().astype(np.float64)

            if apply_motion_artefacts:
                theta_prev = rot_angles[rep][0]
                theta_post = -theta_prev
                HR_SI = np.stack([
                    apply_rotation(HR_SI, theta_prev, rot_dimensions[rep]),
                    HR_SI,
                    apply_rotation(HR_SI, theta_post, rot_dimensions[rep])
                ], axis=-1)

            # Generate Low Resolution data
            apply_noise_flag_val = 1 if apply_noise else 0
            apply_motion_flag_val = 1 if apply_motion_artefacts else 0

            LR_SI = generate_lr_data(
                HR_SI,
                FOV_mm_true_np.tolist(),       # Pass as list if function expects list
                N_true_np.tolist(),
                SDnoise,
                FOV_mm_acq_np.tolist(),
                N_acq_np.tolist(),
                apply_noise_flag_val,        # Pass the flag value
                apply_motion_flag_val      # Pass the flag value
                )

            new_header = nii.header.copy()
            new_header.set_data_dtype(np.uint16)
            new_header.set_zooms(LBC_res_mm)
            new_header['dim'][1:4] = N_acq

            nib.save(nib.Nifti1Image(LR_SI.astype(np.uint16), nii.affine, new_header), output_fname)

    # Generate LR brain and ROI masks
    info = nii.header.copy()
    info.set_data_dtype(np.uint16)
    info.set_zooms(LBC_res_mm)
    info['dim'][1:4] = N_acq

    # Brain mask: combine tissues 3,4,5,6,7
    ROI = np.isin(tissue_map, [3, 4, 5, 6, 7])
    ROI_dist = bwdist(~ROI)
    LR_brain_mask = generate_lr_data(
        ROI_dist,
        FOV_mm_true, N_true, SDnoise,
        FOV_mm_acq, N_acq,
        0, 0
    ) >= 6

    # ROI mask: tissues 3,4,7
    ROI = np.isin(tissue_map, [3, 4, 7])
    LR_ROI_mask = generate_lr_data(
        bwdist(~ROI),
        FOV_mm_true, N_true, SDnoise,
        FOV_mm_acq, N_acq,
        0, 0
    ) >= 1

    # Save LR masks
    LR_ROI_mask_output_fname = LR_ROI_mask_fname % (
        LBC_res_mm[0], LBC_res_mm[1], LBC_res_mm[2]
    )

    nib.save(nib.Nifti1Image(LR_brain_mask.astype(np.uint16), nii.affine, info), LR_brain_mask_fname)
    nib.save(nib.Nifti1Image((LR_ROI_mask * LR_brain_mask).astype(np.uint16), nii.affine, info), LR_ROI_mask_output_fname)