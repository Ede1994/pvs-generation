#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This script generates segmentation maps for the MIDA model.
1  - Background
2  - Cerebrospinal fluid
3  - Normal-appearing white matter
4  - White matter hyperintensity
5  - Recent stroke lesion
6  - Cortical grey matter
7  - Deep grey matter
8  - Blood vessels

Version:    0.0.1
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import numpy as np
import nibabel as nib
import scipy.io as sio

from config.conf import (
    MIDA_mapping_fname,
    MIDA_map_fname,
    MIDA_WMH_fname,
    MIDA_RSL_fname,
    MIDA_tissue_map_fname,
    MIDA_tissue_map_ageing_fname,
    MIDA_T2_fname,
    MIDA_brain_mask_fname
)
from config.params import N_true, HRes_mm, SI as SI_array


# ***************************************************************************
# * Main
# ***************************************************************************
def main():
    # Load mapping between original 116 labels and reduced labels
    mat = sio.loadmat(MIDA_mapping_fname)
    mapping = mat['mapping']   # shape (116, 2)

    # Load MIDA segmentation and masks
    orig_nifti = nib.load(MIDA_map_fname)
    HR_orig_segmap = orig_nifti.get_fdata().astype(int)
    WMH_segmap      = nib.load(MIDA_WMH_fname).get_fdata().astype(int)
    RSL_segmap      = nib.load(MIDA_RSL_fname).get_fdata().astype(int)

    # Map tissues to reduced classes
    HR_tissue_map_clean = np.zeros_like(HR_orig_segmap, dtype=np.uint16)
    for c in range(1, mapping.shape[0] + 1):
        new_label = int(mapping[c - 1, 1])
        HR_tissue_map_clean[HR_orig_segmap == c] = new_label

    # Add pathological tissue label (WMH → class 4, except CSF)
    HR_tissue_map_ageing = HR_tissue_map_clean.copy()
    mask_wmh = (WMH_segmap == 1) & (HR_tissue_map_ageing != 2)
    HR_tissue_map_ageing[mask_wmh] = 4

    # Crop along second dimension (MATLAB cols 141:end → Python 140:)
    def crop_pad_transform(arr):
        # crop
        a = arr[:, 140:, :]
        # pad: ((before, after) for each dim)
        a = np.pad(a,
            pad_width=((0, 0), (70, 70), (65, 65)),
            mode='constant',
            constant_values=1
        )
        # permute dims [3,1,2] → (z, x, y)
        a = np.transpose(a, (2, 0, 1))
        # flip along axis=1
        return np.flip(a, axis=1)

    HR_tissue_map_clean = crop_pad_transform(HR_tissue_map_clean)
    HR_tissue_map_ageing = crop_pad_transform(HR_tissue_map_ageing)

    # Prepare header and affine for output NIfTIs
    affine = np.eye(4)
    hdr = orig_nifti.header.copy()
    hdr.set_data_dtype(np.uint16)
    hdr.set_qform(affine, code=1)
    hdr.set_sform(affine, code=1)
    hdr['pixdim'][1:4] = HRes_mm

    # Save cleaned and ageing tissue maps
    nib.save(
        nib.Nifti1Image(HR_tissue_map_clean, affine, header=hdr),
        MIDA_tissue_map_fname
    )
    nib.save(
        nib.Nifti1Image(HR_tissue_map_ageing, affine, header=hdr),
        MIDA_tissue_map_ageing_fname
    )

    # Generate and save high-resolution T2-like image and brain mask
    HR_SI = SI_array[HR_tissue_map_clean - 1]
    nib.save(
        nib.Nifti1Image(HR_SI.astype(np.uint16), affine, header=hdr),
        MIDA_T2_fname
    )
    brain_mask = (HR_SI > 0).astype(np.uint16)
    nib.save(
        nib.Nifti1Image(brain_mask, affine, header=hdr),
        MIDA_brain_mask_fname
    )

if __name__ == "__main__":
    main()
