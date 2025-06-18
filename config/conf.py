#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This configuration file allows adding all relevant folders to the project.
Date   :    18.06.2025
'''

# ***************************************************************************
# * Import
# ***************************************************************************
import os
import sys

# ***************************************************************************
# * Configuration
# ***************************************************************************
# limit number of computational threads (e.g., for MKL, OpenMP)
os.environ.setdefault("OMP_NUM_THREADS", "15")
os.environ.setdefault("MKL_NUM_THREADS", "15")

# add project folders to PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), 'Artefacts'))
sys.path.append(os.path.join(os.getcwd(), 'Utils'))

# output folder
output_folder = 'output'

# MIDA model configuration parameters
MIDA_mapping_fname = os.path.join('input', 'mapping.mat')
MIDA_mapping_orientation_maps_fname = os.path.join('input', 'mapping_orientation_maps.mat')
MIDA_map_fname = os.path.join('input', 'MIDA_v1.0', 'MIDA_v1_voxels', 'MIDA_v1.nii')
MIDA_WMH_fname = os.path.join('input', 'Masks', 'WMH_mask.nii.gz')
MIDA_RSL_fname = os.path.join('input', 'Masks', 'RSL_mask.nii.gz')
MIDA_tissue_map_fname = os.path.join('input', 'HR_tissue_map.nii.gz')
MIDA_tissue_map_ageing_fname = os.path.join('input', 'HR_tissue_map_ageing.nii.gz')
MIDA_T2_fname = os.path.join('input', 'HR_T2w.nii.gz')
MIDA_brain_mask_fname = os.path.join('input', 'HR_brainmask.nii.gz')
MIDA_Vent_fname = os.path.join('input', 'HR_ventricles.nii.gz')

# PVS location map configuration parameters
PVS_reference_T2_fname = os.path.join('input', 'PVSMaps', 'ref_case.nii.gz')
PVS_map_output_pattern = os.path.join('input', 'PVSMaps', 'PVSmap_{}_to_MIDA.nii.gz')

# Output configuration parameters
HR_PVS_map_output_pattern   = os.path.join('output', 'PVS_mask_per_case', 'HR_PVS_mask_Case_{}_Rep_{}.nii.gz')
LR_PVS_map_output_pattern   = os.path.join('output', 'PVS_mask_per_case', 'LR_PVS_mask_Case_{}_Rep_{}_Res_({}_{}_{}).nii.gz')
HR_SI_output_pattern        = os.path.join('output', 'SI', 'HR_SI_Case_{}_Rep_{}.nii.gz')
LR_SI_output_pattern        = os.path.join('output', 'SI', 'LR_SI_Case_{}_Rep_{}_Res_({}_{}_{} ).nii.gz')
LR_likelihood_pattern       = os.path.join('output', 'Likelihood', 'LR_likelihood_{}_Case_{}_Rep_{}_Res_({}_{}_{}).nii.gz')
LR_likelihood_thresholded   = os.path.join('output', 'Likelihood', 'LR_likelihood_{}_thresholded_{}_Case_{}_Rep_{}.mat')
HR_ROI_mask_fname           = os.path.join('output', 'HR_ROI_mask.nii.gz')
LR_ROI_mask_fname           = os.path.join('output', 'LR_ROI_mask_Res_({}_{}_{}).nii.gz')
HR_brain_mask_fname         = os.path.join('output', 'HR_brainmask.nii.gz')
LR_brain_mask_fname         = os.path.join('output', 'LR_brainmask_Res_({}_{}_{}).nii.gz')
performance_fname           = os.path.join('output', 'performance_outputs', 'performance_Filter_{}_Res_({}_{}_{} ).mat')
toPlot_fname                = os.path.join('output', 'performance_outputs', 'toPlot_Filter_{}_Res_({}_{}_{} ).mat')