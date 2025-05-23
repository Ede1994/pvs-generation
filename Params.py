## Parameter file
#  This configuration file allows setting up imaging and tissue
#  characteristics.
#  
# (c) Jose Bernal 2021

import os
import numpy as np

# Reference masks
use_MIDA_ageing_mask = True  # flag indicating whether to use the MIDA model with pathological regions

# Imaging parameters - dimensions are LR (Phase encoding), AP (Freq Encoding), SI (Slice encoding)
FOV_mm_true = np.array([240, 240, 240])        # Acquired FoV
HRes_mm = np.array([0.5, 0.5, 0.5])            # MSS3 resolution
N_true = np.floor_divide(FOV_mm_true, HRes_mm).astype(int)  # number of points acquired

FOV_mm_acq = np.array([240, 240, 240])         # Acquired FoV
LBC_res_mm = np.array([1, 1, 1])               # LBC resolution
N_acq = np.floor_divide(FOV_mm_acq, LBC_res_mm).astype(int)  # number of points acquired

# Experiment parameters
apply_motion_artefacts = False  # flag indicating whether to induce motion artefacts
apply_noise = True              # flag indicating whether to add noise or not

# Noise extent (standard deviation of background noise)
SDnoise = 0 # 7.1423  # [IQR 5.9272, 8.4299]

# Tissue parameters per class
#  1 - Background
#  2 - Cerebrospinal fluid
#  3 - Normal-appearing white matter
#  4 - White matter hyperintensity
#  6 - Cortical grey matter
#  7 - Deep grey matter
#  8 - Blood vessels
#  9 - PVS
NumRegions = 9
CSF_class_idx = 2
NAWM_class_idx = 3
WMH_class_idx = 4
dGM_class_idx = 7
RSL_class_idx = 5

SI = np.array([0, 1152, 395, 657, 0, 450, 395, 0, 547])

# PVS generation parameters
NRep = 10                # number of repetitions
wstats = np.arange(1, 7) # widths between 1 and 6 mm
lstats = np.arange(1, 21)# lengths between 1 and 20 mm
PVS_vol_space = np.array([21, 21, 21])  # voxels

# PVS characteristics
lengths, widths = np.meshgrid(lstats, wstats, indexing='xy')

# 1st constraint - lengths > widths
mask = (lengths > widths)
lengths = lengths[mask]
widths = widths[mask]

# 2nd constraint - eccentricity > 0.8
eccentricity = np.sqrt((lengths / 2)**2 - (widths / 2)**2) / (lengths / 2)
mask = (eccentricity > 0.8)
lengths = lengths[mask]
widths = widths[mask]