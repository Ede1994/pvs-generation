## Downsampling PVS maps
#  This script downsamples HR maps for simulation
#
# (c) Jose Bernal 2021

import numpy as np
import nibabel as nib
# from scipy.ndimage import distance_transform_edt as bwdist # Removed as it's likely not the desired input for simple downsampling
from tqdm import tqdm
import os # Ensure os is imported for os.path.exists

from Utils.generateLRData import generate_lr_data # Your external function

# Import configuration and parameter settings
from Conf import *
from Params import *

for iCase in tqdm(range(len(lengths)), desc="Cases"):
    for rep in range(NRep):
        # define input and output filenames
        # Ensure iCase and rep are correctly formatted if the pattern expects strings (though .format handles ints)
        input_fname = HR_PVS_map_output_pattern.format(str(iCase), str(rep))
        output_fname = LR_PVS_map_output_pattern.format(
            str(iCase), str(rep),
            str(LBC_res_mm[0]),
            str(LBC_res_mm[1]),
            str(LBC_res_mm[2])
        )

        if os.path.exists(output_fname):
            continue
        
        if not os.path.exists(input_fname):
            print(f"PVS input file {input_fname} not found, skipping Case {iCase} Rep {rep}.")
            continue

        nii = nib.load(input_fname)
        
        # --- CORRECTED INPUT FOR DOWNSAMPLING ---
        # Assume nii.get_fdata() is the HR PVS mask where PVS=1 and background=0.
        # Convert to float for the downsampling function if it expects float.
        hr_pvs_mask_data = nii.get_fdata().astype(np.float32)
        
        # If you absolutely must use the distance transform for a specific reason,
        # then the problem likely lies with SDnoise or the threshold value,
        # or how generate_lr_data handles such input.
        # For standard PVS map downsampling, the binary mask is the correct input.
        # HR_map_original_approach = bwdist(nii.get_fdata() == 0) 

        # Downsample using external function
        # CRITICAL: Try with SDnoise = 0 first to isolate its effect.
        # If SDnoise must be non-zero, you'll need to understand its impact on zero-value regions.
        LR_map = generate_lr_data(
            hr_pvs_mask_data, # Use the binary PVS mask
            FOV_mm_true,
            N_true,
            SDnoise,      # TRY SETTING THIS TO 0 for debugging
            FOV_mm_acq,
            N_acq,
            False,
            False
        )

        # --- HEADER AND AFFINE UPDATE ---
        new_header = nii.header.copy()
        new_header.set_data_dtype(np.uint8) # Final output is a binary mask

        # Set the voxel dimensions (zooms) for the low-resolution image
        new_header.set_zooms(tuple(LBC_res_mm))

        # Set the matrix dimensions for the low-resolution image
        if len(N_acq) == 3:
            # new_header['dim'] structure: [ndim, dim1, dim2, dim3, dim4, ...]
            # We are setting spatial dimensions (dim1, dim2, dim3)
            current_dims = list(new_header['dim']) # Get current dims
            current_dims[0] = 3 # Number of spatial dimensions
            current_dims[1:4] = list(N_acq) # Set new spatial dimensions
            new_header['dim'] = np.array(current_dims, dtype=new_header['dim'].dtype)
        else:
            print(f"Warning: N_acq length {len(N_acq)} is not 3. Header 'dim' not optimally updated.")

        # Construct a new affine matrix for the low-resolution image
        # This is crucial for correct orientation and position in viewers.
        # Assumes new data is axis-aligned. If original had rotations/shears, this is a simplification.
        lr_affine = np.diag(list(LBC_res_mm) + [1.0])
        
        # Set the origin: The following assumes the FoV is centered around (0,0,0) in world space.
        # The affine's translation part (last column) maps voxel (0,0,0) to a world coordinate.
        # If voxel (0,0,0) is the corner of the FoV, and FoV is centered at world (0,0,0):
        # origin_offset = - (np.array(FOV_mm_acq) / 2.0)
        # lr_affine[0:3, 3] = origin_offset
        
        # A common approach: try to adapt the original affine's origin and orientation.
        # This is complex if resolutions and FoVs change non-proportionally.
        # For simplicity, if your generate_lr_data effectively resamples to a grid whose
        # first voxel's center should map to a specific known coordinate (e.g., from original affine):
        # lr_affine[0:3, 3] = nii.affine[0:3, 3] # This keeps the same starting corner coordinate as HR
                                                # This might be correct if generate_lr_data preserves starting corner.

        # If generate_lr_data effectively centers the FOV_mm_acq in the same space as FOV_mm_true,
        # then a more robust way is needed. For now, a simple diagonal affine with LBC_res_mm.
        # If the original affine had rotations, they are lost here.
        # A truly robust solution would involve understanding the grid `generate_lr_data` outputs to.
        # If generate_lr_data is like a k-space operation, the output FoV (FOV_mm_acq) is key.
        # Let's assume the affine from `nii.affine`'s orientation part can be preserved and only scaling applied.
        new_affine_matrix = nii.affine.copy()
        for i_dim in range(3): # Scale diagonal elements by ratio of old res to new res
            # Old voxel size in this dim (approx from affine if axis aligned)
            old_vox_size = np.linalg.norm(nii.affine[:3, i_dim])
            # New voxel size
            new_vox_size = LBC_res_mm[i_dim]
            # Scaling factor for this column of the affine
            scale = new_vox_size / old_vox_size
            new_affine_matrix[:3, i_dim] *= scale
        # The translation component (origin) might also need adjustment if the FoV center shifts.
        # For now, we'll use this scaled affine. It preserves orientation.

        # --- THRESHOLDING ---
        # If LR_map now contains partial volume fractions (0.0 to 1.0),
        # 0.5 is a common threshold.
        # If after using hr_pvs_mask_data and SDnoise=0, the output is still all white,
        # then LR_map values are all > 0.5, indicating an issue with generate_lr_data
        # or the interpretation of its output for binary inputs.
        threshold_value = 0.5 
        # print(f"LR_map min: {np.min(LR_map)}, max: {np.max(LR_map)}, mean: {np.mean(LR_map)}") # For debugging

        # Save downsampled LR PVS map
        nib.save(
            nib.Nifti1Image((LR_map).astype(np.uint8), new_affine_matrix, new_header),
            output_fname
        )
