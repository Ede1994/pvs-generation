## Create HR PVS maps
#  This script creates HR PVS maps for simulation
#
# (c) Jose Bernal 2021
# generate_hr_pvs_maps.py

import numpy as np
import nibabel as nib
from scipy import ndimage
from Conf import (
    MIDA_tissue_map_fname,
    HR_ROI_mask_fname,
    HR_PVS_map_output_pattern
)
from Params import lengths, widths, PVS_vol_space, NRep, N_true
from Utils.blockproc3 import blockproc3
from Utils.create_PVS_creator import create_PVS_creator
from Utils.summariseNeighbourhood import summarise_neighbourhood

def main():
    # Load tissue map
    tissue_nifti = nib.load(MIDA_tissue_map_fname)
    tissue_map = tissue_nifti.get_fdata().astype(np.uint8)

    # Construct ROI: NAWM (3), WMH (4), deep GM (7)
    roi = (tissue_map == 3) | (tissue_map == 4) | (tissue_map == 7)
    # Distance transform of background
    dist = ndimage.distance_transform_edt(~roi)
    roi = dist >= 4

    # Save ROI mask
    roi_img = nib.Nifti1Image(roi.astype(np.uint8), tissue_nifti.affine, tissue_nifti.header)
    nib.save(roi_img, HR_ROI_mask_fname + '.nii.gz')

    # PVS parameters
    NVox    = PVS_vol_space

    # For each PVS size (case) and repetition
    for iCase, (length, width) in enumerate(zip(lengths, widths), start=1):
        for rep in range(1, NRep + 1):
            pvs_creator = create_PVS_creator(length, width, NVox)

            # Random feasible locations within ROI
            feasible = roi & (np.random.rand(*roi.shape) >= 0.5)
            # Enforce one PVS centre per block
            blk = np.ceil(NVox * 0.7).astype(int)
            feasible = blockproc3(
                feasible.astype(np.uint8),
                blksz=blk,
                fun=summarise_neighbourhood,
                border=[0, 0, 0],
                numworkers=1
            ).astype(bool)

            # Get voxel coordinates
            xs, ys, zs = np.where(feasible)
            mask = np.zeros_like(roi, dtype=bool)
            half = NVox // 2
            fov_vox = N_true  # dimensions of the volume in voxels

            for idx in range(xs.size):
                x, y, z = xs[idx], ys[idx], zs[idx]

                # Bounding box for this PVS
                xmin = max(x - half[0], 0)
                xmax = min(x + half[0], roi.shape[0] - 1)
                ymin = max(y - half[1], 0)
                ymax = min(y + half[1], roi.shape[1] - 1)
                zmin = max(z - half[2], 0)
                zmax = min(z + half[2], roi.shape[2] - 1)

                # Compute rotation matrix to align [0,0,1] with vector towards far corner
                A = np.array([0, 0, 1], dtype=float)
                B = fov_vox - np.array([x, y, z], dtype=float)
                B = B / np.linalg.norm(B)

                def GG(A, B):
                    cr = np.cross(A, B)
                    d = np.dot(A, B)
                    return np.array([
                        [ d,            -np.linalg.norm(cr), 0],
                        [ np.linalg.norm(cr),  d,            0],
                        [ 0,                   0,             1]
                    ])

                def FFi(A, B):
                    v = B - np.dot(A, B)*A
                    v = v / np.linalg.norm(v)
                    return np.stack([A, v, np.cross(B, A)], axis=1)

                def UU(Fi, G):
                    return Fi.dot(G).dot(np.linalg.inv(Fi))

                U = UU(FFi(B, A), GG(B, A))

                # full PVS block (shape NVox)
                pvs_full = pvs_creator(U)  # shape (nx, ny, nz)
                total_voxels = pvs_full.sum()

                # define sub‐volume bounds in image
                xmin = max(x - half[0], 0)
                xmax = min(x + half[0], roi.shape[0] - 1)
                ymin = max(y - half[1], 0)
                ymax = min(y + half[1], roi.shape[1] - 1)
                zmin = max(z - half[2], 0)
                zmax = min(z + half[2], roi.shape[2] - 1)

                # define how much we clipped on each side
                cx0 = xmin - (x - half[0])   # positive if we clipped lower bound
                cx1 = (x + half[0]) - xmax   # positive if we clipped upper bound
                cy0 = ymin - (y - half[1])
                cy1 = (y + half[1]) - ymax
                cz0 = zmin - (z - half[2])
                cz1 = (z + half[2]) - zmax

                # crop the full PVS block to match the sub‐volume size
                pvs = pvs_full[
                    cx0 : pvs_full.shape[0] - cx1,
                    cy0 : pvs_full.shape[1] - cy1,
                    cz0 : pvs_full.shape[2] - cz1
                ]

                # extract the corresponding sub‐regions of mask and roi
                sub_mask = mask[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
                sub_roi  = roi[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]

                # Check no overlap and fully within ROI
                if (pvs & sub_mask).sum() == 0 and (pvs & sub_roi).sum() == total_voxels:
                    mask[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] |= pvs

            # Zero outside ROI
            mask[~roi] = False

            # Keep only connected components large enough
            structure = np.ones((3, 3, 3), dtype=int)
            labels, num = ndimage.label(mask, structure=structure)
            counts = np.bincount(labels.ravel())
            # Threshold: volume >= length * (width/2)^2 * pi
            thresh = length * (width / 2)**2 * np.pi
            valid_labels = np.nonzero(counts >= thresh)[0]
            valid_labels = valid_labels[valid_labels != 0]  # exclude background
            mask_corrected = np.isin(labels, valid_labels)

            # Save PVS mask
            fname = HR_PVS_map_output_pattern.format(iCase, rep) + '.nii.gz'
            out_img = nib.Nifti1Image(mask_corrected.astype(np.uint8), tissue_nifti.affine, tissue_nifti.header)
            nib.save(out_img, fname)

if __name__ == "__main__":
    main()

