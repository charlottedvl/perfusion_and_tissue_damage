"""
Compute lesion volume proxies from brain perfusion images (.nii.gz) by:
- Thresholding absolute cerebral blood flow (aCBF)
- Optionally thresholding relative CBF (rCBF) against a healthy reference

@author: Charlotte Devillé
"""

# Import python3 modules
import argparse
import numpy as np
import nibabel as nib
import os
import yaml

# Read input
parser = argparse.ArgumentParser(description="compute lesion proxies from perfusion images (*.nii.gz)")
parser.add_argument("--healthy_file", help="path to image file of the healthy state",
                    type=str, default='./results/p0000/perfusion_healthy/perfusion.nii.gz')
parser.add_argument("--occluded_file", help="path to image file of the occluded state",
                    type=str, default='./results/p0000/perfusion_RMCAo/perfusion.nii.gz')
parser.add_argument("--res_fldr", help="path to folder where results will be saved",
                    type=str, default=None)
parser.add_argument("--background_value", help="value used for background voxels",
                    type=int, default=-1024)
parser.add_argument("--rCBF_thrshld", help="rCBF<rCBF_thrshld -> lesion ",
                    type=float, default=0.3)
parser.add_argument("--aCBF_thrshld", help="CBF<aCBF_thrshld -> lesion [ml/min/100g] ",
                    type=float, default=5)
parser.add_argument('--save_figure', action='store_true',
                    help="save figure showing image along midline slices")
parser.set_defaults(save_figure=False)

# Parse input arguments
args = parser.parse_args()
aCBF_threshold = args.aCBF_thrshld
rCBF_threshold = args.rCBF_thrshld
occluded_file = args.occluded_file
healthy_file = args.healthy_file
result_folder = args.res_fldr

if not os.path.exists( str(result_folder) ):
    print('Path to result folder is defined based on the location of the occluded image file')
    path_elements = occluded_file.split('/')[:-1]
    result_folder = os.path.join(*path_elements)

# Load occluded image
occluded_img = nib.load(occluded_file)
occluded_header = occluded_img.header
voxel_volume = occluded_header.get("srow_x")[0] * occluded_header.get("srow_y")[1] * occluded_header.get("srow_z")[2] / 1000 # [ml]
occluded_img = occluded_img.get_fdata()

# Compute lesion volume from aCBF threshold
lesion_mask = np.logical_and(occluded_img < aCBF_threshold, occluded_img != parser.parse_args().background_value )
volume_lesion_aCBF = np.sum(lesion_mask) * voxel_volume

try:
    # Try loading healthy file
    healthy_img = nib.load(healthy_file)
    header_h = healthy_img.header
    healthy_img = healthy_img.get_fdata()
    
    if occluded_img.shape != healthy_img.shape:
        print("The shapes of occluded and healthy images do not match!")
        raise ValueError("Shape mismatch")

    # Compute rCBF based volume
    healthy_img_available = True
    rCBF = occluded_img/healthy_img
    volume_lesion_rCBF = np.sum(rCBF<rCBF_threshold) * voxel_volume

except (FileNotFoundError, ValueError) as e:
    healthy_img_available = False
    print('Reference image of healthy state is not available: ', e)

# Always save the aCBF-based lesion volume
result_file = os.path.join(result_folder,"perfusion_outcome.yml")
with open(result_file, 'a') as outfile:
    yaml.safe_dump(
        {'img_core-volume_' + '{:02d}'.format(int(aCBF_threshold)) + '_aCBF_mL': float(volume_lesion_aCBF)},
        outfile
    )
    # Save rCBF-based lesion volume as well
    if healthy_img_available:
        yaml.safe_dump(
            {'img_core-volume_' + '{:02d}'.format(int(100 * rCBF_threshold)) + '%_rCBF_mL': float(volume_lesion_rCBF)},
            outfile
        )
