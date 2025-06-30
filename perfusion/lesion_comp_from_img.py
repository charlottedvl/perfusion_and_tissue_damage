"""
Compute lesion volume proxies from brain perfusion images (.nii.gz) by:
- Thresholding absolute cerebral blood flow (aCBF)
- Optionally thresholding relative CBF (rCBF) against a healthy reference

The calculated volumes are saved to a YAML file.

@author: Charlotte Devillé
"""

# Import python3 modules
import argparse
import numpy as np
import nibabel as nib
import os
import yaml


def create_lesion_comp_parser():
    """
    Creates and configures an argparse.ArgumentParser for the lesion volume computation script.

    This parser defines command-line arguments required for specifying input
    perfusion images, output folder, and threshold values for lesion computation.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    # Read input
    parser = argparse.ArgumentParser(description="compute lesion proxies from perfusion images (*.nii.gz)")
    parser.add_argument("--healthy_file", help="path to image file of the healthy state", type=str,
                        default='./results/p0000/perfusion_healthy/perfusion.nii.gz')
    parser.add_argument("--occluded_file", help="path to image file of the occluded state", type=str,
                        default='./results/p0000/perfusion_RMCAo/perfusion.nii.gz')
    parser.add_argument("--res_fldr", help="path to folder where results will be saved", type=str, default=None)
    parser.add_argument("--background_value", help="value used for background voxels", type=int, default=-1024)
    parser.add_argument("--rCBF_thrshld", help="rCBF<rCBF_thrshld -> lesion ", type=float, default=0.3)
    parser.add_argument("--aCBF_thrshld", help="CBF<aCBF_thrshld -> lesion [ml/min/100g] ", type=float, default=5)
    parser.add_argument('--save_figure', action='store_true',
                        help="save figure showing image along midline slices", default=False)
    return parser


def load_occluded_image(occluded_file):
    """
    Loads the occluded perfusion image and extracts its data, header, and voxel volume.

    Args:
        occluded_file (str): The full path to the NIfTI file of the occluded brain state.

    Returns:
        tuple: A tuple containing:
            - occluded_image_data (numpy.ndarray): The 3D NumPy array of the image voxel data.
            - voxel_vol (float): The volume of a single voxel in milliliters (ml).
    """
    occluded_image = nib.load(occluded_file)
    occluded_image_data = occluded_image.get_fdata()
    occluded_header = occluded_image.header
    voxel_vol = occluded_header.get("srow_x")[0] * occluded_header.get("srow_y")[1] * occluded_header.get("srow_z")[
        2] / 1000  # [ml]
    return occluded_image_data, voxel_vol


def compute_lesion_volume_aCBF(occluded_image, threshold, background_value, voxel_vol):
    """
    Computes the lesion volume based on an absolute Cerebral Blood Flow (aCBF) threshold.

    Voxels are considered part of the lesion if their aCBF value is below the
    given threshold, and they are not part of the background.

    Args:
       occluded_image (numpy.ndarray): The 3D NumPy array of the occluded CBF image data.
       threshold (float): The aCBF threshold value (e.g., in ml/min/100g).
       background_value (int): The value representing background voxels in the image.
       voxel_vol (float): The volume of a single voxel in milliliters (ml).

    Returns:
       float: The computed lesion volume in milliliters (ml).
    """
    lesion_mask = np.logical_and(occluded_image < threshold, occluded_image != background_value)
    return np.sum(lesion_mask) * voxel_vol


def compute_lesion_volume_rCBF(occluded_image, healthy_image, threshold, voxel_vol):
    """
    Computes the lesion volume based on a relative Cerebral Blood Flow (rCBF) threshold.

    rCBF is calculated as occluded_image / healthy_image. Voxels are considered
    part of the lesion if their rCBF value is below the given threshold.

    Args:
        occluded_image (numpy.ndarray): The 3D NumPy array of the occluded CBF image data.
        healthy_image (numpy.ndarray): The 3D NumPy array of the healthy reference CBF image data.
        threshold (float): The rCBF threshold value (e.g., 0.3 for 30% of healthy CBF).
        voxel_vol (float): The volume of a single voxel in milliliters (ml).

    Returns:
        float: The computed lesion volume in milliliters (ml).
    """
    rCBF = occluded_image / healthy_image
    return np.sum(rCBF < threshold) * voxel_vol


def save_lesion_volumes(results_file, healthy_image_available, aCBF_vol, aCBF_thr, rCBF_vol=None, rCBF_thr=None):
    """
    Saves the computed lesion volumes to a YAML file.

    The file is opened in append mode ('a'), allowing multiple runs to add data.
    Both aCBF and rCBF-based volumes (if available) are saved.

    Args:
        results_file (str): The full path to the YAML file where results will be saved.
        healthy_image_available (bool): True if a healthy reference image was loaded and used
                                        for rCBF computation, False otherwise.
        aCBF_vol (float): The lesion volume computed using the aCBF threshold.
        aCBF_thr (float): The aCBF threshold value used for computation.
        rCBF_vol (float, optional): The lesion volume computed using the rCBF threshold.
                                    Defaults to None.
        rCBF_thr (float, optional): The rCBF threshold value used for computation.
                                    Defaults to None.

    Returns:
        None
    """
    with open(results_file, 'a') as outfile:
        yaml.safe_dump(
            {'img_core-volume_' + '{:02d}'.format(int(aCBF_thr)) + '_aCBF_mL': float(aCBF_vol)},
            outfile
        )
        # Save rCBF-based lesion volume as well
        if healthy_image_available:
            yaml.safe_dump(
                {'img_core-volume_' + '{:02d}'.format(int(100 * rCBF_thr)) + '%_rCBF_mL': float(
                    rCBF_vol)},
                outfile
            )


def main():
    # Parse input arguments
    parser = create_lesion_comp_parser()

    # Get input
    args = parser.parse_args()
    aCBF_threshold = args.aCBF_thrshld
    rCBF_threshold = args.rCBF_thrshld
    occluded_file = args.occluded_file
    healthy_file = args.healthy_file

    result_folder = args.res_fldr
    if not os.path.exists(str(result_folder)):
        print('Path to result folder is defined based on the location of the occluded image file')
        path_elements = occluded_file.split('/')[:-1]
        result_folder = os.path.join(*path_elements)

    # Load occluded image
    occluded_img, voxel_volume = load_occluded_image(occluded_file)

    # Compute lesion volume from aCBF threshold
    volume_lesion_aCBF = compute_lesion_volume_aCBF(occluded_img, aCBF_threshold, parser.parse_args().background_value,
                                                    voxel_volume)
    volume_lesion_rCBF = None

    try:
        # Try loading healthy file
        healthy_img = nib.load(healthy_file)
        healthy_img = healthy_img.get_fdata()

        if occluded_img.shape != healthy_img.shape:
            print("The shapes of occluded and healthy images do not match!")
            raise ValueError("Shape mismatch")

        # Compute rCBF based volume
        healthy_img_available = True
        volume_lesion_rCBF = compute_lesion_volume_rCBF(occluded_img, healthy_img, rCBF_threshold, voxel_volume)

    except (FileNotFoundError, ValueError) as e:
        healthy_img_available = False
        print('Reference image of healthy state is not available: ', e)

    # Always save the aCBF-based lesion volume, optionally save the rCBF-based lesion volume
    result_file = os.path.join(result_folder, "perfusion_outcome.yml")
    save_lesion_volumes(result_file, healthy_img_available, volume_lesion_aCBF, aCBF_threshold, volume_lesion_rCBF,
                        rCBF_threshold)


if __name__ == "__main__":
    main()
