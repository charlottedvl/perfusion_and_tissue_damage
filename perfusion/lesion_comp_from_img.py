# installed python3 modules
import sys
import argparse
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import yaml

np.set_printoptions(linewidth=200)


# %% READ INPUT
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
aCBF_thrshld = parser.parse_args().aCBF_thrshld
rCBF_thrshld = parser.parse_args().rCBF_thrshld


file_o = parser.parse_args().occluded_file
file_h = parser.parse_args().healthy_file

res_fldr = parser.parse_args().res_fldr
if not os.path.exists( str(res_fldr) ):
    print('Path to result folder is defined based on the location of the occluded image file')
    mypath_list = file_o.split('/')[:-1]
    res_fldr = os.path.join(*mypath_list)

img_o = nib.load(file_o)
header_o = img_o.header
voxel_volume = header_o.get("srow_x")[0] * header_o.get("srow_y")[1] * header_o.get("srow_z")[2] / 1000 # [ml]
img_o = img_o.get_fdata()

mymask = np.logical_and(img_o<aCBF_thrshld, img_o!=parser.parse_args().background_value )
V_CBF_ST_thrshld = np.sum(mymask)*voxel_volume

try:
    img_h = nib.load(file_h)
    header_h = img_h.header
    img_h = img_h.get_fdata()
    
    if not img_o.shape == img_h.shape:
        print("The shapes of occluded and healthy images do not match!")
        sys.exit
    
    img_h_avail = True
    rCBF = img_o/img_h
    V_rCBF_ST_thrshld = np.sum(rCBF<rCBF_thrshld)*voxel_volume
    
except:
    img_h_avail = False
    print('Reference image of healthy state is not available')

if img_h_avail:
    my_res_file = os.path.join(res_fldr,"perfusion_outcome.yml")
    with open(my_res_file, 'a') as outfile:
        yaml.safe_dump(
            {'img_core-volume_'+'{:02d}'.format(int(aCBF_thrshld))+'_aCBF_mL': float(V_CBF_ST_thrshld)},
            outfile )
        yaml.safe_dump(
            {'img_core-volume_'+'{:02d}'.format(int(100*rCBF_thrshld))+'%_rCBF_mL': float(V_rCBF_ST_thrshld)},
            outfile )
else:
    my_res_file = os.path.join(res_fldr,"perfusion_outcome.yml")
    with open(my_res_file, 'a') as outfile:
        yaml.safe_dump(
            {'img_core-volume_'+'{:02d}'.format(int(aCBF_thrshld))+'_aCBF_mL': float(V_CBF_ST_thrshld)},
            outfile )

