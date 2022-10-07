import glob
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

for scan in scans:
    fullres_image_dir = f"../data/DTU/scan{scan}/image"
    instance_dir = f"../data/DTU/scan{scan}/"
    out_dir = f"../data/DTU_padded_highres/scan{scan}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # copy camera
    
    cmd = f"cp {instance_dir}cameras.npz {out_dir}/"
    print(cmd)
    os.system(cmd)

    def glob_data(data_dir):
        data_paths = []
        data_paths.extend(glob.glob(data_dir))
        data_paths = sorted(data_paths)
        return data_paths
                
    image_paths = glob_data(os.path.join('{0}'.format(fullres_image_dir), "*.png"))
    np_depth_paths = glob_data(os.path.join('{0}'.format(instance_dir), "*_depth.npy"))
    np_normal_paths = glob_data(os.path.join('{0}'.format(instance_dir), "*_normal.npy"))

    H, W = 1200, 1600
    offset = (W-H) // 2

    for idx, (a, b, c) in enumerate(zip(image_paths, np_depth_paths, np_normal_paths)):

        image = cv2.imread(a)
        depth = np.load(b)
        normal = np.load(c)
        
        # bilinear upsample
        depth = F.interpolate(torch.from_numpy(depth)[None, None], (H, H), mode='bilinear', align_corners=True)[0, 0].numpy()
        normal = F.interpolate(torch.from_numpy(normal)[None], (H, H), mode='bilinear', align_corners=True)[0].numpy()
        
        #image = cv2.resize(image, (512, 384))
        cv2.imwrite(os.path.join(out_dir, "%06d_rgb.png"%(idx)), image)
        
        depth_paded = np.zeros((H, W))
        depth_paded[:, offset:offset+H] = depth
        
        normal_paded = np.zeros((3, H, W))
        normal_paded[:, :, offset:offset+H] = normal
        
        mask = np.zeros((H, W))
        mask[:, offset:offset+H] = 1.0
        
        np.save(os.path.join(out_dir, "%06d_depth.npy"%(idx)), depth_paded)
        np.save(os.path.join(out_dir, "%06d_normal.npy"%(idx)), normal_paded)
        np.save(os.path.join(out_dir, "%06d_mask.npy"%(idx)), mask)
        
        plt.imsave(os.path.join(out_dir, "%06d_depth.png"%(idx)), depth_paded)
        plt.imsave(os.path.join(out_dir, "%06d_normal.png"%(idx)), np.moveaxis(normal_paded, [0, 1, 2], [2, 0, 1]))
        plt.imsave(os.path.join(out_dir, "%06d_mask.png"%(idx)), mask)
    

    
    
    
    
    