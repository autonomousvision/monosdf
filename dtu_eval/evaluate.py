from logging import root
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse
import trimesh
from pathlib import Path
import subprocess

from evaluate_single_scene import cull_scan


# Ground truth DTU point cloud path
Offical_DTU_Dataset = "./Offical_DTU_Dataset"
scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

out_dir_prefix = "evaluation/"
Path(out_dir_prefix).mkdir(parents=True, exist_ok=True)

# output file to save quantitative results
evaluation_txt_file = "evaluation/DTU.csv"
evaluation_txt_file = open(evaluation_txt_file, 'w')

root_dir = '../exps/'
exp_names =["dtu_3views"]

for exp in exp_names:    
    for scan in scans:
        out_dir = os.path.join(out_dir_prefix, str(scan))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        vis_out_dir = os.path.join(out_dir_prefix, exp)
        Path(vis_out_dir).mkdir(parents=True, exist_ok=True)
        
        cur_root = os.path.join(root_dir, f"{exp}_{scan}")
        
        files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "*/plots/*.ply"))))
        files.sort(key=lambda x:os.path.getmtime(x))
        
        for ply_file in files[-1:]:    
            iter_num = Path(ply_file).stem
            cur_vis_out_dir = os.path.join(out_dir_prefix, exp)
            Path(cur_vis_out_dir).mkdir(parents=True, exist_ok=True)
        
            print(ply_file)

            # delete mesh by mask
            result_mesh_file = os.path.join(out_dir, f"{exp}_{iter_num}.ply")
            cull_scan(scan, ply_file, result_mesh_file)
            
            cmd = f"python eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {cur_vis_out_dir}"
            print(cmd)
            #acc, comp, overall 
            output = subprocess.check_output(cmd, shell=True).decode("utf-8")
            output = output.replace(" ", ",")
            
            evaluation_txt_file.write(f"{exp},{scan},{iter_num},{output}")
            evaluation_txt_file.flush()
