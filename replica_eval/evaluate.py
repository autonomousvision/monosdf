import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
import glob

import trimesh
from pathlib import Path
import subprocess


scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]


root_dir = "../exps"
exp_name = "replica_mlp"
out_dir = "evaluation/replica_mlp"
Path(out_dir).mkdir(parents=True, exist_ok=True)

evaluation_txt_file = "evaluation/replica_mlp.csv"
evaluation_txt_file = open(evaluation_txt_file, 'w')


for idx, scan in enumerate(scans):
    idx = idx + 1
    # test set
    #if not (idx in [4, 6, 7]):
    #    continue
    
    cur_exp = f"{exp_name}_{idx}"
    cur_root = os.path.join(root_dir, cur_exp)
    # use first timestamps
    dirs = sorted(os.listdir(cur_root))
    cur_root = os.path.join(cur_root, dirs[0])
    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "plots/*.ply"))))
    
    files.sort(key=lambda x:os.path.getmtime(x))
    ply_file = files[-1]
    print(ply_file)

    # curmesh
    cull_mesh_out = os.path.join(out_dir, f"{scan}.ply")
    cmd = f"python cull_mesh.py --input_mesh {ply_file} --input_scalemat ../data/Replica/scan{idx}/cameras.npz --traj ../data/Replica/scan{idx}/traj.txt --output_mesh {cull_mesh_out}"
    print(cmd)
    os.system(cmd)

    cmd = f"python eval_recon.py --rec_mesh {cull_mesh_out} --gt_mesh ../data/Replica/cull_GTmesh/{scan}.ply"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",")
    print(output)
    
    evaluation_txt_file.write(f"{scan},{Path(ply_file).name},{output}")
    evaluation_txt_file.flush()