from pathlib import Path
import numpy as np
import os
from glob import glob
import cv2
import open3d as o3d
import torch
import trimesh
import trimesh
import json


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def load(config_file):
    tmp_json = json.load(open(config_file))
    extrinsic = np.array(tmp_json["extrinsic"]).reshape(4, 4).T
    pose = np.linalg.inv(extrinsic)
    return pose    


def render_scan(scan_id, mesh, out_path):
    
    instance_dir = os.path.join(data_dir, 'scan{0}'.format(scan_id))
    
    image_paths = glob_data(os.path.join('{0}'.format(instance_dir), "*_rgb.png"))
    n_images = len(image_paths)
    
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    scale_mats_0 = np.load(cam_file)['scale_mat_0']

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)

        intrinsics_all.append(intrinsics)
        pose_all.append(pose)
        
    H, W = 1080, 1920

    # create tmp camera pose file for open3d
    camera_config = Path("video_poses")
    camera_config.mkdir(exist_ok=True, parents=True)

    for image_id in range(n_images):
        
        c2w = pose_all[image_id]
        w2c = np.linalg.inv(c2w)

        K = intrinsics_all[0].copy()
        K[:2, :] *= 2.
     
        tmp_json = json.load(open('c1.json'))
        tmp_json["extrinsic"] = w2c.T.reshape(-1).tolist()
        
        tmp_json["intrinsic"]["intrinsic_matrix"] = K[:3,:3].T.reshape(-1).tolist()
        tmp_json["intrinsic"]["height"] = H 
        tmp_json["intrinsic"]["width"] = W 
        json.dump(tmp_json, open('video_poses/tmp%d.json'%(image_id), 'w'), indent=4)
    
    cmd = f"python render_trajectory_open3d.py {mesh} \"{out_path}\" {camera_config}"
    os.system(cmd)


data_dir = '../data/tnt_advanced'
scan = 1
mesh_path = 'courtroom.ply'
out_path = './rendering/'
Path(out_path).mkdir(exist_ok=True, parents=True)

render_scan(scan, mesh_path, out_path)

