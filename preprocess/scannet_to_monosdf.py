import numpy as np
import cv2
import torch
import os
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import json
import trimesh
import glob
import PIL
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

image_size = 384
trans_totensor = transforms.Compose([
    transforms.CenterCrop(image_size*2),
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
])
depth_trans_totensor = transforms.Compose([
    transforms.Resize([968, 1296], interpolation=PIL.Image.NEAREST),
    transforms.CenterCrop(image_size*2),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])


out_path_prefix = '../data/custom'
data_root = '/home/yuzh/Projects/datasets/scannet/'
scenes = ['scene0050_00']
out_names = ['scan1']

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    folders = ["image", "mask", "depth"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    # load color 
    color_path = os.path.join(data_root, scene, 'frames', 'color')
    color_paths = sorted(glob.glob(os.path.join(color_path, '*.jpg')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(color_paths)
    
    # load depth
    depth_path = os.path.join(data_root, scene, 'frames', 'depth')
    depth_paths = sorted(glob.glob(os.path.join(depth_path, '*.png')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(depth_paths)

    # load intrinsic
    intrinsic_path = os.path.join(data_root, scene, 'frames', 'intrinsic', 'intrinsic_color.txt')
    camera_intrinsic = np.loadtxt(intrinsic_path)
    print(camera_intrinsic)

    # load pose
    pose_path = os.path.join(data_root, scene, 'frames', 'pose')
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for pose_path in pose_paths:
        c2w = np.loadtxt(pose_path)
        poses.append(c2w)
    poses = np.array(poses)

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print(center, scale)

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    out_index = 0
    cameras = {}
    pcds = []
    H, W = 968, 1296

    # center crop by 2 * image_size
    offset_x = (W - image_size * 2) * 0.5
    offset_y = (H - image_size * 2) * 0.5
    camera_intrinsic[0, 2] -= offset_x
    camera_intrinsic[1, 2] -= offset_y
    # resize from 384*2 to 384
    resize_factor = 0.5
    camera_intrinsic[:2, :] *= resize_factor
    
    K = camera_intrinsic
    print(K)
    
    for idx, (valid, pose, depth_path, image_path) in enumerate(zip(valid_poses, poses, depth_paths, color_paths)):
        print(idx, valid)
        if idx % 10 != 0: continue
        if not valid : continue
        
        target_image = os.path.join(out_path, "image/%06d.png"%(out_index))
        print(target_image)
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        mask = (np.ones((image_size, image_size, 3)) * 255.).astype(np.uint8)

        target_image = os.path.join(out_path, "mask/%03d.png"%(out_index))
        cv2.imwrite(target_image, mask)

        # load depth
        target_image = os.path.join(out_path, "depth/%06d.png"%(out_index))
        depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        #import pdb; pdb.set_trace()
        depth_PIL = Image.fromarray(depth)
        new_depth = depth_trans_totensor(depth_PIL)
        new_depth = np.asarray(new_depth)
        plt.imsave(target_image, new_depth, cmap='viridis')
        np.save(target_image.replace(".png", ".npy"), new_depth)
        
        
        # save pose
        pcds.append(pose[:3, 3])
        pose = K @ np.linalg.inv(pose)
        
        #cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d"%(out_index)] = scale_mat
        cameras["world_mat_%d"%(out_index)] = pose

        out_index += 1

    #np.savez(os.path.join(out_path, "cameras_sphere.npz"), **cameras)
    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)
