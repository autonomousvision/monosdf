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



# image [720, 1280]
# depth [720, 1280]
image_size = 384
trans_totensor = transforms.Compose([
    transforms.CenterCrop(720),
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
])
depth_trans_totensor = transforms.Compose([
    transforms.CenterCrop(720),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])


out_path_prefix = '../data/Apartment/'
data_root = '../nice-slam/'
scenes = ['Apartment']
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
    color_path = os.path.join(data_root, scene, 'color')
    color_paths = sorted(glob.glob(os.path.join(color_path, '*.jpg')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(color_paths)
    
    # load depth
    depth_path = os.path.join(data_root, scene, 'depth')
    depth_paths = sorted(glob.glob(os.path.join(depth_path, '*.png')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(depth_paths)

    # load intrinsic
    intrinsic_path = os.path.join(data_root, scene, 'intrinsic.json')
    camera_intrinsic = np.array(json.load(open(intrinsic_path))["intrinsic_matrix"]).reshape(3, 3).T
    print(camera_intrinsic)
    
    # load pose
    poses = []
    pose_path = os.path.join(data_root, scene, 'scene', 'trajectory.log')
    
    with open(pose_path) as f:
        content = f.readlines()

        # Load .log file.
        for i in range(0, len(content), 5):
            # format %d (src) %d (tgt) %f (fitness)
            data = list(map(float, content[i].strip().split(' ')))
            ids = (int(data[0]), int(data[1]))
            fitness = data[2]

            # format %f x 16
            c2w = np.array(
                list(map(float, (''.join(
                    content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

            poses.append(c2w)
    poses = np.stack(poses)
    print(poses.shape, len(depth_paths), len(color_paths))

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
    H, W = 720, 1280
    print(camera_intrinsic)
    # center crop by 720
    offset_x = (W - 720) * 0.5
    offset_y = (H - 720) * 0.5
    camera_intrinsic[0, 2] -= offset_x
    camera_intrinsic[1, 2] -= offset_y
    # resize
    resize_factor = 384 / 720.
    camera_intrinsic[:2, :] *= resize_factor
    
    K = np.eye(4)
    K[:3, :3] = camera_intrinsic
    print(K)
    
    for idx, (valid, pose, depth_path, image_path) in enumerate(zip(valid_poses, poses, depth_paths, color_paths)):
        print(idx, valid)
        if idx % 20 != 0: continue
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

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)
