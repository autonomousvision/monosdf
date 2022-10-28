import torch
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# copy from vis-mvsnet
def find_files(dir, exts=['*.png', '*.jpg']):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

# copy from vis-mvsnet
def load_cam(file: str):
    """ read camera txt file """
    cam = np.zeros((2, 4, 4))
    with open(file) as f:
        words = f.read().split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    return cam

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
            
# adatpted from https://github.com/dakshaau/ICP/blob/master/icp.py#L4 for rotation only 
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    AA = A
    BB = B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    return R


#TODO merge the following 4 function to one single function

# align depth map in the x direction from left to right
def align_x(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[1] == depth2.shape[1]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, :, s2:e2], depth1[:, :, s1:e1], torch.ones_like(depth1[:, :, s1:e1]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1], depth1.shape[2] + depth2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = depth1[:, :, :s1]
    result[:, :, depth1.shape[2]:] = depth2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    result[:, :, s1:depth1.shape[2]] = depth1[:, :, s1:] * weight + depth2_aligned[:, :, :e2] * (1 - weight)

    return result

# align depth map in the y direction from top to down
def align_y(depth1, depth2, s1, e1, s2, e2):
    assert depth1.shape[0] == depth2.shape[0]
    assert depth1.shape[2] == depth2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    # aligh depth2 to depth1
    scale, shift = compute_scale_and_shift(depth2[:, s2:e2, :], depth1[:, s1:e1, :], torch.ones_like(depth1[:, s1:e1, :]))

    depth2_aligned = scale * depth2 + shift   
    result = torch.ones((1, depth1.shape[1] + depth2.shape[1] - (e1 - s1), depth1.shape[2]))

    result[:, :s1, :] = depth1[:, :s1, :]
    result[:, depth1.shape[1]:, :] = depth2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1-s1))[None, :, None]
    result[:, s1:depth1.shape[1], :] = depth1[:, s1:, :] * weight + depth2_aligned[:, :e2, :] * (1 - weight)

    return result

# align normal map in the x direction from left to right
def align_normal_x(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[1] == normal2.shape[1]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, :, s2:e2].reshape(3, -1).T, normal1[:, :, s1:e1].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1], normal1.shape[2] + normal2.shape[2] - (e1 - s1)))

    result[:, :, :s1] = normal1[:, :, :s1]
    result[:, :, normal1.shape[2]:] = normal2_aligned[:, :, e2:]

    weight = np.linspace(1, 0, (e1-s1))[None, None, :]
    
    result[:, :, s1:normal1.shape[2]] = normal1[:, :, s1:] * weight + normal2_aligned[:, :, :e2] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

# align normal map in the y direction from top to down
def align_normal_y(normal1, normal2, s1, e1, s2, e2):
    assert normal1.shape[0] == normal2.shape[0]
    assert normal1.shape[2] == normal2.shape[2]

    assert (e1 - s1) == (e2 - s2)
    
    R = best_fit_transform(normal2[:, s2:e2, :].reshape(3, -1).T, normal1[:, s1:e1, :].reshape(3, -1).T)

    normal2_aligned = (R @ normal2.reshape(3, -1)).reshape(normal2.shape)
    result = np.ones((3, normal1.shape[1] + normal2.shape[1] - (e1 - s1), normal1.shape[2]))

    result[:, :s1, :] = normal1[:, :s1, :]
    result[:, normal1.shape[1]:, :] = normal2_aligned[:, e2:, :]

    weight = np.linspace(1, 0, (e1-s1))[None, :, None]
    
    result[:, s1:normal1.shape[1], :] = normal1[:, s1:, :] * weight + normal2_aligned[:, :e2, :] * (1 - weight)
    result = result / (np.linalg.norm(result, axis=0) + 1e-15)[None]
    
    return result

parser = argparse.ArgumentParser(description='Generate high resolution outputs')

parser.add_argument('--mode', required=True, help="choose from creating patches or merge patches")
args = parser.parse_args()

assert args.mode in ["create_patches", "merge_patches"]

# data-folder from vis-mvsnet
data_root = '../data/tanksandtemples/advanced'
scenes = ['Courtroom']

# temporary folders
out_path_prefix = "./highres_tmp"

# output folder for hihg-resolution cues 
out_path_for_training = '../data/highresTNT/'
out_scan_id = 1


for scene in scenes:
    # temporary folders for overlapped images
    out_path = os.path.join(out_path_prefix, "scan1")
    os.makedirs(out_path, exist_ok=True)
    print(out_path)
    out_folder = os.path.join(out_path, "image")
    os.makedirs(out_folder, exist_ok=True)
    print(out_folder)

    # high-resolutin image used for training
    out_path_for_training = os.path.join(out_path_for_training, f"scan{out_scan_id}")
    os.makedirs(out_path_for_training, exist_ok=True)

    # load poses and images
    pose_dir = os.path.join(data_root, scene, 'cams')
    images_dir = os.path.join(data_root, scene, "images")
    print(pose_dir)
    print(images_dir)

    poses = find_files(pose_dir, exts=["*.txt"])
    rgbs = find_files(images_dir, exts=["*.jpg"])
    
    # only use 3 images for debug
    # poses = poses[:3]
    # rgbs = rgbs[:3]

    out_index = 0
    cameras = {}
    cam_pos = []

    # created camera poses for monosdf training and save overlapped images to disk
    for pose_file, image in zip(poses, rgbs):
        # filter here
        pose = load_cam(pose_file)[0]
        intrinsic = load_cam(pose_file)[1]
        cam_pos.append(np.linalg.inv(pose)[:3, 3])
        
        # we will resize the image from 1080x1920to 1152x2048 because 1152 and 2048 have common factor 384 which is the size for Omnidata-model
        # So we crop a 360x360 patch and resize to 384x384
        # the overlapped region is 120 * 2 = 240 in eash side
        # and therefore we modify the camera intrinsic to match the resized image
        intrinsic[:2, :] *= 384 / 360.

        pose = intrinsic @ pose

        if args.mode == 'create_patches':
            image = cv2.imread(image)
        
            size = 360
            
            H, W = image.shape[:2]
            assert H == 1080
            assert W == 1920

            x = W // 120
            y = H // 120
            
            # crop images
            for j in range(y-2):
                for i in range(x-2):
                    image_cur = image[j*120:j*120+size, i*120:i*120+size, :]
                    print(image_cur.shape)
                    target_file = os.path.join(out_folder, "%06d_%02d_%02d.jpg"%(out_index, j, i))
                    print(target_file)
                    cv2.imwrite(target_file, image_cur)
            
            # save middle file for alignments
            image_cur = image[1080//2-180:1080//2+180, 1920//2-180:1920//2+180]

            print(image_cur.shape)
            target_file = os.path.join(out_folder, "%06d_mid.jpg"%(out_index))
            print(target_file)
            cv2.imwrite(target_file, image_cur)
        elif args.mode == 'merge_patches':
            #continue
            image = cv2.imread(image)
            image = cv2.resize(image, (2048, 1152))
            out_image_path = os.path.join(out_path_for_training, "%06d_rgb.png"%(out_index))
            cv2.imwrite(out_image_path, image)
        else:
            raise NotImplementedError

        scale_mat = np.eye(4).astype(np.float32)
        scale_mat = np.linalg.inv(scale_mat)

        cameras["scale_mat_%d"%(out_index)] = scale_mat
        cameras["world_mat_%d"%(out_index)] = pose

        out_index += 1
    
    # save camera poses
    np.savez(os.path.join(out_path_for_training, "cameras.npz"), **cameras)
    if args.mode == 'create_patches':
        exit(-1)
    
    
    out_index = 0
    
    H, W = 1080, 1920
    x = W // 120
    y = H // 120
    
    for pose_file, image in zip(poses, rgbs):
        depths_row = []
        # align depth maps from left to right row by row
        for j in range(y-2):            
            depths = []
            for i in range(x-2):
                depth_path = os.path.join(out_path, "%06d_%02d_%02d_depth.npy"%(out_index, j, i))
                depth = np.load(depth_path)
                depth = torch.from_numpy(depth)[None]
                depths.append(depth)
                
            # align from left to right
            depth_left = depths[0]
            s1 = 128
            s2 = 0
            e2 = 128 *2
            for depth_right in depths[1:]:
                depth_left = align_x(depth_left, depth_right, s1, depth_left.shape[2], s2, e2)
                s1 += 128
            depths_row.append(depth_left)
            print(depth_left.shape)

        depth_top = depths_row[0]
        # align depth maps from top to down
        s1 = 128
        s2 = 0
        e2 = 128 *2
        for depth_bottom in depths_row[1:]:
            depth_top = align_y(depth_top, depth_bottom, s1, depth_top.shape[1], s2, e2)
            s1 += 128

        # depth is up to scale so don't need to align to middle part
        mid_file = os.path.join(out_path, "%06d_mid_depth.npy"%(out_index))
        mid_depth = np.load(mid_file)
        mid_depth = torch.from_numpy(mid_depth)[None]
        
        scale, shift = compute_scale_and_shift(depth_top[:, 1152//2-192:1152//2+192, 2048//2-192:2048//2+192 ], mid_depth, torch.ones_like(mid_depth))
        depth_top = scale * depth_top + shift
        depth_top = (depth_top - depth_top.min()) / (depth_top.max() - depth_top.min())
        
        plt.imsave(os.path.join(out_path_for_training ,"%06d_depth.png"%(out_index)), depth_top[0].numpy(), cmap='viridis')
        np.save(os.path.join(out_path_for_training ,"%06d_depth.npy"%(out_index)), depth_top.detach().cpu().numpy()[0])    

        # normal
        normals_row = []
        # align normal maps from left to right row by row  
        for j in range(y-2):            
            normals = []
            for i in range(x-2):
                normal_path = os.path.join(out_path, "%06d_%02d_%02d_normal.npy"%(out_index, j, i))
                normal = np.load(normal_path)
                normal = normal * 2. - 1.
                normal = normal / (np.linalg.norm(normal, axis=0) + 1e-15)[None]
                normals.append(normal)
            
            # align from left to right
            normal_left = normals[0]
            s1 = 128
            s2 = 0
            e2 = 128 *2
            for normal_right in normals[1:]:
                normal_left = align_normal_x(normal_left, normal_right, s1, normal_left.shape[2], s2, e2)
                s1 += 128
            normals_row.append(normal_left)
            print(normal_left.shape)

        normal_top = normals_row[0]
        # align normal maps from top to down
        s1 = 128
        s2 = 0
        e2 = 128 *2
        for normal_bottom in normals_row[1:]:
            print(normal_top.shape, normal_bottom.shape)
            normal_top = align_normal_y(normal_top, normal_bottom, s1, normal_top.shape[1], s2, e2)
            s1 += 128

        # align to middle part
        mid_file = os.path.join(out_path, "%06d_mid_normal.npy"%(out_index))
        mid_normal = np.load(mid_file)
        mid_normal = mid_normal * 2. - 1.
        mid_normal = mid_normal / (np.linalg.norm(mid_normal, axis=0) + 1e-15)[None]
                
        R = best_fit_transform(normal_top[:, 1152//2-192:1152//2+192, 2048//2-192:2048//2+192 ].reshape(3, -1).T, mid_normal.reshape(3, -1).T)
        normal_top = (R @ normal_top.reshape(3, -1)).reshape(normal_top.shape)

        plt.imsave(os.path.join(out_path_for_training ,"%06d_normal.png"%(out_index)), np.moveaxis(normal_top, [0,1, 2], [2, 0, 1]) * 0.5 + 0.5)
        np.save(os.path.join(out_path_for_training ,"%06d_normal.npy"%(out_index)), (normal_top + 1.) / 2.)
        out_index += 1
