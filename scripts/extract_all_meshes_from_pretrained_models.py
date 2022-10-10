import os
os.makedirs("meshes", exist_ok=True)

confs, checkpoints, scan_ids, resolutions, out_folders = [], [], [], [], []

# DTU
dtu_scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

pairs = [
    # pretrained, conf, only need to match the network architecture
    ("dtu_3views_grids", "dtu_grids_3views.conf"),
    ("dtu_3views_mlp", "dtu_mlp_3views.conf"),
    ("dtu_allviews_grids", "dtu_grids_3views.conf"),
    ("dtu_allviews_mlp", "dtu_mlp_3views_4layercolor.conf"),
]
for ckpt, conf in pairs:
    for scan_id in dtu_scans:
        confs.append(conf)
        checkpoints.append(os.path.join("../pretrained_models/", ckpt, f"scan{scan_id}.pth"))
        scan_ids.append(scan_id)
        resolutions.append(512)
        out_folders.append(os.path.join("../meshes/", ckpt))

# ScanNet, post-processing is needed for ScanNet
ScanNet_scans = [1, 2, 3, 4]

pairs = [
    # pretrained, conf, only the network architecture needs to match
    ("scannet_grids", "scannet_grids.conf"),
    ("scannet_mlp", "scannet_mlp.conf"),
]
for ckpt, conf in pairs:
    for scan_id in ScanNet_scans:
        confs.append(conf)
        checkpoints.append(os.path.join("../pretrained_models/", ckpt, f"scan{scan_id}.pth"))
        scan_ids.append(scan_id)
        resolutions.append(512)
        out_folders.append(os.path.join("../meshes/", ckpt))

# tnt lowres
tnt_scans = [1, 2, 3, 4]

pairs = [
    # pretrained, conf, only the network architecture needs to match
    ("tnt_lowres_grids", "tnt_grids.conf"),
    ("tnt_lowres_mlp", "tnt_mlp.conf"),
]
for ckpt, conf in pairs:
    for scan_id in tnt_scans:
        confs.append(conf.replace(".conf", f"_{scan_id}.conf"))
        checkpoints.append(os.path.join("../pretrained_models/", ckpt, f"scan{scan_id}.pth"))
        scan_ids.append(scan_id)
        resolutions.append(1024)
        out_folders.append(os.path.join("../meshes/", ckpt))

for conf, checkpoint, scan_id, resolution, out_folder in zip(confs, checkpoints, scan_ids, resolutions, out_folders):
    cmd = f"cd code && python evaluation/eval.py --conf confs/{conf} --checkpoint {checkpoint} --scan_id {scan_id} --resolution {resolution} --evals_folder {out_folder}"
    print(cmd)
    os.system(cmd)

# highresTNT, only need to match the architecture and grid boundary when extract mesh
cmd = f"cd code && python evaluation/eval.py --conf confs/tnt_grids_1.conf --checkpoint ../pretrained_models/tnt_highres/Courtroom.pth --scan_id 1 --resolution 4096 --evals_folder ../meshes/tnt_highres/"
print(cmd)
os.system(cmd)