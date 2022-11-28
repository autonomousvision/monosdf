from glob import glob
import os
import sys
from turtle import width
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


ply_file = sys.argv[1]

ply = o3d.io.read_triangle_mesh(ply_file)
ply.compute_vertex_normals()
ply.paint_uniform_color([1, 1, 1])
vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window("rendering", width=1920, height=1080)

index = -1

def move_forward(vis):
    # This function is called within the o3d.visualization.Visualizer::run() loop
    # The run loop calls the function, then re-render
    # So the sequence in this function is to:
    # 1. Capture frame
    # 2. index++, check ending criteria
    # 3. Set camera
    # 4. (Re-render)
    ctr = vis.get_view_control()
    
    global index

    if index >= 0:
        vis.capture_screen_image(os.path.join(sys.argv[2], "render_{}.jpg".format(index)), False)
    index = index + 1
    if index < 240:
        param = o3d.io.read_pinhole_camera_parameters(sys.argv[3] + "/tmp%d.json"%(index))
        ctr.convert_from_pinhole_camera_parameters(
            param, allow_arbitrary=True)
    else:
        vis.register_animation_callback(None)
        exit(-1)

    return False


vis.add_geometry(ply)
vis.get_render_option().load_from_json('render.json')

vis.register_animation_callback(move_forward)
vis.run()
vis.destroy_window()