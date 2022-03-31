import configuration as config
from utility.cam_models import *
import cv2
import numpy as np

import json
import os
import sys

# Add project library to Python path
python_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding '{python_src_dir}' to sys.path")
sys.path.append(python_src_dir)  # /code/python/src/
sys.path.append(os.path.dirname(python_src_dir))  # /code/python/
# sys.path.append(os.path.join(python_src_dir, "../.."))  # CR: unused?

# Data directory /data/
# TODO: remove the trailing slash once all usages of TEST_DATA_DIR are updated
TEST_DATA_DIR = os.path.abspath("../../../../data/") + "/"

# Set the PyTorch hub folder as an environment variable
# TODO: use os.path.join instead of string
os.environ['TORCH_HOME'] = TEST_DATA_DIR + 'models/'

if __name__ == "__main__":
    data_root = config.TEST_DATA_DIR

    json_file = json.load(open(data_root + 'brown_00/calibration.json', 'r'))  # Calibration file from Elliot

    # There are 6 cameras. I dont know to which one belongs the image under test. I use all of them.
    camera_params = json_file['cameras']
    img = cv2.cvtColor(cv2.imread(data_root + "brown_00/video-frame-3840x2880.png"), cv2.COLOR_BGR2RGB)
    img = np.moveaxis(img, 0, 1)[::-1, ...]     # Rotate image 90º counter clockwise

    for cam in camera_params:
        #   Uncomment to undistort the whole input image
        # mapx, mapy = create_perspective_undistortion_LUT(img.shape[:-1], ind_camera, sf=-7)
        # undistorted = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR,
        #                         borderMode=cv2.BORDER_CONSTANT)
        RGB_subimages, depth_subimages, camera_array_size = sample_img(img, cam, run_midas=False)
        plot_img_array(camera_array_size, RGB_subimages)
        plot_img_array(camera_array_size, depth_subimages)
