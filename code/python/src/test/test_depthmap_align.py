import configuration as config
from utility.fs_utility import FileNameConvention as FNC

from utility import fs_utility
from utility import depthmap_align
from utility import depthmap_utils

import os
import numpy as np
from PIL import Image

from utility.logger import Logger
log = Logger(__name__)
log.logger.propagate = False

# test single image resultion and single deformation gride
def test_alignment_grid_single_res_single(data_root, erp_image_filepath, padding_size = 0.1):
    """align the depth map on single level grid and single resolution image.
    """
    # 0) load disparity maps from disk
    fnc = FNC()
    fnc.set_filename_basename("0001_rgb")
    fnc.set_filepath_folder(data_root + "debug/")
    subimage_depthmap_list = []
    for index in range(0, 20):
        subimage_depthmap_filepath = fnc.subimage_depthmap_filename_expression.format(index)
        print("load depth map {}".format(subimage_depthmap_filepath))
        dispmap = depthmap_utils.read_pfm(subimage_depthmap_filepath)[0]
        subimage_depthmap_list.append(depthmap_utils.disparity2depth(dispmap))

    # 1) align the depth map
    # stitch with single-scale
    depthmap_aligner = depthmap_align.DepthmapAlign()
    depthmap_aligner.root_dir = ""  # output alignment coefficient
    depthmap_aligner.pyramid_layer_number = 1
    depthmap_aligner.subsample_pixelcorr_ratio = 1.0
    depthmap_aligner.align_method = "group"
    depthmap_aligner.weight_project = 1.0
    depthmap_aligner.weight_smooth = 0.1
    depthmap_aligner.weight_scale = 0.0
    depthmap_aligner.depthmap_norm_mothod = "midas"

    erp_image_data = np.asarray(Image.open(data_root + erp_image_filepath))
    depthmap_aligned, coeffs_scale, coeffs_offset, subimage_cam_param_list = \
        depthmap_aligner.align_multi_res(erp_image_data, subimage_depthmap_list, padding_size)

    # 2) save the alignment result
    for index in range(0, len(depthmap_aligned)):
        depthmap_utils.write_pfm(FNC.aligned_depth_filepath_expression.format(index), depthmap_aligned[index].astype(np.float32), scale=1)
        depthmap_utils.depth_visual_save(depthmap_aligned[index], FNC.aligned_depth_filepath_expression.format(index) + "_vis.jpg")


def test_alignment_grid_multi_res_single(data_dir_depthmap_align):
    pass


def test_alignment_grid_multi_res_multi(data_dir_depthmap_align):
    """
    Align the 20 subimages depth map.
    """
    erp_image_dir = data_dir_depthmap_align + "3D60_00/"
    erp_image_filename = "rgb_00.png"
    padding_size = 0.4
    erp_image_filepath = erp_image_dir + erp_image_filename
    filename_base, _ = os.path.splitext(erp_image_filename)


    # 0) load disparity maps
    subimage_depthmap_list = []
    for index in range(0, 20):
        subimage_depthmap_filepath = FNC.depth_filepath_expression.format(index)
        print("load depth map {}".format(subimage_depthmap_filepath))
        dispmap = depthmap_utils.read_pfm(subimage_depthmap_filepath)[0]
        subimage_depthmap_list.append(depthmap_utils.disparity2depth(dispmap))

    # stitch with mulit-scale
    depthmap_aligner = depthmap_align.DepthmapAlign()
    depthmap_aligner.root_dir = ""  # output alignment coefficient
    depthmap_aligner.pyramid_layer_number = 1
    depthmap_aligner.subsample_pixelcorr_ratio = 0.4
    depthmap_aligner.align_method = "group"
    depthmap_aligner.weight_project = 1.0
    depthmap_aligner.weight_smooth = 0.1
    depthmap_aligner.weight_scale = 0.0
    depthmap_aligner.depthmap_norm_mothod = "midas"

    erp_image_data = np.asarray(Image.open(erp_image_filepath))
    depthmap_aligned, coeffs_scale, coeffs_offset, subimage_cam_param_list = \
        depthmap_aligner.align_multi_res(erp_image_data, subimage_depthmap_list, padding_size)

    for index in range(0, len(depthmap_aligned)):
        depthmap_utils.write_pfm(FNC.aligned_depth_filepath_expression.format(index), depthmap_aligned[index].astype(np.float32), scale=1)
        depthmap_utils.depth_visual_save(depthmap_aligned[index], FNC.aligned_depth_filepath_expression.format(index) + "_vis.jpg")


if __name__ == "__main__":
    data_root_dir = config.TEST_DATA_DIR
    data_dir_depthmap_align = data_root_dir  + "erp_00/"# + "depthmap_align/"
    fs_utility.dir_make(data_dir_depthmap_align)

    erp_image_filepath = "0001_rgb.jpg"

    test_list = [0]

    if 0 in test_list:
        # test single image resultion and single deformation gride
        test_alignment_grid_single_res_single(data_dir_depthmap_align, erp_image_filepath)

    if 1 in test_list:
        test_alignment_grid_multi_res_single(data_dir_depthmap_align, erp_image_filepath)

    if 2 in test_list:
        test_alignment_grid_multi_res_multi(data_dir_depthmap_align, erp_image_filepath)
