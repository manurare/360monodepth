import configuration as config

from utility import depthmap_utils
from utility import image_io

import numpy as np

import pathlib


from utility.logger import Logger
log = Logger(__name__)
log.logger.propagate = False

def test_depth_ico_visual_save(depth_filepath_expression, output_path, file_number):
    dispmap_list = []
    for index in range(0, file_number):
        depth_filepath = depth_filepath_expression.format(index)
        dispmap, _ = depthmap_utils.read_pfm(str(depth_filepath))
        dispmap_list.append(dispmap)
    depthmap_utils.depth_ico_visual_save(dispmap_list, output_path)


def test_run_midas(image_filepath_expression, depth_filepath_expression, file_number):
    for index in range(0, file_number):
        rgb_filepath = image_filepath_expression.format(index)
        depth_filepath = depth_filepath_expression.format(index)

        log.info("Estimate dispmap from {} to \n\t{}.".format(rgb_filepath, depth_filepath))

        depthmap_data = depthmap_utils.rgb2dispmap(rgb_filepath, True)
        depthmap_utils.write_pfm(depth_filepath, depthmap_data, scale=1)
        depthmap_utils.depth_visual_save(depthmap_data, depth_filepath + ".jpg")


def test_visualize_depth_map(data_dir):
    """visualize the pfm file and output
    """
    data_dir =  pathlib.Path(data_dir)
    for file_name in data_dir.iterdir():
        filepath =  data_dir / file_name
        print(filepath)
        if filepath.is_dir():
            test_visualize_depth_map(str(filepath))

        # visualize pfm file
        if filepath.suffix == ".pfm":
            dispmap, _ = depthmap_utils.read_pfm(str(filepath))
            dispmap_vis_filepath = str(filepath) + ".jpg"
            depthmap_utils.depth_visual_save(dispmap, str(dispmap_vis_filepath))
            log.debug("visualize disparity map {}".format(str(filepath)))
        
        if filepath.suffix == ".dpt":
            dispmap = depthmap_utils.read_dpt(str(filepath))
            dispmap_vis_filepath = str(filepath) + ".jpg"
            depthmap_utils.depth_visual_save(dispmap, str(dispmap_vis_filepath))
            log.debug("visualize depth map {}".format(str(filepath)))


def test_read_exr(exr_file_path):
    import cv2
    depth_min = 0
    depth_max = 10

    depthmap_cv = np.array(cv2.imread(exr_file_path, cv2.IMREAD_ANYDEPTH))
    depthmap_utils.depthmap_histogram(depthmap_cv)
    print("The depth map range is [{}, {}]" ,)


    depthmap_cv = np.clip(depthmap_cv, depth_min, depth_max)
    depthmap_utils.depth_visual_save(depthmap_cv, exr_file_path + "_cv.jpg")

    depthmap_openexr = depthmap_utils.read_exr(exr_file_path)
    depthmap_openexr = np.clip(depthmap_openexr, depth_min, depth_max)
    depthmap_utils.depth_visual_save(depthmap_openexr, exr_file_path + "_openexr.jpg")

    diff = depthmap_cv - depthmap_openexr
    print("The difference mean is {}".format(np.mean(diff)))


def test_depthmap_pyramid():  
    from scipy import interpolate
    # create original depth map
    depthmap_number = 3

    depthmap_grid_width = 5
    depthmap_grid_height = 8

    depthmap_width = 50
    depthmap_height = 80

    depthmap_list = []

    for _ in range(0, depthmap_number):
        grid_mat = np.random.rand(depthmap_grid_height, depthmap_grid_width)
        x = np.array(range(depthmap_grid_width))#.astype(np.float64)/ depthmap_width * depthmap_grid_width
        y = np.array(range(depthmap_grid_height))#.astype(np.float64)/ depthmap_height * depthmap_grid_height
        depthmap_sampler = interpolate.interp2d(x, y, grid_mat, kind='linear')

        # sample larger image
        depthmap_x = np.array(range(depthmap_width)).astype(np.float64)/ depthmap_width * depthmap_grid_width
        depthmap_y = np.array(range(depthmap_height)).astype(np.float64)/ depthmap_height * depthmap_grid_height
        # depthmap_xx, depthmap_yy = np.meshgrid(depthmap_x, depthmap_y)
        depthmap_zz = depthmap_sampler(depthmap_x, depthmap_y).reshape(depthmap_height, depthmap_width)
        # image_io.image_show(grid_mat)
        # image_io.image_show(depthmap_zz)
        depthmap_list.append(depthmap_zz)

    # make depth map pyramid
    pyramid_level = 3
    downsample_ratio = 0.8
    depthmap_pyramid = depthmap_utils.depthmap_pyramid(depthmap_list, pyramid_level, downsample_ratio)

    # show result
    image_io.image_show_pyramid(depthmap_pyramid)


if __name__ == "__main__":

    test_list = [4]

    if 0 in test_list:
        depth_filepath_expression = config.TEST_DATA_DIR + "erp_00/0001_rgb_disp_{:03d}.pfm"
        output_path = config.TEST_DATA_DIR + "erp_00/0001_rgb_disp_all.pfm"
        test_depth_ico_visual_save(depth_filepath_expression, output_path, 20)

    if 1 in test_list:
        data_root = config.TEST_DATA_DIR
        erp_image_dir = data_root + "erp_00/"
        image_filepath_expression = erp_image_dir + "cubemap_rgb_padding_{}.jpg"
        depth_filepath_expression = erp_image_dir + "cubemap_rgb_padding_{}.pfm"
        file_number = 6
        test_run_midas(image_filepath_expression, depth_filepath_expression, file_number)

    if 2 in test_list:
        # pfm_dir = pathlib.Path(config.TEST_DATA_DIR + "erp_00/")
        # pfm_dir = "/mnt/sda1/workdata/opticalflow_data/replica_360_2k"
        pfm_dir = "D:/workspace_windows/InstaOmniDepth/InstaOmniDepth_github/code/cpp/bin/Release/"
        test_visualize_depth_map(pfm_dir)

    if 3 in test_list:
        exr_file_path = config.TEST_DATA_DIR + "erp_02/273_area_21_depth_0_Left_Down_0.0.exr"
        test_read_exr(exr_file_path)

    if 4 in test_list:
        test_depthmap_pyramid()
