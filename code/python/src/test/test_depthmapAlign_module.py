import configuration as config

from utility import image_io, serialization
from utility import depthmap_utils
from utility import fs_utility

import numpy as np

from utility.logger import Logger
log = Logger(__name__)
log.logger.propagate = False

try:
    from instaOmniDepth import depthmapAlign
except:
    log.error("Can not load depthmapAlign module from instaOmniDepth. Please build and install from c++ code.")


def test_numpy_data_transformation():
    # numpy test function
    print("Test 1: test numpy tranformation function.")
    test_array = np.arange(0, 20).reshape((4, -1))
    # depthmapAlign.python_numpy_mat_test(test_array.astype(np.int32))
    print("The following is python matrix data:")
    print(test_array)
    print("\nThe following is output of the CPP module:")
    depthmapAlign.python_numpy_mat_test(test_array.astype(np.float64))


def test_mock_data_generateion(data_root_dir, data_ref_depthmap_filename, data_tar_depthmap_filename,
                               corr_ref2tar_filename, corr_tar2ref_filename):
    # generate synthetic data for debug
    debug_data_type = 0  # 0 is simple debug data, 1 is random debug data
    frame_number = 2     # frame_number should be larger than 1

    depthpmap, align_coeff, pixels_corresponding_list = depthmapAlign.create_debug_data(debug_data_type, frame_number)
    print("depth map list: {}".format(depthpmap))
    print("align coeff ground truth list: {}".format(align_coeff))
    print("pixel corresponding relationship: {}".format(pixels_corresponding_list))
    # image_io.image_show(depthpmap[0])
    # image_io.image_show(depthpmap[1])

    # output data to file
    depthmap_utils.write_pfm(data_root_dir + data_ref_depthmap_filename, depthpmap[0])
    depthmap_utils.write_pfm(data_root_dir + data_tar_depthmap_filename, depthpmap[1])

    # output visualized depth map
    depthmap_utils.depth_visual_save(depthpmap[0], data_root_dir + data_ref_depthmap_filename + ".jpg")
    depthmap_utils.depth_visual_save(depthpmap[1], data_root_dir + data_tar_depthmap_filename + ".jpg")

    serialization.pixel_corresponding_save(data_root_dir + corr_ref2tar_filename,
                                           data_ref_depthmap_filename, "00",
                                           data_tar_depthmap_filename, "00", pixels_corresponding_list[0][1])
    serialization.pixel_corresponding_save(data_root_dir + corr_tar2ref_filename,
                                           data_tar_depthmap_filename, "00",
                                           data_ref_depthmap_filename, "00", pixels_corresponding_list[1][0])
    return depthpmap, pixels_corresponding_list


def test_mock_data_generateion_multiimage(data_dir_depthmap_align):
    # TODO call the cpp module's generate sequence synthetic image
    log.error("Not implement.")
    # generate synthetic data for debug
    debug_data_type = 1  # 0 is simple debug data, 1 is random debug data
    frame_number = 3     # frame_number should be larger than 1

    # 1) create the file name of the depth map, coefficient parameters, pixel corresponding.
    depthmap_filename_expression = "_disp_{:03d}.pfm"

    # 2) get them from cpp module
    depthpmap, align_coeff, pixels_corresponding_list = depthmapAlign.create_debug_data(debug_data_type, frame_number)

    # 3) output to files


def test_depthmap_align_module(data_root_dir, data_ref_depthmap_filename, data_tar_depthmap_filename,
                               corr_ref2tar_filename, corr_tar2ref_filename):
    # CPP Python Module test:
    # Test 1) different grid size
    # Test 2) group alignment with fixed frame

    # 0) load data from disk
    depthmap = []
    depthmap.append(depthmap_utils.read_pfm(data_root_dir + data_ref_depthmap_filename)[0].astype(np.float64))
    depthmap.append(depthmap_utils.read_pfm(data_root_dir + data_tar_depthmap_filename)[0].astype(np.float64))
    pixels_corresponding_list_ = []
    pixels_corresponding_list_.append(serialization.pixel_corresponding_load(data_root_dir + corr_ref2tar_filename))
    pixels_corresponding_list_.append(serialization.pixel_corresponding_load(data_root_dir + corr_tar2ref_filename))

    pixels_corresponding_list = {}
    pixels_corresponding_list[0] = {}
    pixels_corresponding_list[0][1] = pixels_corresponding_list_[0]["pixel_corresponding"]
    pixels_corresponding_list[1] = {}
    pixels_corresponding_list[1][0] = pixels_corresponding_list_[1]["pixel_corresponding"]

    # align sub-image's depth maps, and depthmap_stitch's comment
    root_dir = data_root_dir
    method = "group"
    depthmap_original_ico_index = [0, 1]
    terms_weight = [1.0, 1.0, 0.0]
    depthmap_original = depthmap
    depthmap_aligned = None
    align_coeff = None
    reference_depthamp_ico_index = 1  # the fixed reference frame ico index

    align_coeff_grid_height = 10
    align_coeff_grid_width = 5
    align_coeff_scale = []
    align_coeff_offset = []

    reproj_perpixel_enable = 0
    smooth_pergrid_enable = 0

    for _ in range(0, len(depthmap)):
        align_coeff_initial_scale = np.full((align_coeff_grid_height, align_coeff_grid_width), 0.1, np.float64)
        align_coeff_scale.append(align_coeff_initial_scale)
        align_coeff_initial_offset = np.full((align_coeff_grid_height, align_coeff_grid_width), 0.0, np.float64)
        align_coeff_offset.append(align_coeff_initial_offset)
    try:
        depthmapAlign.init(method)

        depthmapAlign.ceres_solver_option(12,  10, -1, -1)

        depthmap_aligned, align_coeff = depthmapAlign.depthmap_stitch(
            root_dir, 
            terms_weight,
            depthmap_original,
            depthmap_original_ico_index,
            reference_depthamp_ico_index,
            pixels_corresponding_list,
            align_coeff_grid_height, 
            align_coeff_grid_width,
            reproj_perpixel_enable,
            smooth_pergrid_enable,
            align_coeff_scale, 
            align_coeff_offset,
            0)

        depthmapAlign.report_aligned_depthmap_error()

        depthmapAlign.shutdown()
        
    except RuntimeError as error:
        print('Error: ' + repr(error))

    # print(depthmap_aligned)
    print(align_coeff)
    # depthmap_utils.depth_visual_save(depthmap_aligned[0], "./00.jpg")
    # image_io.image_show(depthmap_aligned[0])
    # output to folder
    # output visualized depth map
    depthmap_utils.depth_visual_save(depthmap_aligned[0], data_root_dir + data_ref_depthmap_filename + "_align.jpg")
    depthmap_utils.depth_visual_save(depthmap_aligned[1], data_root_dir + data_tar_depthmap_filename + "_align.jpg")


if __name__ == "__main__":

    test_list = []
    args = config.get_parser()
    test_list.append(args.task)

    data_root_dir = config.TEST_DATA_DIR
    data_dir_depthmap_align = data_root_dir + "depthmap_align/"
    fs_utility.dir_make(data_dir_depthmap_align)

    data_ref_depthmap_filename = "img0_depth_000.pfm"
    data_tar_depthmap_filename = "img0_depth_001.pfm"
    # depthmap_filename_exp = "img0_depth_"

    # data_ref_aligned_depthmap_filename = "img0_depth_000_aligned.pfm"
    # data_tar_aligned_depthmap_filename = "img0_depth_001_aligned.pfm"

    corr_ref2tar_filename = "img0_corr_000_001.json"
    corr_tar2ref_filename = "img0_corr_001_000.json"
    coeff_filename = "img0_coeff.json"

    if 0 in test_list:
        test_numpy_data_transformation()

    if 1 in test_list:
        test_mock_data_generateion(data_dir_depthmap_align, data_ref_depthmap_filename, data_tar_depthmap_filename,
                                   corr_ref2tar_filename, corr_tar2ref_filename)

    if 2 in test_list:
        test_mock_data_generateion_multiimage(data_dir_depthmap_align)

    if 3 in test_list:
        # need the mock data
        test_depthmap_align_module(data_dir_depthmap_align, data_ref_depthmap_filename, data_tar_depthmap_filename,
                                   corr_ref2tar_filename, corr_tar2ref_filename)
