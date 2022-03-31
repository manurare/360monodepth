import configuration as config

from utility import subimage
from utility import serialization
from utility import depthmap_utils
from utility import blending
from utility import projection_icosahedron as proj_ico


import numpy as np
from PIL import Image

import os
from pathlib import Path


def test_erp_ico_cam_intrparams(subimage_rgb_filepath_exp, output_dir, padding_size):
    """
    Tansfrom the face_6 (source) to face_12 (target) with camera parameters.
    """
    # # print the camera parameter
    # image_width = 400
    # image_height = 400
    # padding_size = 0
    # intrparams = subimage.erp_ico_cam_intrparams(image_width, image_height, padding_size)
    # print(intrparams)

    # load rgb image
    subimage_index_src = 5
    subimage_index_tar = 10
    subimage_rgb_filepath_src = subimage_rgb_filepath_exp.format(subimage_index_src)
    subimage_rgb_filepath_tar = subimage_rgb_filepath_exp.format(subimage_index_tar)
    subimage_rgb_data_src = np.asarray(Image.open(subimage_rgb_filepath_src))
    subimage_rgb_data_tar = np.asarray(Image.open(subimage_rgb_filepath_tar))

    tangent_image_height = subimage_rgb_data_src.shape[0]
    tangent_image_width = subimage_rgb_data_src.shape[1]

    # get source and target face camera parameter & data
    subimage_cam_param_list = subimage.erp_ico_cam_intrparams(tangent_image_width, padding_size)
    subimage_cam_parameters_src = subimage_cam_param_list[subimage_index_src]
    print(subimage_cam_parameters_src)
    subimage_cam_parameters_tar = subimage_cam_param_list[subimage_index_tar]
    print(subimage_cam_parameters_tar)

    # tranform the image from source to target
    # 1) Transform pixel form src to world
    points_2d_src_y, points_2d_src_x = np.mgrid[0:tangent_image_height, 0:tangent_image_width]
    points_2d_src = np.squeeze(np.dstack((points_2d_src_x.flatten(), points_2d_src_y.flatten())))
    points_2d_src_homo = np.hstack((points_2d_src, np.ones((tangent_image_width * tangent_image_height, 1))))
    points_3d_src = np.linalg.inv(subimage_cam_parameters_src['intrinsics']['matrix']) @ points_2d_src_homo.T
    points_3d_src_world = np.linalg.inv(subimage_cam_parameters_src['rotation']) @ points_3d_src
    # 2) Transform pixel world src to target
    points_3d_tar_homo = (subimage_cam_parameters_tar['rotation'] @ points_3d_src_world + subimage_cam_parameters_tar['translation'][:, np.newaxis])
    points_2d_tar_homo = subimage_cam_parameters_tar['intrinsics']['matrix'] @ points_3d_tar_homo
    points_2d_tar_homo = (np.divide(points_2d_tar_homo, points_2d_tar_homo[2, :]))

    # 3) sample the image
    from scipy import ndimage
    tangent_image_src2tar = np.zeros_like(subimage_rgb_data_src)
    for channel in range(0, tangent_image_src2tar.shape[2]):
        tangent_image_src2tar[points_2d_src_y, points_2d_src_x, channel] = \
            ndimage.map_coordinates(subimage_rgb_data_tar[:, :, channel],
                                    [points_2d_tar_homo[1, :], points_2d_tar_homo[0, :]], order=1, mode='constant', cval=0.0) \
                .reshape((tangent_image_height, tangent_image_width))

    # output
    filename = output_dir + "src2tar_{}_{}.jpg".format(subimage_index_src, subimage_index_tar)
    Image.fromarray(tangent_image_src2tar.astype(np.uint8)).save(filename)


def test_ico_proj_rgb(erp_image_filepath, padding_size, subimage_size, output_dir, filename_base, debug=False, intermediate_file=True, corr_subsample_factor = 1.0):
    # load erp image
    erp_image_data = np.asarray(Image.open(erp_image_filepath))

    # project to subimage
    subimage_list, camera_param_list, pixel_corr_list = \
        subimage.erp_ico_proj(erp_image_data, padding_size, subimage_size, corr_subsample_factor)

    # output to files
    if intermediate_file:
        print("Output data to file.")
        serialization.save_subimages_data(output_dir, filename_base, subimage_list, camera_param_list, pixel_corr_list, output_corr2file=False)

    # test corresponding
    if debug:
        print("Test the pixels correonding relationship.")
        for index_src in range(0, len(subimage_list)):
            for index_tar in range(0, len(subimage_list)):
                if index_src == index_tar:
                    continue
                src_image_data_image, tar_image_data_image, src_warp_image = \
                    subimage.draw_corresponding(subimage_list[index_src], subimage_list[index_tar], pixel_corr_list[index_src][index_tar])

                src_image_output_path = output_dir + "{:03d}_{:03d}_src.jpg".format(index_src, index_tar)
                Image.fromarray(src_image_data_image).save(src_image_output_path)
                tar_image_output_path = output_dir + "{:03d}_{:03d}_tar.jpg".format(index_src, index_tar)
                Image.fromarray(tar_image_data_image).save(tar_image_output_path)
                src_warp_image_path = output_dir + "{:03d}_{:03d}_src_warp.jpg".format(index_src, index_tar)
                Image.fromarray(src_warp_image).save(src_warp_image_path)

                # print(src_image_output_path)
                # print(tar_image_output_path)
                print(src_warp_image_path)


def test_ico_stitch_rgb(subimage_dir, erp_stitched_image_filepath, subimage_filepath_expression,
                        padding_size, erp_image_height, debug_output_dir):
    # 0) load disparity maps
    subimage_list = []
    for index in range(0, 20):
        subimage_filepath = subimage_filepath_expression.format(index)
        subimage_data = np.asarray(Image.open(subimage_dir + subimage_filepath))
        subimage_list.append(subimage_data)

    # stitch
    erp_image = subimage.erp_ico_stitch(subimage_list, erp_image_height, padding_size)

    # output ERP image
    Image.fromarray(erp_image.astype(np.uint8)).save(erp_stitched_image_filepath)


def test_ico_proj_disparity(erp_depth_filepath, padding_size, subimage_size, subimage_disp_filepath_exp):
    # load erp image
    erp_depthmap_data = depthmap_utils.read_dpt(erp_depth_filepath)
    # erp_dispmap_data = depthmap_utils.depth2disparity(erp_depthmap_data)
    # erp depth map point cloud to obj
    # erp_pc = depthmap_utils.erp_depthmap2pointcloud(erp_depthmap_data)
    # depthmap_utils.pointcloud_save_obj(erp_pc, erp_depth_filepath + ".obj")

    # project to subimage
    subimage_list, _, _ = subimage.erp_ico_proj(erp_depthmap_data, padding_size, subimage_size, corr_downsample_factor=1.0)

    # subimage to point cloud and with rotation
    subimage_list, _, subimage_3dpoints_list_gnom_coord = proj_ico.erp2ico_image(erp_depthmap_data, subimage_size, padding_size, full_face_image=True)
    subimage_3dpoints_list = subimage_3dpoints_list_gnom_coord[0]
    for index in range(0, len(subimage_3dpoints_list)):
        # rotate subimage point cloud to world space
        ico_param = proj_ico.get_icosahedron_parameters(index, padding_size)
        tangent_point_sph = ico_param["tangent_point"]
        subimage_3dpoints_data = depthmap_utils.pointcloud_tang2world(subimage_3dpoints_list[index].reshape(-1, 3).T, tangent_point_sph)

        # output to obj
        if index% 4 ==0:
            print("Output tangent point cloud {}".format(index))
        subimage_pc_filepath = subimage_disp_filepath_exp.format(index) + ".obj"
        depthmap_utils.pointcloud_save_obj(subimage_3dpoints_data, subimage_pc_filepath)

    # visualize to array
    array_image_filepath = subimage_disp_filepath_exp.format(999) + "_vis.jpg"
    depthmap_utils.depth_ico_visual_save(subimage_list, array_image_filepath, subimage_idx_list=None)

    # output to files
    for index in range(0, len(subimage_list)):
        subimage_filepath = subimage_disp_filepath_exp.format(index)
        depthmap_utils.write_pfm(subimage_filepath, subimage_list[index].astype(np.float32))
        depthmap_utils.depth_visual_save(subimage_list[index], subimage_filepath + "_vis.jpg")


def test_ico_stitch_disparity(erp_stitched_depth_filepath, subimage_dispmap_filepath_expression,
                              padding_size, erp_image_height):
    # 0) load disparity maps
    subimage_dispmap_list = []
    for index in range(0, 20):
        subimage_dispmap_filepath = subimage_dispmap_filepath_expression.format(index)
        subimage_dispmap_list.append(depthmap_utils.read_pfm(subimage_dispmap_filepath)[0])

    # stitch
    erp_dispmap = subimage.erp_ico_stitch(subimage_dispmap_list, erp_image_height, padding_size)

    # output ERP image
    depthmap_utils.write_pfm(erp_stitched_depth_filepath, erp_dispmap.astype(np.float32))
    depthmap_utils.depth_visual_save(erp_dispmap, erp_stitched_depth_filepath + "_vis.jpg")


def test_tangent_image_resolution():
    """Compute the tangent image size."""
    erp_image_height_list = [480, 960]
    padding_size_list = [0, 0.1, 0.2, 0.3]
    for erp_image_height in erp_image_height_list:
        for padding_size in padding_size_list:
            tangent_image_width, tangent_image_height = \
                subimage.tangent_image_resolution(erp_image_height * 2, padding_size)
            print("Erp image width {}, padding size {} => Tangent image width {}, height {}"
                  .format(erp_image_height * 2, padding_size, tangent_image_width, tangent_image_height))


def test_ico_proj_disparity_MiDaS(erp_stitched_depth_filepath, subimage_rgb_filepath_exp,
                                  subimage_disp_filepath_exp, padding_size, erp_image_height):
    subimage_dispmap_list = []

    for index in range(0, 20):
        # load the rgb subimage
        subimage_rgb_filepath = subimage_rgb_filepath_exp.format(index)
        # subimage_rgb = np.asarray(Image.open(subimage_rgb_filepath))

        # estimate disparity maps for each face
        subimage_dispmap = depthmap_utils.MiDaS_torch_hub_file(subimage_rgb_filepath)
        subimage_dispmap = depthmap_utils.dispmap_normalize(subimage_dispmap)
        subimage_dispmap_list.append(subimage_dispmap)

        # output to subimage dispmap files
        subimage_dispmap_filepath = subimage_disp_filepath_exp.format(index)
        depthmap_utils.write_pfm(subimage_dispmap_filepath, subimage_dispmap.astype(np.float32))
        depthmap_utils.depth_visual_save(subimage_dispmap, subimage_dispmap_filepath + "_vis.jpg")

    # generate and output ERP disparity map
    erp_dispmap = subimage.erp_ico_stitch(subimage_dispmap_list, erp_image_height, padding_size)
    depthmap_utils.write_pfm(erp_stitched_depth_filepath, erp_dispmap.astype(np.float32))
    depthmap_utils.depth_visual_save(erp_dispmap, erp_stitched_depth_filepath + "_vis.jpg")


if __name__ == "__main__":

    # test_tangent_image_resolution()
    # exit()

    data_root = config.TEST_DATA_DIR
    erp_image_dir = data_root + "erp_00/"
    debug_output_dir = data_root + "erp_00/debug/"
    Path(debug_output_dir).mkdir(parents=True, exist_ok=True)

    erp_image_filename = "0001_rgb.jpg"
    erp_depthmap_filename = "0001_depth.dpt"
    erp_image_filepath = erp_image_dir + erp_image_filename
    erp_depth_filepath = erp_image_dir + erp_depthmap_filename
    erp_stitched_image_filepath = erp_image_dir + "0001_rgb_stitch.jpg"
    erp_stitched_depth_filepath = erp_image_dir + "0001_depth_stitch.dpt"

    output_dir_path = erp_image_dir

    filename_base, _ = os.path.splitext(erp_image_filename)
    pixels_corr_json_filepath_expression = filename_base + "_corr_{:03d}_{:03d}.json"
    subimage_param_filepath_expression = filename_base + "_cam_{:03d}.json"
    # subdepthmap_filepath_expression = filename_base + "_disp_{:03d}.pfm"
    subimage_filepath_expression = filename_base + "_rgb_{:03d}.jpg"
    subimage_dispmap_filepath_expression = filename_base + "_disp_{:03d}.pfm"
    alignedepth_filepath_expression = filename_base + "_disp_{:03d}_aligned.pfm"

    padding_size = 0.4  # padding the subimge
    subimage_size = 400  # the subimage is not square
    erp_image_height = 480
    pixel_corr_downsample_ratio= 0.5

    # functions_enable_list = [0,1,2,3,4,5]
    functions_enable_list = [0,1]

    # 0) project RGB
    if 0 in functions_enable_list:
        debug = False  # warp the subimage the check the pixel correspoding
        intermediate_file = True  # save the pixel corr etc. intermedia information to file
        test_ico_proj_rgb(erp_image_filepath, padding_size, subimage_size,
                          debug_output_dir, filename_base, debug, intermediate_file, corr_subsample_factor=pixel_corr_downsample_ratio)

    # 1) stitch RGB
    if 1 in functions_enable_list:
        test_ico_stitch_rgb(debug_output_dir, erp_stitched_image_filepath, subimage_filepath_expression,
                            padding_size, erp_image_height, debug_output_dir)

    # 2) project disparity map, use ground truth dispmap
    if 2 in functions_enable_list:
        test_ico_proj_disparity(erp_depth_filepath, padding_size, subimage_size, debug_output_dir + subimage_dispmap_filepath_expression)

    # 3) use MiDaS to estimate dispmap from RGB sub-images
    if 3 in functions_enable_list:
        test_ico_proj_disparity_MiDaS(erp_stitched_depth_filepath, debug_output_dir + subimage_filepath_expression,
                                      erp_image_dir + subimage_dispmap_filepath_expression, padding_size, erp_image_height)

    # 4) use the dispmap align python modual to aligned the disparity map
    if 4 in functions_enable_list:
        pass

    # 5) stitch disparity
    if 5 in functions_enable_list:
        test_ico_stitch_disparity(erp_stitched_depth_filepath, debug_output_dir + subimage_dispmap_filepath_expression,
                                  padding_size, erp_image_height)

    # 6) Blending
    if 6 in functions_enable_list:
        blending.ico_blending(erp_image_dir + alignedepth_filepath_expression,
                              debug_output_dir + subimage_param_filepath_expression,
                              erp_image_dir + erp_depthmap_filename, padding_size, erp_image_height)

    # 7) test the camera parameters
    if 7 in functions_enable_list:
        test_erp_ico_cam_intrparams(erp_image_dir + subimage_filepath_expression, debug_output_dir, padding_size)
