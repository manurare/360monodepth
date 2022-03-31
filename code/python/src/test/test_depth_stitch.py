import configuration as config
from utility import cam_models, data_mocker
from utility import depth_stitch
from utility import serialization
from utility import depthmap_utils
from utility import blending
from utility import image_io
from utility import subimage

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import os
import json

"""
Stitch perspective images:
- Input: RGB-D images, camera parameters for each image;
- Output: fish-eye depth map
"""

# TODO simplifying this file, move functions to utility

def test_depthmap_deform():
    # depthmap = np.mgrid[0:15, 0:20].astype(np.float64)
    # depthmap = depthmap[0] * depthmap[1]
    depthmap = np.ones([15, 20]).astype(np.float64)
    image_io.image_show(depthmap)

    scale_x = np.linspace(0.0, 1.0, num=5)
    scale_y = np.linspace(0.0, 1.0, num=4)
    scale_array = np.meshgrid(scale_x, scale_y)
    scale_array = scale_array[0] * scale_array[1]

    # offset_array = np.mgrid[0:4, 0:5]
    # offset_array = offset_array[0] * offset_array[1]
    # offset_array = np.zeros([4,5])
    offset_array = np.ones([4, 5])

    depthmap_new = depth_stitch.depthmap_deform(depthmap, scale_array, offset_array)
    image_io.image_show(depthmap_new)


def test_run_midas_batch(rgb_image_filename_expression, depth_filename_expression, file_number):
    """TODO need test"""
    # load rgb image
    print("load rgb images")
    rgb_subimage_list = []
    for index in range(0, file_number):
        image_data = np.asarray(Image.open(rgb_image_filename_expression.format(index)))[..., :3]
        rgb_subimage_list.append(image_data)

    # estimate disparity map
    print("estimate disparity map")
    depthmap_data_list = depthmap_utils.MiDaS_torch_hub_data(rgb_subimage_list)

    # output disparity map and visualized result.
    print("output disparity map and visualized")
    for index in range(0, file_number):
        depth_filename = depth_filename_expression.format(index)
        depthmap_utils.write_pfm(depth_filename, depthmap_data_list[index], scale=1)
        depthmap_utils.depth_visual_save(depthmap_data_list[index], depth_filename + ".jpg")

        if index % 4 == 0:
            print("output depth map to {}".format(depth_filename))


def test_run_midas(image_filename_expression, depth_filename_expression, file_number, pytorch_hub=True):
    for index in range(0, file_number):
        depthmap_data = None
        if pytorch_hub:
            print("use PyTorch Hub MiDaS.")
            depthmap_data = depthmap_utils.MiDaS_torch_hub_file(image_filename_expression.format(index))
        else:
            image_data = np.asarray(Image.open(image_filename_expression.format(index)))[..., :3]
            image_data = image_data[np.newaxis, :, :, [2, 0, 1]]
            print("use local MiDaS.")
            from MiDaS import MiDaS_utils
            from MiDaS.monodepth_net import MonoDepthNet
            from MiDaS.run import run_depth
            depthmap_data = run_depth(image_data, '../../MiDas/model.pt', MonoDepthNet, depthmap_utils)[0]

        depth_filename = depth_filename_expression.format(index)
        depthmap_utils.write_pfm(depth_filename, depthmap_data, scale=1)
        depthmap_utils.depth_visual_save(depthmap_data, depth_filename + ".jpg")

        if index % 4 == 0:
            print("output depth map to {}".format(depth_filename))


def test_world2cam_slow(img, model):
    W = 500
    H = 500
    z = W/3.0
    x = [i-W/2 for i in range(W)]
    y = [j-H/2 for j in range(H)]
    x_grid, y_grid = np.meshgrid(x, y, sparse=False, indexing='xy')
    point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)]).reshape(3, -1)
    point2d = cam_models.world2cam_slow(point3D, model).T
    mapx = point2d[:, 0]
    mapy = point2d[:, 1]
    mapx = mapx.reshape(H, W).astype(np.float32)
    mapy = mapy.reshape(H, W).astype(np.float32)
    out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR).astype(np.uint8)
    plt.imshow(out)
    plt.show()


def test_erp2fisheye(fisheye_image_path, cam_params_json_file):
    """Fisheye --> ERP --> Fisheye

    Data in OpenCV cs # x:right direction, y:down direction, z:front direction
    """
    # load data and parameters
    camera_params_list = json.load(open(cam_params_json_file, 'r'))  # Calibration file from Elliot
    model = camera_params_list['cameras'][0]

    if not os.path.exists(fisheye_image_path):
        print("{} do not exist.".format(fisheye_image_path))

    img = np.asarray(Image.open(fisheye_image_path))[:, :, :3]

    fisheye_width = img.shape[1]
    fisheye_height = img.shape[0]
    erp_width = 800
    erp_height = 400

    # 0) fisheye 2 ERP image
    phi = [-np.pi + (i+0.5) * (2*np.pi/erp_width) for i in range(erp_width)]
    theta = [-np.pi/2 + (i+0.5) * (np.pi/erp_height) for i in range(erp_height)]
    phi_xy, theta_xy = np.meshgrid(phi, theta, sparse=False, indexing='xy')
    point3D = np.stack([np.cos(theta_xy) * np.sin(phi_xy), np.sin(theta_xy), np.cos(theta_xy) * np.cos(phi_xy)]).reshape(3, -1)
    point2d = cam_models.world2cam_slow(point3D, model).T

    mapx = point2d[:, 0].astype(np.float32)
    mapy = point2d[:, 1].astype(np.float32)
    mapx = mapx.reshape(erp_height, erp_width).astype(np.float32)
    mapy = mapy.reshape(erp_height, erp_width).astype(np.float32)
    mapx = np.where(mapx >= 0.0, mapx, 0.0)
    mapy = np.where(mapy >= 0.0, mapy, 0.0)
    out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR).astype(np.uint8)

    plt.imshow(out)
    plt.show()

    # 1) ERP to fisheye
    x_list = np.linspace(0, fisheye_width, fisheye_width, endpoint=False)
    y_list = np.linspace(0, fisheye_height, fisheye_height, endpoint=False)
    mapx, mapy = np.meshgrid(x_list, y_list)
    fisheye_2d_points = np.stack((mapx.ravel(), mapy.ravel()), axis=1).T
    fisheye_3d_points = cam_models.cam2world(fisheye_2d_points.T, model)  # is [y,x,z]

    # 3d to sph
    radius = np.linalg.norm(fisheye_3d_points, axis=1)
    valid_list = radius > 0.0001  # set the 0 radius to origin.

    azimuth = np.zeros((fisheye_3d_points.shape[0]), np.float)
    polar = np.zeros((fisheye_3d_points.shape[0]), np.float)

    azimuth[valid_list] = np.arctan2(fisheye_3d_points[:, 0][valid_list], fisheye_3d_points[:, 2][valid_list])
    polar[valid_list] = np.arcsin(np.divide(-fisheye_3d_points[:, 1][valid_list], radius[valid_list]))

    # sph to pixels
    x = (azimuth + np.pi) / (2.0 * np.pi) * (2 * erp_height)
    y = -(polar - 0.5 * np.pi) / np.pi * erp_height
    y = y.reshape(fisheye_height, fisheye_width).astype(np.float32)
    x = x.reshape(fisheye_height, fisheye_width).astype(np.float32)

    result = cv2.remap(out, x, y, cv2.INTER_LINEAR)

    plt.imshow(result.astype(np.uint8))
    plt.show()


def test_perspective2fisheye_depth(depth_filename_expression, cam_params_filename_expression, file_number, fisheye_image_params_json_file, fisheye_image_stitch_path):
    depth_data_list = []
    image_param_list = []
    # load image & camera parameters
    for index in range(0, file_number):
        # image_data = np.asarray(Image.open(depth_filename_expression.format(index)))[:, :, :3]
        depth_data, scale = depthmap_utils.read_pfm(depth_filename_expression.format(index))
        depth_data /= scale
        depth_data_list.append(depth_data)
        cam_params_filepath = cam_params_filename_expression.format(index)
        cam_params = serialization.cam_param_json2dict(cam_params_filepath)
        image_param_list.append(cam_params)

        if index % 5 == 0:
            print("loading pinhole depth map file: {}".format(index))
            print("camera image : {} \n parameter file: {}".format(depth_filename_expression.format(index), cam_params_filepath))

    # load fisheye camera parameters
    print("beging to stitch fisheye depth map .....")
    fisheye_params_list = json.load(open(fisheye_image_params_json_file, 'r'))  # Calibration file from Elliot
    fisheye_params = fisheye_params_list['cameras'][0]

    # stitch rgb image
    # fisheye_image = depth_stitch.stitch_depth_subimage(depth_data_list, image_param_list, fisheye_params)
    # fisheye_image = Image.fromarray(fisheye_image)
    # fisheye_image.save(fisheye_image_stitch_path)
    fisheye_image = fisheye_image.astype(np.float32)
    depthmap_utils.write_pfm(fisheye_image_stitch_path, fisheye_image, scale=1)
    depthmap_utils.depth_visual_save(fisheye_image, fisheye_image_stitch_path + ".jpg")


def test_depthmap2mesh(depthmap_filepath, rgb_filepath, output_path):
    # 0) load the depth / color images
    depth_map, _ = depthmap_utils.read_pfm(depthmap_filepath)
    depth_map = 1.0 / depth_map
    rgb_image = np.asarray(Image.open(rgb_filepath))[:, :, :3]

    f = 300.0
    cx = depth_map.shape[1] / 2.0
    cy = depth_map.shape[0] / 2.0
    cam_int_param = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], np.float)

    #  convert the depth map to 3D mesh and output
    texture_filepath = os.path.basename(rgb_filepath)
    depth_stitch.depthmap2mesh(depth_map, rgb_image, cam_int_param, output_path, rgb_image_path=texture_filepath)


def test_find_corresponding(pixels_corresponding_json_filepath,
                            image_filename_expression, cam_params_filename_expression,
                            file_number, fisheye_image_params_json_file):
    """
    output the pixels relationship to json file.
    """
    image_data_list = []
    image_param_list = []

    # load perspective image & camera parameters
    for index in range(0, file_number):
        image_data = np.asarray(Image.open(image_filename_expression.format(index)))[:, :, :3]
        image_data_list.append(image_data)
        cam_params_filepath = cam_params_filename_expression.format(index)
        cam_params = serialization.cam_param_json2dict(cam_params_filepath)
        image_param_list.append(cam_params)

        if index % 5 == 0:
            print("loading pinhole image {}".format(index))
            print("camera image : {} \n parameter file: {}".format(image_filename_expression.format(index), cam_params_filepath))

    # load fisheye camera parameters
    print("beging to computing the pixels corresponding .....")
    fisheye_params_list = json.load(open(fisheye_image_params_json_file, 'r'))  # Calibration file from Elliot
    fisheye_params = fisheye_params_list['cameras'][0]

    # get the pixels corresponding relationship
    for src_index in range(0, file_number):
        for tar_index in range(0, file_number):
            if src_index == tar_index:
                continue

            pixels_corresponding = depth_stitch.find_corresponding(
                image_data_list[src_index], image_param_list[src_index],
                image_data_list[tar_index], image_param_list[tar_index],
                fisheye_params)

            # # plot corresponding & output
            # image1_output_path = image_filename_expression.format(src_index) + "_features.jpg"
            # image2_output_path = image_filename_expression.format(tar_index)  + "_features.jpg"
            # draw_corresponding(image_data_list[src_index], image_data_list[tar_index],pixels_corresponding, image1_output_path,image2_output_path)

            # save the corresponding relation to json file
            pixels_corresponding_json_path = pixels_corresponding_json_filepath.format(src_index, tar_index)
            print("save corresponding to : {}".format(pixels_corresponding_json_path))
            serialization.pixel_corresponding_save(pixels_corresponding_json_path,
                                                   image_filename_expression.format(src_index), image_data_list[src_index],
                                                   image_filename_expression.format(tar_index), image_data_list[tar_index],
                                                   pixels_corresponding)


def test_load_corresponding(pixels_corresponding_json_filepath_expression,
                            subimage_dir,
                            pixels_corresponding_src_index, pixels_corresponding_tar_index):
    """Visualize the correspoding pixels.
    """
    pixels_corresponding_json_filepath = pixels_corresponding_json_filepath_expression.format(pixels_corresponding_src_index, pixels_corresponding_tar_index)
    pixels_corresponding = serialization.pixel_corresponding_load(pixels_corresponding_json_filepath)
    print("load the pixels corresponding file: {}".format(pixels_corresponding_json_filepath))

    src_image_filepath = pixels_corresponding["src_image_filename"]
    tar_image_filepath = pixels_corresponding["tar_image_filename"]

    # load subimage
    src_image_data = Image.open(subimage_dir + src_image_filepath)
    tar_image_data = Image.open(subimage_dir + tar_image_filepath)

    # plot the pixels points
    pixel_corresponding_array = pixels_corresponding["pixel_corresponding"]
    image1_output_path = subimage_dir + src_image_filepath + "_features.jpg"
    image2_output_path = subimage_dir + tar_image_filepath + "_features.jpg"
    subimage.draw_corresponding(src_image_data, tar_image_data, pixel_corresponding_array, image1_output_path, image2_output_path)


def test_perspective2fisheye_rgb(image_filename_expression, cam_params_filename_expression, file_number, fisheye_image_params_json_file, fisheye_image_stitch_path):
    """[summary]
    """
    image_data_list = []
    image_param_list = []
    # load image & camera parameters
    for index in range(0, file_number):
        image_data = np.asarray(Image.open(image_filename_expression.format(index)))[:, :, :3]
        image_data_list.append(image_data)
        cam_params_filepath = cam_params_filename_expression.format(index)
        cam_params = serialization.cam_param_json2dict(cam_params_filepath)
        image_param_list.append(cam_params)

        if index % 5 == 0:
            print("loading pinhole image {}".format(index))
            print("camera image : {} \n parameter file: {}".format(image_filename_expression.format(index), cam_params_filepath))

    # load fisheye camera parameters
    print("beging to stitch fisheye image .....")
    fisheye_params_list = json.load(open(fisheye_image_params_json_file, 'r'))  # Calibration file from Elliot
    fisheye_params = fisheye_params_list['cameras'][0]

    # stitch rgb image
    fisheye_image = cam_models.stitch_rgb_image(image_data_list, image_param_list, fisheye_params)
    fisheye_image = Image.fromarray(fisheye_image)
    fisheye_image.save(fisheye_image_stitch_path)


def test_fisheye_image_proj_rgb(fisheye_image_path, cam_params_json_file, subimage_filepath_expression, subimage_param_filepath_expression):
    """[summary]
    """
    camera_params_list = json.load(open(cam_params_json_file, 'r'))  # Calibration file from Elliot
    camera_params = camera_params_list['cameras'][0]

    if not os.path.exists(fisheye_image_path):
        print("{} do not exist.".format(fisheye_image_path))

    img = np.asarray(Image.open(fisheye_image_path))[:, :, :3]
    sub_images, sub_image_param = cam_models.sample_rgb_image(img, camera_params)

    for index in range(0, len(sub_images)):
        sub_image = sub_images[index]

        im = Image.fromarray(sub_image)
        im.save(subimage_filepath_expression.format(index))

        # save camera parameter for each
        camera_param = sub_image_param[index]
        serialization.cam_param_dict2json(camera_param, subimage_param_filepath_expression.format(index))


if __name__ == "__main__":
    data_root = config.TEST_DATA_DIR
    fisheye_image_dir = data_root + "fisheye_00/"
    file_number = 9

    # fisheye_image_params_json_file = data_root + 'calibration.json'  # Calibration file from Elliot
    fisheye_image_params_json_file = fisheye_image_dir + "calib_results_0.json"
    # fisheye_image_params_json_file = data_root + 'brown_00/calibration.json'

    fisheye_image_path = fisheye_image_dir + "img0.jpg"
    # fisheye_image_dir = data_root + "brown_00/"
    # fisheye_image_path = fisheye_image_dir + "video-frame-3840x2880_0.png"
    # fisheye_image_path = data_root + "video-frame-3840x2880.png"

    filename_base, _ = os.path.splitext(fisheye_image_path)
    subimage_filepath_expression = filename_base + "_rgb_{:03d}.jpg"
    subimage_param_filepath_expression = filename_base + "_cam_{:03d}.json"

    test_list = [10]

    # 0) test projection fish-eye image to perspective images.
    if 0 in test_list:
        test_fisheye_image_proj_rgb(fisheye_image_path, fisheye_image_params_json_file, subimage_filepath_expression, subimage_param_filepath_expression)

    # 1) test stitch perspective images to fisheye images.
    if 1 in test_list:
        # fisheye_image_stitch_path = data_root + "video-frame-3840x2880_stitch.png"
        fisheye_image_stitch_path = fisheye_image_dir + "img0_rgb_stitch.png"
        # test_perspective2fisheye_rgb(subimage_filepath_expression, subimage_param_filepath_expression, file_number, fisheye_image_params_json_file, fisheye_image_stitch_path)

    # 2) fisheye-> erp -> fisheye
    if 2 in test_list:
        test_erp2fisheye(fisheye_image_path, fisheye_image_params_json_file)

    # 3) rgb image to depth map
    if 3 in test_list:
        # subdepthmap_filepath_expression = filename_base + "_{:03d}.pfm"
        subdepthmap_filepath_expression = filename_base + "_disp_{:03d}.pfm"
        test_run_midas(subimage_filepath_expression, subdepthmap_filepath_expression, file_number)

    # 4) test stitch depth map
    if 4 in test_list:
        # fisheye_depth_stitch_path = data_root + "video-frame-3840x2880_stitch.pfm"
        # fisheye_depth_stitch_path = fisheye_image_dir + "img0_stitch.pfm"
        fisheye_depth_stitch_path = fisheye_image_dir + "img0_disp_stitch_aligned.pfm"
        # TODO fix the depth maps stitch wrap around
        test_perspective2fisheye_depth(subdepthmap_filepath_expression, subimage_param_filepath_expression, file_number, fisheye_image_params_json_file, fisheye_depth_stitch_path)

    # 5) test project depth map to mesh
    if 5 in test_list:
        depthmap_filepath = data_root + "OsakaTemple6-cubemap-4.pfm"
        rgb_filepath = data_root + "OsakaTemple6-cubemap-4.jpg"
        output_path = data_root + "OsakaTemple6-cubemap-4.obj"
        test_depthmap2mesh(depthmap_filepath, rgb_filepath, output_path)

    # 6) test find image
    if 6 in test_list:
        pixels_corresponding_json_filepath_expression = filename_base + "_corr_{:03d}_{:03d}.json"
        test_find_corresponding(pixels_corresponding_json_filepath_expression,
                                subimage_filepath_expression, subimage_param_filepath_expression,
                                file_number, fisheye_image_params_json_file)

    # 7) test load pixels corresponding
    if 7 in test_list:
        pixels_corresponding_src_index = 4
        pixels_corresponding_tar_index = 3
        test_load_corresponding(pixels_corresponding_json_filepath_expression,
                                fisheye_image_dir,
                                pixels_corresponding_src_index, pixels_corresponding_tar_index)

    # 8) Blending
    if 8 in test_list:
        alignedepth_filepath_expression = filename_base + "_disp_{:03d}_aligned.pfm"
        blending.preprocessing(alignedepth_filepath_expression, subimage_param_filepath_expression, file_number,
                               pixels_corresponding_json_filepath_expression)

    # 9) MiDaS on Cpu memory
    if 9 in test_list:
        erp_image_dir = data_root + "erp_00/"
        erp_image_filename = "0001_rgb.jpg"
        filename_base, _ = os.path.splitext(erp_image_filename)
        subimage_filepath_expression = erp_image_dir + filename_base + "_rgb_{:03d}.jpg"
        subimage_dispmap_filepath_expression = erp_image_dir + filename_base + "_disp_{:03d}.pfm"
        test_run_midas_batch(subimage_filepath_expression,
                             subimage_dispmap_filepath_expression, 20)

    # 10) test depth map deformation
    if 10 in test_list:
        test_depthmap_deform()
