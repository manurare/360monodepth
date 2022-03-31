import configuration

from utility import cam_models

def test_generate_virtual_camera_parameters():
    """ Test the virtual camera parameters generation.
    """
    # 0) get the camera orientation
    hfov_fisheye = 180
    vfov_fisheye = 180
    hfov_pinhole = 60
    vfov_pinhole = 60
    horizontal_size = 3
    vertical_size = 3
    xyz_rotation_array = cam_models.generate_camera_orientation(hfov_fisheye, vfov_fisheye, hfov_pinhole, vfov_pinhole, horizontal_size, vertical_size)

    print(xyz_rotation_array)

    # 1) get the camera intrinsic and extrinsic parameters
    image_width = 200
    image_height = 400

    params_list = cam_models.get_perspective_camera_parameters(
        hfov_pinhole, vfov_pinhole, image_width, image_height,  xyz_rotation_array.T)

    print("output the camera model parameters:")
    index = 0
    for params in params_list:
        print("-----------camera {} parameters----------".format(index))
        index += 1
        for key in params:
            value = params[key]
            print("key: {} \n value: {}\n".format(key, value))


if __name__ == "__main__":
    test_generate_virtual_camera_parameters()
