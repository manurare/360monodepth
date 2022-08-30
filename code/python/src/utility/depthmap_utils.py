import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import pyramid_gaussian

from PIL import Image
import numpy as np

from struct import unpack
import os
import sys
import re
import gc

import fs_utility
from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def fill_ico_subimage(depth_data_list_, subimage_idx_list):
    """ replace missed subimage with zero matrix.
    """
    depth_data_list = [np.zeros_like(depth_data_list_[0])] * 20
    for subimage_index in range(len(subimage_idx_list)):
        subimage_face_idx = subimage_idx_list[subimage_index]
        depth_data_list[subimage_face_idx] = depth_data_list_[subimage_index]
    return depth_data_list


def depth_ico_visual_save(depth_data_list_, output_path, subimage_idx_list=None):
    """save the visualized depth map array to image file with value-bar.

    :param dapthe_data: The depth data.
    :type dapthe_data: numpy 
    :param output_path: the absolute path of output image.
    :type output_path: str
    :param subimage_idx_list: available subimages index list.
    :type subimage_idx_list: list
    """
    # get vmin and vmax
    # for dispmap in depth_data_list:
    #     if vmin_ > np.amin(dispmap):
    #         vmin_ = np.amin(dispmap)
    #     if vmax_ < np.amax(dispmap):
    #         vmax_ = np.amax(dispmap)
    vmin_ = 0
    vmax_ = 0
    dispmap_array = np.concatenate(depth_data_list_).flatten()
    vmin_idx = int(dispmap_array.size * 0.05)
    vmax_idx = int(dispmap_array.size * 0.95)
    vmin_ = np.partition(dispmap_array, vmin_idx)[vmin_idx]
    vmax_ = np.partition(dispmap_array, vmax_idx)[vmax_idx]

    # add blank image to miss subimage to fill the sub-image array
    depth_data_list = None
    if len(depth_data_list_) != 20 \
            and subimage_idx_list is not None \
            and len(depth_data_list_) == len(subimage_idx_list):
        log.debug("The ico's sub-image size is {}, fill blank sub-images.".format(len(depth_data_list_)))
        # depth_data_list = [np.zeros_like(depth_data_list_[0])] * 20
        # for subimage_index in range(len(subimage_idx_list)):
        #     subimage_face_idx = subimage_idx_list[subimage_index]
        #     depth_data_list[subimage_face_idx] = depth_data_list_[subimage_index]
        depth_data_list = fill_ico_subimage(depth_data_list_, subimage_idx_list)
    elif len(depth_data_list_) == 20:
        depth_data_list = depth_data_list_
    else:
        raise log.error("The sub-image is not completed.")

    # draw image
    figure, axes = plt.subplots(4, 5)
    counter = 0
    for row_index in range(0, 4):
        for col_index in range(0, 5):
            axes[row_index, col_index].get_xaxis().set_visible(False)
            axes[row_index, col_index].get_yaxis().set_visible(False)
            # add sub caption
            axes[row_index, col_index].set_title(str(counter))
            counter = counter + 1
            #
            dispmap_index = row_index * 5 + col_index
            im = axes[row_index, col_index].imshow(depth_data_list[dispmap_index],
                                                   cmap=cm.jet, vmin=vmin_, vmax=vmax_)

    figure.tight_layout()
    plt.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(output_path, dpi=150)
    # plt.show()
    plt.close(figure)

def depth_visual_save(depth_data, output_path, overwrite=True):
    """save the visualized depth map to image file with value-bar.

    :param dapthe_data: The depth data.
    :type dapthe_data: numpy 
    :param output_path: the absolute path of output image.
    :type output_path: str
    """
    depth_data_temp = depth_data.astype(np.float64)

    if fs_utility.exist(output_path, 1) and overwrite:
        log.warn("{} exist.".format(output_path))
        fs_utility.file_rm(output_path)

    # draw image
    fig = plt.figure()
    plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1, wspace=None, hspace=None)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    im = ax.imshow(depth_data_temp, cmap="turbo")
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.savefig(output_path, dpi=150)
    # plt.imsave(output_path, dapthe_data_temp, cmap="turbo")
    plt.close(fig)


def depth_visual(depth_data):
    """
    visualize the depth map
    """
    min = np.min(depth_data)
    max = np.max(depth_data)
    norm = mpl.colors.Normalize(vmin=min, vmax=max)
    cmap = plt.get_cmap('jet')

    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return (m.to_rgba(depth_data)[:, :, :3] * 255).astype(np.uint8)


def rgb2dispmap(image_filepath, pytorch_hub=True):
    """
    Estimate dispmap from rgb image.

    :param image_filepath: the rgb image filepath
    :type image_filepath: str
    :param pytorch_hub: which module should use, defaults to True
    :type pytorch_hub: bool, optional
    :return: MiDaS estimated dispmap
    :rtype: numpy
    """
    depthmap_data = None
    if pytorch_hub:
        log.debug("use PyTorch Hub MiDaS.")
        depthmap_data = MiDaS_torch_hub_file(image_filepath)
    else:
        log.debug("use local MiDaS.")
        # add local MiDas to python path
        dir_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(dir_scripts, "../../MiDaS/"))
        from MiDaS import MiDaS_utils
        from MiDaS.monodepth_net import MonoDepthNet
        from MiDaS.run import run_depth

        image_data = np.asarray(Image.open(image_filepath)[..., :3])
        image_data = image_data[np.newaxis, :, :, [2, 0, 1]]
        MiDaS_module_filepath = dir_scripts + '../../MiDas/model.pt'
        if os.path.exists(MiDaS_module_filepath):
            log.error("MiDaS local module {} does not exist.".format(MiDaS_module_filepath))

        depthmap_data = run_depth(image_data, MiDaS_module_filepath, MonoDepthNet, MiDaS_utils)[0]

    return depthmap_data


def run_persp_monodepth(rgb_image_data_list, persp_monodepth, use_large_model=True):
    if (persp_monodepth == "midas2") or (persp_monodepth == "midas3"):
        MiDaS_torch_hub_data(rgb_image_data_list, persp_monodepth, use_large_model=use_large_model)
    if persp_monodepth == "boost":
        boosting_monodepth(rgb_image_data_list)


def MiDaS_torch_hub_data(rgb_image_data_list, persp_monodepth, use_large_model=True):
    """Estimation the single RGB image's depth with MiDaS downloading from Torch Hub.
    reference: https://pytorch.org/hub/intelisl_midas_v2/

    :param rgb_image_path: the RGB image file path.
    :type rgb_image_path: str
    :param use_large_model: the MiDaS model type.
    :type use_large_model: bool, optional
    """
    import torch

    # 1)initial PyTorch run-time environment
    if use_large_model:
        if persp_monodepth == "midas2":
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        if persp_monodepth == "midas3":
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    else:
        if persp_monodepth == "midas2":
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        if persp_monodepth == "midas3":
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if use_large_model:
        transform = midas_transforms.default_transform
    else:
        transform = midas_transforms.small_transform

    disparity_map_list = []

    for index in range(0, len(rgb_image_data_list)):
        img = rgb_image_data_list[index]
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        disparity_map_list.append(output)
        del output
        del input_batch
        del prediction
        torch.cuda.empty_cache()

        if index % 10 ==0:
            log.debug("MiDaS estimate {} rgb image's disparity map.".format(index))

    del midas
    gc.collect()
    torch.cuda.empty_cache()
    return disparity_map_list



def MiDaS_torch_hub_file(rgb_image_path, use_large_model=True):
    """Estimation the single RGB image's depth with MiDaS downloading from Torch Hub.
    reference: https://pytorch.org/hub/intelisl_midas_v2/

    :param rgb_image_path: the RGB image file path.
    :type rgb_image_path: str
    :param use_large_model: the MiDaS model type.
    :type use_large_model: bool, optional
    """
    import cv2
    import torch
    # import urllib.request

    # import matplotlib.pyplot as plt

    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # urllib.request.urlretrieve(url, filename)
    # use_large_model = True

    if use_large_model:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    else:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if use_large_model:
        transform = midas_transforms.default_transform
    else:
        transform = midas_transforms.small_transform

    img = cv2.imread(rgb_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    # plt.imshow(output)
    # plt.show()
    return output


def read_dpt(dpt_file_path):
    """read depth map from *.dpt file.

    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data


def read_exr(exp_file_path):
    """Read depth map from EXR file

    :param exp_file_path: file path
    :type exp_file_path: str
    :return: depth map data
    :rtype: numpy 
    """
    import array
    import OpenEXR
    import Imath

    # Open the input file
    file = OpenEXR.InputFile(exp_file_path)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R, G, B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")]
    # R,G,B channel is same
    R_np = np.array(R).reshape((sz[1], sz[0]))

    return R_np



def read_pfm(path):
    """Read pfm file.

    :param path: the PFM file's path.
    :type path: str
    :return: the depth map array and scaler of depth
    :rtype: tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            log.error("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            log.error("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write depth data to pfm file.

    :param path: pfm file path
    :type path: str
    :param image: depth data
    :type image: numpy
    :param scale: Scale, defaults to 1
    :type scale: int, optional
    """
    if image.dtype.name != "float32":
        #raise Exception("Image dtype must be float32.")
        log.warn("The depth map data is {}, convert to float32 and save to pfm format.".format(image.dtype.name))
        image_ = image.astype(np.float32)
    else :
        image_ = image

    image_ = np.flipud(image_)

    color = None
    if len(image_.shape) == 3 and image_.shape[2] == 3:  # color image
        color = True
    elif len(image_.shape) == 2 or len(image_.shape) == 3 and image_.shape[2] == 1:  # greyscale
        color = False
    else:
        log.error("Image must have H x W x 3, H x W x 1 or H x W dimensions.")
        # raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    with open(path, "wb") as file:
        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image_.shape[1], image_.shape[0]))
        endian = image_.dtype.byteorder
        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale
        file.write("%f\n".encode() % scale)
        image_.tofile(file)


def depthmap_histogram(data):
    """
    Visualize the pixel value distribution.
    """
    data_max = int(np.max(data))
    data_min = int(np.min(data) + 0.5)
    big_number = 40 #int((data_max - data_min + 1) / 10000)
    print("depth map max is {}, min is {}, bin nubmer is {}".format(np.max(data), np.min(data), big_number))
    bin_range = np.linspace(data_min, data_max+1, big_number)
    histogram, bin_edges = np.histogram(data, bin_range, range=(np.min(data), np.max(data)))
    # configure and draw the histogram figure
    plt.figure()
    plt.title("Depth map data distribution.")
    plt.xlabel("depth value")
    plt.ylabel("pixels")
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()


def depth2disparity(depth_map, baseline=1.0, focal=1.0):
    """
    Convert the depth map to disparity map.

    :param depth_map: depth map data
    :type depth_map: numpy
    :param baseline: [description], defaults to 1
    :type baseline: float, optional
    :param focal: [description], defaults to 1
    :type focal: float, optional
    :return: disparity map data, 
    :rtype: numpy
    """
    no_zeros_index = np.where(depth_map != 0)
    disparity_map = np.full(depth_map.shape, np.Inf, np.float64)
    disparity_map[no_zeros_index] = (baseline * focal) / depth_map[no_zeros_index]
    return disparity_map


def disparity2depth(disparity_map,  baseline=1.0, focal=1.0):
    """Convert disparity value to depth value.
    """
    no_zeros_index = np.where(disparity_map != 0)
    depth_map = np.full(disparity_map.shape, np.Inf, np.float64)
    depth_map[no_zeros_index] = (baseline * focal) / disparity_map[no_zeros_index]
    return depth_map


def dispmap_normalize(dispmap, method = "", mask = None):
    """Normalize a disparity map.

    TODO support mask

    :param dispmap: the original disparity map.
    :type dispmap: numpy
    :param method: the normalization method's name.
    :type method: str
    :param mask: The mask map, available pixel is 1, invalid is 0.
    :type mask: numpy
    :return: normalized disparity map.
    :rtype: numpy
    """
    if mask is None:
        mask = np.ones_like(dispmap, dtype= np.bool)

    dispmap_norm = None
    if method == "naive":
        dispmap_mean = np.mean(dispmap)
        dispmap_norm = dispmap / dispmap_mean
    elif method == "midas":
        median_dispmap = np.median(dispmap[mask])
        dev_dispmap = np.sum(np.abs(dispmap[mask] - median_dispmap)) / np.sum(mask)
        dispmap_norm = np.full(dispmap.shape, np.nan, dtype= np.float64)
        dispmap_norm[mask] = (dispmap[mask] - median_dispmap) / dev_dispmap
    elif method == "range01":
        max_index = np.argsort(dispmap, axis=None)[int(dispmap.size * 0.96)]
        min_index = np.argsort(dispmap, axis=None)[int(dispmap.size * 0.04)]
        max = dispmap.flatten()[max_index]
        min = dispmap.flatten()[min_index]
        dispmap_norm = (dispmap - min) / (max-min)
    else:
        log.error("Normalize methoder {} do not supprot".format(method))
    return dispmap_norm


def subdepthmap_erp2tang(subimage_depthmap_erp, gnomonic_coord_xy):
    """ Covert the subimage's depth map from erp to tangent space.

    :param subimage_depthmap: the subimage's depth map in perspective projection, [height, width].
    :param gnomonic_coord: The tangent image each pixels location in gnomonic space, [height, width] * 2.
    """
    gnomonic_coord_x = gnomonic_coord_xy[0]
    gnomonic_coord_y = gnomonic_coord_xy[1]

    # convert the ERP depth map value to tangent image coordinate depth value
    center2pixel_length = np.sqrt(np.square(gnomonic_coord_x) + np.square(gnomonic_coord_y) + np.ones_like(gnomonic_coord_y))
    subimage_depthmap_persp = np.divide(subimage_depthmap_erp, center2pixel_length)
    return subimage_depthmap_persp


def subdepthmap_tang2erp(subimage_depthmap_persp, gnomonic_coord_xy):
    """ Convert the depth map from perspective to ERP space.

    :param subimage_erp_depthmap: subimage's depth map of ERP space.
    :type subimage_erp_depthmap: numpy 
    :param gnomonic_coord_xy: The tangent image's pixels gnomonic coordinate, x and y.
    :type gnomonic_coord_xy: list
    """
    gnomonic_coord_x = gnomonic_coord_xy[0]
    gnomonic_coord_y = gnomonic_coord_xy[1]
    center2pixel_length = np.sqrt(np.square(gnomonic_coord_x) + np.square(gnomonic_coord_y) + np.ones_like(gnomonic_coord_y))
    subimage_depthmap_erp = subimage_depthmap_persp * center2pixel_length
    return subimage_depthmap_erp


def depthmap_pyramid(depthmap_list, pyramid_layer_number, pyramid_downscale):
    """ Create the all depth maps pyramid.

    :param depthmap_list: The list of depth map
    :type depthmap_list: list
    :param pyramid_layer_number: the pyramid level number
    :type pyramid_layer_number: int
    :param pyramid_downscale: pyramid downsample ration, coarse_level_size = fine_level_size * pyramid_downscale
    :type pyramid_downscale: float
    :return: the pyramid for each depth map. the 1st index is pyramid level, 2nd is image index, [pyramid_idx][image_idx], 1st (index 0) level is coarsest image.
    :rtype: list
    """    
    depthmap_number = len(depthmap_list)
    depthmap_pryamid = [[0] * depthmap_number for i in range(pyramid_layer_number)]
    for index in range(0, depthmap_number):
        if pyramid_layer_number == 1:
            depthmap_pryamid[0][index] = depthmap_list[index].astype(np.float64)
        else:
            depthmap = depthmap_list[index]
            pyramid = tuple(pyramid_gaussian(depthmap, max_layer=pyramid_layer_number - 1, downscale=pyramid_downscale, multichannel=False))
            for layer_index in range(0, pyramid_layer_number):
                depthmap_pryamid[pyramid_layer_number - layer_index - 1][index] = pyramid[layer_index].astype(np.float64)

    return depthmap_pryamid