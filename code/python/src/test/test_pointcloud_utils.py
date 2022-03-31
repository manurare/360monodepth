
import configuration as config

from utility import pointcloud_utils
from utility import depthmap_utils
from utility import image_io
from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def test_depthmap2pointcloud_erp(depth_filepath, rgb_filepath, output_ply_filepath):
    """ Test ERP depth + rgb image to color point cloud.
    """
    # load data from disk
    erp_depthmap = depthmap_utils.read_dpt(depth_filepath)
    erp_rgb = image_io.image_read(rgb_filepath)

    # to point cloud
    pointcloud_utils.depthmap2pointcloud_erp(erp_depthmap, erp_rgb, output_ply_filepath)


if __name__ == "__main__":
    data_root = config.TEST_DATA_DIR + "erp_00/"
    depth_filepath = data_root + "0001_depth.dpt"
    rgb_filepath = data_root + "0001_rgb.jpg"

    output_ply_filepath = data_root + "0001_pointcloud.ply"
    test_depthmap2pointcloud_erp(depth_filepath, rgb_filepath, output_ply_filepath)
