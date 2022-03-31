
import configuration as config

from utility import depthmap_utils
from utility import metrics
from utility import image_io
from utility.logger import Logger

import numpy as np

log = Logger(__name__)
log.logger.propagate = False


def test_delta_inlier_ratio_map(dispmap_predict_filepath, depthmap_gt_filepath, output_filepath):
    """ Test the error and error map.
    """
    # 0) load data
    gt_depthmap = depthmap_utils.read_dpt(depthmap_gt_filepath)
    gt_dispmap = depthmap_utils.depth2disparity(gt_depthmap)

    dispmap_predict, _ = depthmap_utils.read_pfm(dispmap_predict_filepath)

    mask = np.ones_like(gt_dispmap)

    # 1) compute the metric
    for index in range(1, 4):
        delta_data = metrics.delta_inlier_ratio(dispmap_predict, gt_dispmap, mask, index)
        print("dalta_{} error is {}".format(index, delta_data))
        delta_data_map = metrics.delta_inlier_ratio_map(dispmap_predict, gt_dispmap, mask, index)
        depthmap_utils.depth_visual_save(delta_data_map, output_filepath + "error_delta_{}.jpg".format(index))

    abs_rel_error = metrics.abs_rel_error(dispmap_predict, gt_dispmap, mask)
    abs_rel_error_map = metrics.abs_rel_error_map(dispmap_predict, gt_dispmap, mask)
    print("Absolute Relative Difference error is {}".format(abs_rel_error))
    depthmap_utils.depth_visual_save(abs_rel_error_map, output_filepath + "error_abs_rel_error.jpg")

    sq_rel_error = metrics.sq_rel_error(dispmap_predict, gt_dispmap, mask)
    sq_rel_error_map = metrics.sq_rel_error_map(dispmap_predict, gt_dispmap, mask)
    print("Square Relative Difference error is {}".format(sq_rel_error))
    depthmap_utils.depth_visual_save(sq_rel_error_map, output_filepath + "error_sq_rel_error.jpg")

    lin_rms_sq_error = metrics.lin_rms_sq_error(dispmap_predict, gt_dispmap, mask)
    lin_rms_sq_error_map = metrics.lin_rms_sq_error_map(dispmap_predict, gt_dispmap, mask)
    print("RMSE (linear) is {}".format(lin_rms_sq_error))
    depthmap_utils.depth_visual_save(lin_rms_sq_error_map, output_filepath + "error_lin_rms_sq_error.jpg")

    log_rms_sq_error = metrics.log_rms_sq_error(dispmap_predict, gt_dispmap, mask)
    log_rms_sq_error_map = metrics.log_rms_sq_error_map(dispmap_predict, gt_dispmap, mask)
    print("RMSE (log) is {}".format(log_rms_sq_error))
    depthmap_utils.depth_visual_save(log_rms_sq_error_map, output_filepath + "error_log_rms_sq_error.jpg")

    log_rms_scale_invariant = metrics.log_rms_scale_invariant(dispmap_predict, gt_dispmap, mask)
    log_rms_scale_invariant_map = metrics.log_rms_scale_invariant_map(dispmap_predict, gt_dispmap, mask)
    print("RMSE (log scale invariant) is {}".format(log_rms_scale_invariant))
    depthmap_utils.depth_visual_save(log_rms_scale_invariant_map, output_filepath + "error_log_rms_scale_invariant_map.jpg")


if __name__ == "__main__":
    data_root = config.TEST_DATA_DIR + "erp_00/"

    depth_gt_filepath = data_root + "0001_depth.dpt"
    dispmap_filepath = data_root + "0001_dispmap_aligned.pfm"
    output_filepath = data_root + "debug/"

    test_delta_inlier_ratio_map(dispmap_filepath, depth_gt_filepath, output_filepath)
