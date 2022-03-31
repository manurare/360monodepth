import configuration as config

from utility import plot_figure

from utility.logger import Logger
import numpy as np

log = Logger(__name__)
log.logger.propagate = False


def test_draw_ico_tangent_planes():
    # draw the figure 2.
    pf = plot_figure.PlotFigure()

    data_output = config.TEST_DATA_DIR

    radius = 1
    padding_size = 0.0
    subimage_shift_ratio = 1.0
    obj_file_path = data_output + "fig2.obj"
    texture_enable = True
    texture_filename = "ico_rgb_src.png"
    pf.draw_ico_tangent_planes(radius, padding_size, obj_file_path, subimage_shift_ratio, texture_enable, texture_filename)


def test_draw_ico_tangent_planes_texture():
    data_root_dir = config.TEST_DATA_DIR
    subimage_filename_exp = "ico_rgb_src_{}.png"
    texture_filename = "ico_rgb_src.png"

    subimage_filename_list = []
    for idx in range(20):
        subimage_filename_list.append(subimage_filename_exp.format(idx))

    pf = plot_figure.PlotFigure()
    pf.draw_ico_tangent_planes_texture_stitch(data_root_dir, subimage_filename_list, texture_filename)


if __name__ == "__main__":
    test_draw_ico_tangent_planes_texture()
    test_draw_ico_tangent_planes()
