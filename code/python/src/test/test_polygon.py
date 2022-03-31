from numpy.lib.polynomial import poly
import configuration

from utility import polygon
from utility import gnomonic_projection as gp

from utility.logger import Logger
import numpy as np

log = Logger(__name__)
log.logger.propagate = False


def test_generate_hexagon():
    vertex_list = polygon.generate_hexagon(circumradius=40, draw_enable=True)
    print(vertex_list)


def test_inside_hexagon():
    circumradius = 40
    hexagon_points_list = polygon.generate_hexagon(circumradius)

    test_time = 10
    for idx in range(test_time):
        # point_list = np.array([[20,30]],dtype = float)
        mu, sigma = 0, circumradius * 1.0  # mean and standard deviation
        point_list = np.random.normal(mu, sigma, (1, 2))
        from PIL import Image, ImageDraw
        image_width = circumradius * 3
        image_height = circumradius * 3
        offset_width = image_width * 0.5
        offset_height = image_height * 0.5
        image = Image.new('RGB', (image_height, image_width), 'white')
        draw = ImageDraw.Draw(image)
        hexagon_points_list_ = np.zeros_like(hexagon_points_list)
        hexagon_points_list_[:, 0] = hexagon_points_list[:, 0] + offset_width  # the origin at upper-left of image.
        hexagon_points_list_[:, 1] = hexagon_points_list[:, 1] + offset_height
        draw.polygon(tuple(map(tuple, hexagon_points_list_)), outline='black', fill='red')
        point_list_ = point_list + np.array(((offset_width, offset_height)), dtype=np.double)
        draw.point((point_list_[0][0], point_list_[0][1]), fill='green')
        image.show()

        inside_result = gp.inside_polygon_2d(point_list, hexagon_points_list)
        print("{} - {} - {}".format(idx, point_list_[0], inside_result[0]))


def test_isect_line_plane_3D():
    # line_point_0 = np.array([0,0,0], dtype = np.float32)
    # line_point_1 = np.array([0,0,1], dtype = np.float32)
    # plane_point = np.array([0,0,3], dtype = np.float32)
    # plane_norm = np.array([0,0,1], dtype = np.float32)

    line_point_0 = np.array([0, 0, 0], dtype=np.float32)
    line_point_1 = np.array([1, 1, 0], dtype=np.float32)
    plane_point = np.array([5, 5, 5], dtype=np.float32)
    plane_norm = np.array([1, 1, 1], dtype=np.float32)
    isect_point_gt = np.array([7.5, 7.5, 0], dtype=np.float32)

    isect_point = polygon.isect_line_plane_3D(line_point_0, line_point_1, plane_point, plane_norm)
    print(isect_point)
    print("GT: {}".format(isect_point_gt))


def test_triangle_bounding_rectangle_3D():

    # headpoint = np.array([0, 1, 0], dtype=np.float32)
    # edgepoints = np.array([[-1, -1, 0], [1, -1, 0]], dtype=np.float32)

    headpoint = np.array([0, 1, 5], dtype=np.float32)
    edgepoints = np.array([[-1, -1, 2], [1, -1, 2]], dtype=np.float32)

    vertices = polygon.triangle_bounding_rectangle_3D(headpoint, edgepoints)
    print(vertices)


if __name__ == "__main__":
    # test_generate_hexagon()
    # test_inside_hexagon()
    # test_isect_line_plane_3D()
    test_triangle_bounding_rectangle_3D()
