
import configuration as config

from utility import spherical_coordinates as sc
from utility.logger import Logger

log = Logger(__name__)
log.logger.propagate = False

import numpy as np
import unittest

class TestSphericalCoordinates(unittest.TestCase):

    def setUp(self) -> None:
        self.erp_image_height = 512
        self.erp_image_width = self.erp_image_height * 2
        self.erp_points = np.zeros([2, 9], np.float32)
        self.erp_points[:, 0] = [0, 0]
        self.erp_points[:, 1] = [self.erp_image_width - 1, 0]
        self.erp_points[:, 2] = [0, self.erp_image_height-1]
        self.erp_points[:, 3] = [self.erp_image_width - 1, self.erp_image_height-1]
        # center corners
        self.erp_points[:, 4] = [(self.erp_image_width - 1) / 2.0, (self.erp_image_height - 1) / 2.0]
        # four corners
        self.erp_points[:, 5] = [-0.5, -0.5]
        self.erp_points[:, 6] = [self.erp_image_width - 1 + 0.5 , -0.5]
        self.erp_points[:, 7] = [-0.5, self.erp_image_height - 1 + 0.5]
        self.erp_points[:, 8] = [self.erp_image_width - 1 + 0.5 , self.erp_image_height - 1 + 0.5]

        self.sph_points = np.zeros([2, 9], np.float32)
        self.sph_points[:, 0] = [-np.pi + (np.pi * 2 / self.erp_image_width) / 2.0, np.pi / 2.0 - (np.pi / self.erp_image_height) / 2.0]
        self.sph_points[:, 1] = [np.pi - (np.pi * 2 / self.erp_image_width) / 2.0, np.pi / 2.0 - (np.pi / self.erp_image_height) / 2.0]
        self.sph_points[:, 2] = [-np.pi + (np.pi * 2 / self.erp_image_width) / 2.0, -np.pi / 2.0 + (np.pi / self.erp_image_height) / 2.0]
        self.sph_points[:, 3] = [np.pi - (np.pi * 2 / self.erp_image_width) / 2.0, -np.pi / 2.0 + (np.pi / self.erp_image_height) / 2.0]
        # center corners
        self.sph_points[:, 4] = [0, 0]
        # four corners
        self.sph_points[:, 5] = [-np.pi, np.pi/2.0]
        self.sph_points[:, 6] = [-np.pi, np.pi/2.0]
        self.sph_points[:, 7] = [-np.pi, np.pi/2.0]
        self.sph_points[:, 8] = [-np.pi, np.pi/2.0]

        return super().setUp()

    def test_erp2sph(self):
        sph_points_from_erp = sc.erp2sph(self.erp_points, self.erp_image_height, True)
        # print("--:\n{}".format(sph_points_from_erp.T))
        # print("GT:\n{}".format(self.sph_points.T))
        self.assertTrue(np.allclose(sph_points_from_erp, self.sph_points))

    def test_sph2erp(self):
        erp_points_from_sph = sc.sph2erp_0(self.sph_points, self.erp_image_height, True)
        temp = sc.erp_pixel_modulo_0(self.erp_points, self.erp_image_height)
        # print("--:\n{}".format(erp_points_from_sph.T))
        # print("GT:\n{}".format(temp.T))
        self.assertTrue(np.allclose(erp_points_from_sph, temp, atol=1e-4, rtol=1e-4))

    def test_great_circle_distance_uv(self):
        """"test great_circle_distance
        """
        point_pair = []
        point_pair.append([[-np.pi / 2.0, np.pi / 2.0], [-np.pi / 2.0, 0.0]])
        point_pair.append([[-np.pi / 4.0, np.pi / 4.0], [-np.pi / 4.0, -np.pi / 4.0]])
        point_pair.append([[0.0, 0.0], [np.pi / 2.0, 0]])
        point_pair.append([[np.pi / 2.0, 0], [np.pi / 2.0, -np.pi / 4.0]])
        point_pair.append([[0.0, -np.pi / 4.0], [np.pi, -np.pi / 4.0]])
        point_pair.append([[0.0, -np.pi / 2.0], [np.pi / 2.0, -np.pi / 2.0]])
        point_pair.append([[-np.pi * 3.0 / 4.0, 0.0], [np.pi * 3.0 / 4.0, 0.0]])
        point_pair.append([[0.0, 0.0], [0.0, 0.0]])
        point_pair.append([[np.pi / 4.0, np.pi / 4.0], [np.pi / 4.0, np.pi / 4.0]])
        point_pair.append([[0.0, -np.pi / 4.0], [0.0, np.pi / 4.0]])
        point_pair.append([[-np.pi / 4.0, np.pi / 4.0], [np.pi / 4.0, np.pi / 4.0]])
        point_pair.append([[-np.pi / 6.0, -np.pi / 8.0], [np.pi / 4.0, 0.0]])

        result = [np.pi / 2.0, np.pi / 2.0, np.pi / 2.0, np.pi / 4.0, np.pi / 2.0, 0.0, np.pi / 2.0, 0.0, 0.0, np.pi / 2, 1.0471975511965979, 1.32933932]

        points_1_u = np.zeros(len(point_pair))
        points_1_v = np.zeros(len(point_pair))
        points_2_u = np.zeros(len(point_pair))
        points_2_v = np.zeros(len(point_pair))
        for index in range(len(point_pair)):
            # term = point_pair[index]
            points_1_u[index] = point_pair[index][0][0]
            points_1_v[index] = point_pair[index][0][1]

            points_2_u[index] = point_pair[index][1][0]
            points_2_v[index] = point_pair[index][1][1]

        points_1_u = points_1_u[np.newaxis, ...]
        points_1_v = points_1_v[np.newaxis, ...]
        points_2_u = points_2_u[np.newaxis, ...]
        points_2_v = points_2_v[np.newaxis, ...]

        result_comput = sc.great_circle_distance_uv(points_1_u, points_1_v, points_2_u, points_2_v, radius=1)
        result_comput = result_comput[0]

        for index in range(len(point_pair)):
            print("----{}-----{}".format(index, point_pair[index]))
            print("error:    {}".format(np.sqrt(np.abs(result_comput[index] - result[index]))))
            print("GT:       {}".format(result[index]))
            print("Computed: {}".format(result_comput[index]))
            if not np.isclose(result[index], result_comput[index]):
                log.error(" ")

if __name__ == '__main__':
    unittest.main()