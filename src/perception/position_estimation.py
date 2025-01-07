import numpy as np
import open3d as o3d


class PositionEstimation:
    """
        This class is used to estimate the position of any object 
        given a depth image, and the corresponding camera matrix
    """

    def __init__(self, camera_matrix):
        """
            Args:
                camera_matrix:
                    The camera matrix that will be used to project the depth image to a pointcloud
                    We can average out the pointcloud data to find the position, 
                    or use some matching algorithm to determine the object's orientation as well
        """
        self.camera_matrix = camera_matrix

    def determine_position(self):
        position = np.array([0, 0, 0])

        return position
