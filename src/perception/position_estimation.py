import numpy as np
import open3d as o3d
from scipy.linalg import rq
import pybullet as p


class PositionEstimation:
    """
        This class is used to estimate the position of any object 
        given a depth image, and the corresponding camera matrix
    """

    def __init__(self, camera_settings, camera_projection_matrix=None):
        """
            Args:
                camera_matrix:
                    The camera matrix that will be used to project the depth image to a pointcloud
                    We can average out the pointcloud data to find the position, 
                    or use some matching algorithm to determine the object's orientation as well
        """
        self.camera_intrinsics = None
        self.camera_settings = camera_settings
        self.camera_projection_matrix = camera_projection_matrix

    def determine_position(self, segmented_depth_img):
        """
            Accepts the segmented depth image, converts it to a pointcloud and determines the position by averaging the pcl
        """

        linearized_depth = self.linearize_depth_image(segmented_depth_img)
        pcd = self.depth_to_pcl(linearized_depth)
        position = np.mean(pcd.points, axis=0) + np.array([-0.6, -0.5, 0.3])
       
        return position

    def compute_camera_intrinsics(self):

        return self.camera_intrinsics
    
    def get_intrinsics(self):
        fov = self.camera_settings["fov"]
        width = self.camera_settings["width"]
        height = self.camera_settings["height"]
        aspect = width / height
        fx = width / (2 * aspect * np.tan(np.radians(fov / 2)))
        fy = height / (2 * np.tan(np.radians(fov / 2)))
        cx = width / 2
        cy = height / 2
        return fx, fy, cx, cy

    def get_extrinsics(self):
        Tc = np.array([[1,  0,  0,  0],
                    [0,  -1,  0,  0],
                    [0,  0,  -1,  0],
                    [0,  0,  0,  1]]).reshape(4,4)
        view_matrix = p.computeViewMatrix(
            self.camera_settings['stat_cam_pos'],
            self.camera_settings['stat_cam_target_pos'],
            cameraUpVector=[0, 0, 1])
        view_matrix = np.array(view_matrix).reshape(4, 4)

        T = np.linalg.inv(view_matrix) @ Tc

        return T

    def linearize_depth_image(self, depth_img):
        near = self.camera_settings["near"]
        far = self.camera_settings["far"]

        linearized_depth = far * near / (far - (far - near) * depth_img)

        return linearized_depth

    def depth_to_pcl(self, linearized_depth_img):
        (fx, fy, cx, cy) = self.get_intrinsics()
        o3d_img = o3d.geometry.Image((linearized_depth_img*1000).astype(np.uint16))
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(width=self.camera_settings["width"], 
                                  height=self.camera_settings["height"], 
                                  fx=fx, 
                                  fy=fy, 
                                  cx=cx, 
                                  cy=cy)
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth=o3d_img,
            intrinsic=intrinsics,   #)
            extrinsic=self.get_extrinsics() )

        forward = np.array(self.camera_settings['stat_cam_target_pos']) - np.array(self.camera_settings['stat_cam_pos'])
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.column_stack([right, up, forward])

        point_cloud = point_cloud.translate(np.array(self.camera_settings["stat_cam_pos"])*(0.4))
        # print(f"the camera position is {np.array(self.camera_settings['stat_cam_pos']).reshape(3, 1)}")
        # point_cloud = point_cloud.rotate(R, center=np.array(self.camera_settings['stat_cam_pos']).reshape(3, 1))

        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        # o3d.visualization.draw_geometries([point_cloud, coordinate_frame])
        return point_cloud