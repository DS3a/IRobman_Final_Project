import numpy as np
from typing import Tuple, Sequence, Optional, Any
import open3d as o3d
import pybullet as p  # Import pybullet for visualization
from sklearn.decomposition import PCA
import random

class GraspGeneration:
    def __init__(self):
        pass

    def sample_grasps(
        self,
        center_point: np.ndarray,
        num_grasps: int,
        radius: float = 0.1,
        sim = None
        object_pcd: o3d.geometry.PointCloud = None
    ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates multiple random grasp poses around a given point cloud.

        Args:
            center: Center around which to sample grasps.
            num_grasps: Number of random grasp poses to generate
            radius: Maximum distance offset from the center (meters)

        Returns:
            list: List of rotations and Translations
        """

        grasp_poses_list = []
        table_height = sim.robot.pos[2] + 0.01 # 0.01m higher than robot base from visualisation
        
        for idx in range(num_grasps):
            # Sample a grasp center and rotation of the grasp
            # Sample a random vector in R3 for axis angle representation
            # Return the rotation as rotation matrix + translation
            # Translation implies translation from a center point
            theta = np.random.uniform(0, 2*np.pi)

            # phi = np.random.uniform(0, np.pi)
            # this creates a lot of points around the pole. The points are not uniformly distributed around the sphere.
            # There is some transformation that can be applied to the random variable to remedy this issue, TODO look into that

            phi = np.arccos(1 - 2 * np.random.uniform(0, 1))
            # phi = np.arccos(np.random.uniform(0, 1))
            # source https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
            r = radius * (np.random.uniform(0, 1))**(1/3)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            grasp_center = center_point + np.array([x, y, z])
            grasp_center[2] = max(grasp_center[2], table_height)
            print(f"grasp_center: {grasp_center}")

            # axis = np.random.normal(size=3)
            # axis = np.array([0, 0, -1])
            # axis /= np.linalg.norm(axis)
            # angle = np.random.uniform(0, 2 * np.pi)

            # K =  np.array([
            #     [0, -axis[2], axis[1]],
            #     [axis[2], 0, -axis[0]],
            #     [-axis[1], axis[0], 0],
            # ])
            # R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*K.dot(K)

           # offset = np.random.uniform(0, np.pi/12)
            offset = 0
            
            Rx = np.array([
                [1,  0,  0],
                [ 0, np.cos(offset+np.pi/2),  -np.sin(offset+np.pi/2)],
                [ 0, np.sin(offset+np.pi/2),  np.cos(offset+np.pi/2)]
            ])
            
            # Generate a random angle for X rotation
            theta = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Rotation matrix about X-axis
            Ry = np.array([
                [cos_t, 0, sin_t],
                [ 0, 1, 0],
                [-sin_t, 0, cos_t]
            ])

            # R = np.eye(3)

            # offset = np.random.uniform(0, np.pi/12)
            offset = 0
            if object_pcd is not None:
                object_pcd.estimate_normals()
                surface_normals = object_pcd.normals
                pca = PCA(n_components=3)
                # find normals which are opposite to each other and generate R is that direction
                pca.fit(surface_normals)
                principal_directions = pca.components_
                two_numbers = random.sample([0, 1, 2], 2)  # Pick two distinct numbers
                one_number = random.choice(two_numbers)
                second_number = list(set(two_numbers) - {one_number})[0]
                x_axis = principal_directions[one_number]
                # x_axis = principal_directions[1]
                x_axis = np.array(x_axis, dtype=float)
                x_axis /= np.linalg.norm(x_axis)  # Normalize the x-axis direction
                # Generate a random vector that is not parallel to x_axis
                rand_vec = principal_directions[second_number]
                # rand_vec = principal_directions[0]
                rand_vec -= rand_vec.dot(x_axis) * x_axis  # Remove any component along x_axis
                rand_vec /= np.linalg.norm(rand_vec)  # Normalize
                # Compute the third axis as the cross product
                z_axis = np.cross(x_axis, rand_vec)
                z_axis /= np.linalg.norm(z_axis)  # Normalize
                # The y-axis is perpendicular to both x and z
                y_axis = np.cross(z_axis, x_axis)
                # Construct the rotation matrix
                R = np.column_stack((x_axis, y_axis, z_axis))
                assert grasp_center.shape == (3,)
                grasp_poses_list.append((R, grasp_center))
                return grasp_poses_list
            else:
                offset = np.random.uniform(0, np.pi/6)

                Rx = np.array([
                    [1,  0,  0],
                    [ 0, np.cos(offset+np.pi/2),  -np.sin(offset+np.pi/2)],
                    [ 0, np.sin(offset+np.pi/2),  np.cos(offset+np.pi/2)]
                ])
                
                # Generate a random angle for X rotation
                theta = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
                cos_t, sin_t = np.cos(theta), np.sin(theta)

                # Rotation matrix about X-axis
                Ry = np.array([
                    [cos_t, 0, sin_t],
                    [ 0, 1, 0],
                    [-sin_t, 0, cos_t]
                ])

                # Ry = np.eye(3)

                Rx_again = np.array([
                    [1, 0, 0],
                    [0, np.cos(offset), -np.sin(offset)],
                    [0, np.sin(offset), np.cos(offset)]
                ])

                # Final rotation matrix: First apply Rx, then Rx_again
                R = Rx @ Ry @ Rx_again 
                assert grasp_center.shape == (3,)
                grasp_poses_list.append((R, grasp_center))

                return grasp_poses_list
 





            # assert R.shape == (3, 3)
   

    def check_grasp_collision(
        self,
        grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
        # object_mesh: o3d.geometry.TriangleMesh,
        object_pcd,
        num_colisions: int = 10,
        tolerance: float = 0.00001) -> bool:
        """
        Checks for collisions between a gripper grasp pose and target object
        using point cloud sampling.

        Args:
            grasp_meshes: List of mesh geometries representing the gripper components
            object_mesh: Triangle mesh of the target object
            num_collisions: Threshold on how many points to check
            tolerance: Distance threshold for considering a collision (in meters)

        Returns:
            bool: True if collision detected between gripper and object, False otherwise
        """
        # Combine gripper meshes
        combined_gripper = o3d.geometry.TriangleMesh()
        for mesh in grasp_meshes:
            combined_gripper += mesh

        # Sample points from both meshes
        num_points = 5000 # Subsample both meshes to this many points
        gripper_pcl = combined_gripper.sample_points_uniformly(number_of_points=num_points)
        # object_pcl = object_mesh.sample_points_uniformly(number_of_points=num_points)
        object_pcl = object_pcd

        # Build KDTree for object points
        is_collision = False
        object_kd_tree = o3d.geometry.KDTreeFlann(object_pcl)
        collision_count = 0
        for point in gripper_pcl.points:
            [_, idx, distance] = object_kd_tree.search_knn_vector_3d(point, 1)
            if distance[0] < tolerance:
                collision_count += 1
                if collision_count >= num_colisions:
                    return True  # Collision detected

        return is_collision
    
    def grasp_dist_filter(self,
                        center_grasp: np.ndarray,
                        mesh_center: np.ndarray,
                        tolerance: float = 0.05)->bool:
        is_within_range = False
        #######################TODO#######################
        if np.linalg.norm(center_grasp - mesh_center) < tolerance:
            is_within_range = True
        ##################################################
        return is_within_range
    
    def check_grasp_containment(
        self,
        left_finger_center: np.ndarray,
        right_finger_center: np.ndarray,
        finger_length: float,
        object_pcd: o3d.geometry.PointCloud,
        num_rays: int,
        rotation_matrix: np.ndarray, # rotation-mat
        visualize_rays: bool = False  # 是否在PyBullet中可视化射线
    ) -> Tuple[bool, float, float]:
        """
        Checks if any line between the gripper fingers intersects with the object mesh.

        Args:
            left_finger_center: Center of Left finger of grasp
            right_finger_center: Center of Right finger of grasp
            finger_length: Finger Length of the gripper.
            object_pcd: Point Cloud of the target object
            clearance_threshold: Minimum required clearance between object and gripper



        Returns:
            tuple[bool, float]: (intersection_exists, intersection_depth)
            - intersection_exists: True if any line between fingers intersects object

            - intersection_depth: Depth of deepest intersection point
        """

        left_center = np.asarray(left_finger_center)
        right_center = np.asarray(right_finger_center)

        intersections = []
        # Check for intersections between corresponding points
        object_tree = o3d.geometry.KDTreeFlann(object_pcd)

        obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=object_pcd, 
                                                                                          alpha=0.016)
        # I just tuned alpha till I got a complete mesh with no holes, which had the best fidelity to the shape from the pcd

        obj_triangle_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj_triangle_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        obj_id = scene.add_triangles(obj_triangle_mesh_t)

        # visualize_3d_objs([obj_triangle_mesh])

        # now to know which direction to cast the rays towards, I added another coordinate 
        #       frame in the cell above question 1 in this task (task 2)
        # As shown in the coordinate frame, the fingers' tips begin at the at y=0, z=0 line, 
        # while the rest of the fingers extend along the +y axis

        hand_width = np.linalg.norm(left_center-right_center)
        finger_vec = np.array([0, finger_length, 0])
        ray_direction = (left_center - right_center)/hand_width
        # tolerance = 0.00001
        rays_hit = 0
        contained = False
        rays = []
        # max_interception_depth = 0
        # photon_translation = 1/50000  # I chose this as we are sampling the object into 50000 points
        
        # 用于存储射线的起点和终点，用于可视化
        ray_start_points = []
        ray_end_points = []
        
        # move the right centre to the start of the finger instead of the geometric centre
        right_center = right_center - rotation_matrix.dot(finger_vec/2)
        for i in range(num_rays):
            print(f"ray {i+1}/{num_rays}")
            # we are casting a ray from the right finger to the left
            right_new_center = right_center + rotation_matrix.dot((i/num_rays)*finger_vec)
            rays.append([np.concatenate([right_new_center, ray_direction])])
            
            # 存储射线起点和终点用于可视化
            ray_start_points.append(right_new_center)
            ray_end_points.append(right_new_center + ray_direction * hand_width)

        # 在PyBullet中可视化射线
        debug_lines = []
        if visualize_rays:
            print("在PyBullet中可视化射线...")
            for start, end in zip(ray_start_points, ray_end_points):
                line_id = p.addUserDebugLine(
                    start.tolist(), 
                    end.tolist(), 
                    lineColorRGB=[1, 0, 0],  # 红色
                    lineWidth=1,
                    lifeTime=2  # 2秒后自动消失
                )
                debug_lines.append(line_id)

        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_t)
        print(ans['t_hit'])

        rays_hit = 0
        max_interception_depth = o3d.core.Tensor([0.0], dtype=o3d.core.Dtype.Float32)
        rays = []
        for idx, hit_point in enumerate(ans['t_hit']):
            print(f"the hitpoint is {hit_point[0] < hand_width}")
            if hit_point[0] < hand_width:
                # I need to cast a ray from the left finger to check the depth and find the intersection depth
                contained = True
                rays_hit += 1
                left_new_center = left_center + rotation_matrix.dot((idx/num_rays)*finger_vec)
                rays.append([np.concatenate([left_new_center, -ray_direction])])

        containment_ratio = 0.0
        if contained:
            rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
            ans_left = scene.cast_rays(rays_t)


            for idx, hitpoint in enumerate(ans['t_hit']):
                left_idx = 0
                if hitpoint[0] < hand_width: 
                    interception_depth = hand_width - ans_left['t_hit'][0].item() - hitpoint[0].item()
                    max_interception_depth = max(max_interception_depth, interception_depth)
                    left_idx += 1

        print(f"the max interception depth is {max_interception_depth}")
        containment_ratio = rays_hit / num_rays
        print(f"射线命中率: {containment_ratio:.4f} ({rays_hit}/{num_rays})")
        
        intersections.append(contained)
        # intersections.append(max_interception_depth[0])
        # return contained, containment_ratio


        return any(intersections), containment_ratio, max_interception_depth.item()


    # def check_grasp_containment(
    #     self,
    #     left_finger_center: np.ndarray,
    #     right_finger_center: np.ndarray,
    #     finger_length: float,
    #     object_pcd: o3d.geometry.PointCloud,
    #     num_rays: int,
    #     rotation_matrix: np.ndarray, # rotation-mat
    #     visualize_rays: bool = False  # Whether to visualize rays in PyBullet
    # ) -> Tuple[bool, float, float]:
    #     """
    #     Checks if any line between the gripper fingers intersects with the object mesh.

    #     Args:
    #         left_finger_center: Center of Left finger of grasp
    #         right_finger_center: Center of Right finger of grasp
    #         finger_length: Finger Length of the gripper.
    #         object_pcd: Point Cloud of the target object
    #         num_rays: Number of rays to cast
    #         rotation_matrix: Rotation matrix for the grasp
    #         visualize_rays: Whether to visualize rays in PyBullet

    #     Returns:
    #         tuple[bool, float, float]: 
    #         - intersection_exists: True if any line between fingers intersects object
    #         - containment_ratio: Ratio of rays that hit the object
    #         - intersection_depth: Depth of deepest intersection point
    #     """
    #     left_center = np.asarray(left_finger_center)
    #     right_center = np.asarray(right_finger_center)

    #     intersections = []
    #     # Check for intersections between corresponding points
    #     object_tree = o3d.geometry.KDTreeFlann(object_pcd)

    #     # Calculate object height and bounding box
    #     points = np.asarray(object_pcd.points)
    #     min_point = np.min(points, axis=0)
    #     max_point = np.max(points, axis=0)
    #     object_height = max_point[2] - min_point[2]
    #     object_center = (min_point + max_point) / 2
        
    #     print(f"Object height: {object_height:.4f}m")
    #     print(f"Object center point: {object_center}")

    #     obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=object_pcd, 
    #                                                                                       alpha=0.016)
    #     # I just tuned alpha till I got a complete mesh with no holes, which had the best fidelity to the shape from the pcd
        
    #     obj_triangle_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj_triangle_mesh)
    #     scene = o3d.t.geometry.RaycastingScene()
    #     obj_id = scene.add_triangles(obj_triangle_mesh_t)

    #     # visualize_3d_objs([obj_triangle_mesh])

    #     # now to know which direction to cast the rays towards, I added another coordinate 
    #     #       frame in the cell above question 1 in this task (task 2)
    #     # As shown in the coordinate frame, the fingers' tips begin at the at y=0, z=0 line, 
    #     # while the rest of the fingers extend along the +y axis

    #     hand_width = np.linalg.norm(left_center-right_center)
    #     finger_vec = np.array([0, finger_length, 0])
    #     ray_direction = (left_center - right_center)/hand_width
        
    #     # For low height objects, adjust ray starting height
    #     if object_height < 0.05:  # If object height is less than 5cm
    #         print("Detected low height object, adjusting ray height...")
            
    #         # Calculate the difference in z-axis between finger center and object center
    #         z_diff = (right_center[2] + left_center[2]) / 2 - object_center[2]
    #         print(f"left_center: {left_center}, right_center: {right_center}, object_center: {object_center}")
    #         # If finger center is higher than object center, lower finger center to object center height
    #         if z_diff > 0.01:  # If difference is greater than 1cm
    #             height_adjustment = z_diff - 0.01  # Keep 1cm margin
    #             right_center[2] -= height_adjustment
    #             left_center[2] -= height_adjustment
    #             print(f"Ray height adjusted: {height_adjustment:.4f}m")
        
    #     # For storing ray start and end points for visualization
    #     ray_start_points = []
    #     ray_end_points = []
        
    #     # ===== Calculate gripper width direction =====
    #     print("Calculating gripper width direction...")
    #     # Calculate gripper width direction vector (perpendicular to both ray_direction and finger_vec)
    #     # First calculate finger_vec direction in world coordinate system
    #     world_finger_vec = rotation_matrix.dot(finger_vec)
    #     # Calculate width direction vector (cross product gives vector perpendicular to both vectors)
    #     width_direction = np.cross(ray_direction, world_finger_vec)
    #     # Normalize
    #     width_direction = width_direction / np.linalg.norm(width_direction)
        
    #     # Define width direction parameters
    #     width_planes = 1  # Number of planes on each side in width direction
    #     width_offset = 0.02  # Offset between planes (meters)
        
    #     # ===== Generate multiple parallel ray planes =====
    #     print("Generating multiple parallel ray planes...")
    #     # Center plane (original plane)
    #     rays = []
    #     contained = False
    #     rays_hit = 0
        
    #     # Parallel planes on both sides in width direction
    #     for plane in range(1, width_planes + 1):
    #         # Calculate current plane offset
    #         current_offset = width_offset * plane
            
    #         # Right side plane
    #         for i in range(num_rays):
    #             # Calculate sampling point in length direction, with offset in width direction
    #             right_point = right_center + rotation_matrix.dot((i/num_rays)*finger_vec) + width_direction * current_offset
    #             # Add ray from right offset point to left offset point
    #             rays.append([np.concatenate([right_point, ray_direction])])
                
    #             # Store ray start and end points for visualization - using actual finger width
    #             ray_start_points.append(right_point)
    #             ray_end_points.append(right_point + ray_direction * hand_width)
            
    #         # Left side plane
    #         for i in range(num_rays):
    #             # Calculate sampling point in length direction, with offset in width direction
    #             right_point = right_center + rotation_matrix.dot((i/num_rays)*finger_vec) - width_direction * current_offset
    #             # Add ray from right offset point to left offset point
    #             rays.append([np.concatenate([right_point, ray_direction])])
                
    #             # Store ray start and end points for visualization - using actual finger width
    #             ray_start_points.append(right_point)
    #             ray_end_points.append(right_point + ray_direction * hand_width)
        
    #     print(f"Total rays generated: {len(rays)}")
        
    #     # Visualize rays in PyBullet
    #     debug_lines = []
    #     if visualize_rays:
    #         print("Visualizing rays in PyBullet...")
    #         for start, end in zip(ray_start_points, ray_end_points):
    #             line_id = p.addUserDebugLine(
    #                 start.tolist(), 
    #                 end.tolist(), 
    #                 lineColorRGB=[1, 0, 0],  # Red
    #                 lineWidth=1,
    #                 lifeTime=5  # Disappear after 5 seconds
    #             )
    #             debug_lines.append(line_id)
        
    #     # Perform ray casting
    #     rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    #     ans = scene.cast_rays(rays_t)
        
    #     # Process ray casting results
    #     rays_hit = 0
    #     max_interception_depth = o3d.core.Tensor([0.0], dtype=o3d.core.Dtype.Float32)
    #     rays_from_left = []
        
    #     # Process all ray results
    #     print("Processing ray casting results...")
    #     for idx, hit_point in enumerate(ans['t_hit']):
    #         # Check if ray hits object using actual finger width
    #         if hit_point[0] < hand_width:
    #             contained = True
    #             rays_hit += 1
                
    #             # Only calculate depth for rays in center plane (original plane)
    #             if idx < num_rays:
    #                 left_new_center = left_center + rotation_matrix.dot((idx/num_rays)*finger_vec)
    #                 rays_from_left.append([np.concatenate([left_new_center, -ray_direction])])
        
    #     containment_ratio = 0.0
    #     if contained:
    #         # Process rays from left side (only for center plane)
    #         if rays_from_left:
    #             rays_t = o3d.core.Tensor(rays_from_left, dtype=o3d.core.Dtype.Float32)
    #             ans_left = scene.cast_rays(rays_t)
                
    #             for idx, hitpoint in enumerate(ans['t_hit']):
    #                 if idx < num_rays:  # Only process rays in center plane
    #                     left_idx = 0
    #                     # Calculate interception depth using actual finger width
    #                     if hitpoint[0] < hand_width: 
    #                         interception_depth = hand_width - ans_left['t_hit'][0].item() - hitpoint[0].item()
    #                         max_interception_depth = max(max_interception_depth, interception_depth)
    #                         left_idx += 1

    #     print(f"the max interception depth is {max_interception_depth}")
    #     # Calculate total ray hit ratio
    #     total_rays = len(rays)
    #     containment_ratio = rays_hit / total_rays
    #     print(f"Ray hit ratio: {containment_ratio:.4f} ({rays_hit}/{total_rays})")
        
    #     intersections.append(contained)
    #     # intersections.append(max_interception_depth[0])
    #     # return contained, containment_ratio

    #     return any(intersections), containment_ratio, max_interception_depth.item()
