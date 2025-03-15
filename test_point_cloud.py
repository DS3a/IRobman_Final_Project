import os
import glob
import yaml
import time
import random
import numpy as np
import pybullet as p
import open3d as o3d
from pybullet_object_models import ycb_objects  # type:ignore
from src.simulation import Simulation
from src.ik_solver import DifferentialIKSolver
from src.obstacle_tracker import ObstacleTracker
from src.rrt_star import RRTStarPlanner
from src.grasping.grasp_generation_new import GraspGeneration
# from src.grasping.grasp_generation import GraspGeneration
from src.grasping import utils
from scipy.spatial.transform import Rotation

# Check if PyBullet has NumPy support enabled
numpy_support = p.isNumpyEnabled()
print(f"PyBullet NumPy support enabled: {numpy_support}")

def convert_depth_to_meters(depth_buffer, near, far):
    """
    convert depth buffer values to actual distance (meters)
    
    Parameters:
    depth_buffer: depth buffer values obtained from PyBullet
    near, far: near/far plane distances
    
    Returns:
    actual depth values in meters
    """
    return far * near / (far - (far - near) * depth_buffer)

def get_camera_intrinsic(width, height, fov):
    """
    calculate intrinsic matrix from camera parameters
    
    Parameters:
    width: image width (pixels)
    height: image height (pixels)
    fov: vertical field of view (degrees)

    Returns:
    camera intrinsic matrix
    """    
    # calculate focal length
    f = height / (2 * np.tan(np.radians(fov / 2)))
    
    # calculate principal point
    cx = width / 2
    cy = height / 2
    
    intrinsic_matrix = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    
    return intrinsic_matrix

def depth_image_to_point_cloud(depth_image, mask, rgb_image, intrinsic_matrix):
    """
    depth image to camera coordinate point cloud
    
    Parameters:
    depth_image: depth image (meters)
    mask: target object mask (boolean array)
    rgb_image: RGB image
    intrinsic_matrix: camera intrinsic matrix
    
    Returns:
    camera coordinate point cloud (N,3) and corresponding colors (N,3)
    """
    # extract pixel coordinates of target mask
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        raise ValueError("No valid pixels found in target mask")
    
    # extract depth values of these pixels
    depths = depth_image[rows, cols]
    
    # image coordinates to camera coordinates
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # calculate camera coordinates
    x = -(cols - cx) * depths / fx # negative sign due to PyBullet camera orientation???
    y = -(rows - cy) * depths / fy
    z = depths
    
    # stack points
    points = np.vstack((x, y, z)).T
    
    # extract RGB colors
    colors = rgb_image[rows, cols, :3].astype(np.float64) / 255.0
    
    return points, colors

def transform_points_to_world(points, camera_extrinsic):
    """
    transform points from camera coordinates to world coordinates
    
    Parameters:
    points: point cloud in camera coordinates (N,3)
    camera_extrinsic: camera extrinsic matrix (4x4)
    
    Returns:
    point cloud in world coordinates (N,3)
    """
    # convert point cloud to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # transform point cloud using extrinsic matrix
    world_points_homogeneous = (camera_extrinsic @ points_homogeneous.T).T # points in rows
    
    # convert back to non-homogeneous coordinates
    world_points = world_points_homogeneous[:, :3]
    
    return world_points

def get_camera_extrinsic(camera_pos, camera_R):
    """
    build camera extrinsic matrix (transform from camera to world coordinates)
    
    Parameters:
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (3x3)
    
    Returns:
    camera extrinsic matrix (4x4)
    """
    # build 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = camera_R
    extrinsic[:3, 3] = camera_pos
    
    return extrinsic

def build_object_point_cloud_ee(rgb, depth, seg, target_mask_id, config, camera_pos, camera_R):
    """
    build object point cloud using end-effector camera RGB, depth, segmentation data
    
    Parameters:
    rgb: RGB image
    depth: depth buffer values
    seg: segmentation mask
    target_mask_id: target object ID
    config: configuration dictionary
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (from camera to world coordinates)
    
    Returns:
    Open3D point cloud object
    """
    # read camera parameters
    cam_cfg = config["world_settings"]["camera"]
    width = cam_cfg["width"]
    height = cam_cfg["height"]
    fov = cam_cfg["fov"]  # vertical FOV
    near = cam_cfg["near"]
    far = cam_cfg["far"]
    
    # create target object mask
    object_mask = (seg == target_mask_id)
    if np.count_nonzero(object_mask) == 0:
        raise ValueError(f"Target mask ID {target_mask_id} not found in segmentation.")
    
    # extract depth buffer values for target object
    metric_depth = convert_depth_to_meters(depth, near, far)
    
    # get intrinsic matrix
    intrinsic_matrix = get_camera_intrinsic(width, height, fov)
    
    # convert depth image to point cloud
    points_cam, colors = depth_image_to_point_cloud(metric_depth, object_mask, rgb, intrinsic_matrix)
    
    # build camera extrinsic matrix
    camera_extrinsic = get_camera_extrinsic(camera_pos, camera_R)
    
    # transform points to world coordinates
    points_world = transform_points_to_world(points_cam, camera_extrinsic)
    
    # create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def get_ee_camera_params(robot, config):
    """
    get end-effector camera position and rotation matrix
    
    Parameters:
    robot: robot object
    config: configuration dictionary
    
    Returns:
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (from camera to world coordinates)
    """
    # end-effector pose
    ee_pos, ee_orn = robot.get_ee_pose()
    
    # end-effector rotation matrix
    ee_R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    print("End effector orientation matrix:")
    print(ee_R)
    # camera parameters
    cam_cfg = config["world_settings"]["camera"]
    ee_offset = np.array(cam_cfg["ee_cam_offset"])
    ee_cam_orn = cam_cfg["ee_cam_orientation"]
    ee_cam_R = np.array(p.getMatrixFromQuaternion(ee_cam_orn)).reshape(3, 3)
    # calculate camera position
    camera_pos = ee_pos # why ee_pos + ee_R @ ee_offset will be wrong?
    # calculate camera rotation matrix
    camera_R = ee_R @ ee_cam_R
    
    return camera_pos, camera_R

# for linear trajectory in Cartesian space
def generate_cartesian_trajectory(sim, ik_solver, start_joints, target_pos, target_orn, steps=100):
    """
    generate linear Cartesian trajectory in Cartesian space
    """
    # set start position
    for i, joint_idx in enumerate(sim.robot.arm_idx):
        p.resetJointState(sim.robot.id, joint_idx, start_joints[i])
    
    # get current end-effector pose
    ee_state = p.getLinkState(sim.robot.id, sim.robot.ee_idx)
    print(f"ee_state_0={np.array(ee_state[0])}, ee_state_1={np.array(ee_state[1])}")
    start_pos = np.array(ee_state[0])
    
    # generate linear trajectory
    trajectory = []
    for step in range(steps + 1):
        t = step / steps  # normalize step
        
        # linear interpolation
        pos = start_pos + t * (target_pos - start_pos)
        
        # solve IK for current Cartesian position
        current_joints = ik_solver.solve(pos, target_orn, start_joints, max_iters=50, tolerance=0.001)
        
        # add solution to trajectory
        trajectory.append(current_joints)
        
        # reset to start position
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, start_joints[i])
    
    return trajectory

# for trajectory in joint space
def generate_trajectory(start_joints, end_joints, steps=100):
    """
    generate smooth trajectory from start to end joint positions
    
    Parameters:
    start_joints: start joint positions
    end_joints: end joint positions
    steps: number of steps for interpolation
    
    Returns:
    trajectory: list of joint positions
    """
    trajectory = []
    for step in range(steps + 1):
        t = step / steps  # normalize step
        # linear interpolation
        point = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
        trajectory.append(point)
    return trajectory

def generate_rrt_star_trajectory(sim, rrt_planner, start_joints, target_joints, visualize=True):
    """
    Generate a collision-free trajectory using RRT* planning.
    
    Args:
        sim: Simulation instance
        rrt_planner: RRTStarPlanner instance
        start_joints: Start joint configuration
        target_joints: Target joint configuration
        visualize: Whether to visualize the planning process
        
    Returns:
        Smooth trajectory as list of joint configurations
    """
    print("Planning path with RRT*...")
    
    # Plan path using RRT*
    path, path_cost = rrt_planner.plan(start_joints, target_joints)
    
    if not path:
        print("Failed to find a valid path!")
        return []
    
    print(f"Path found with {len(path)} waypoints and cost {path_cost:.4f}")
    
    # Generate smooth trajectory
    trajectory = rrt_planner.generate_smooth_trajectory(path, smoothing_steps=20)
    
    print(f"Generated smooth trajectory with {len(trajectory)} points")
    
    # Visualize the path if requested
    if visualize:
        # Clear previous visualization
        rrt_planner.clear_visualization()
        
        # Visualize the path
        for i in range(len(path) - 1):
            start_ee, _ = rrt_planner._get_current_ee_pose(path[i])
            end_ee, _ = rrt_planner._get_current_ee_pose(path[i+1])
            
            p.addUserDebugLine(
                start_ee, end_ee, [0, 0, 1], 3, 0)
            
    return trajectory

def visualize_point_clouds(collected_data, show_frames=True, show_merged=True):
    """
    Visualize collected point clouds using Open3D
    
    Parameters:
    collected_data: list of dictionaries containing point cloud data
    show_frames: whether to show coordinate frames
    show_merged: whether to show merged point cloud
    """
    if not collected_data:
        print("No point cloud data to visualize")
        return
        
    geometries = []
    
    # Add world coordinate frame
    if show_frames:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        geometries.append(coord_frame)
    
    if show_merged:
        # Merge point clouds using ICP
        print("Merging point clouds using ICP...")
        merged_pcd = iterative_closest_point(collected_data)
        if merged_pcd is not None:
            # Keep original colors from point clouds
            geometries.append(merged_pcd)
            print(f"Added merged point cloud with {len(merged_pcd.points)} points")
    else:
        # Add each point cloud and its camera frame
        for i, data in enumerate(collected_data):
            if 'point_cloud' in data and data['point_cloud'] is not None:
                # Add point cloud
                geometries.append(data['point_cloud'])
                
                # Add camera frame
                if show_frames:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    camera_frame.translate(data['camera_position'])
                    camera_frame.rotate(data['camera_rotation'])
                    geometries.append(camera_frame)
                    
                print(f"Added point cloud {i+1} with {len(data['point_cloud'].points)} points")
    
    print("Launching Open3D visualization...")
    o3d.visualization.draw_geometries(geometries)

def iterative_closest_point(collected_data):
    """
    Merge multiple point clouds using ICP registration
    
    Parameters:
    collected_data: list of dictionaries containing point cloud data
    
    Returns:
    merged_pcd: merged point cloud
    """
    if not collected_data:
        return None
        
    # Use the first point cloud as reference
    merged_pcd = collected_data[0]['point_cloud']
    
    # ICP parameters
    threshold = 0.005  # distance threshold
    trans_init = np.eye(4)  # initial transformation
    
    # Merge remaining point clouds
    for i in range(1, len(collected_data)):
        current_pcd = collected_data[i]['point_cloud']
        
        # Perform ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            current_pcd, merged_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Transform current point cloud
        current_pcd.transform(reg_p2p.transformation)
        
        # Merge point clouds
        merged_pcd += current_pcd
        
        # Optional: Remove duplicates using voxel downsampling
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)
        
        print(f"Merged point cloud {i+1}, fitness: {reg_p2p.fitness}")
    
    return merged_pcd

def run_grasping(config, sim, collected_point_clouds):
    """
    Execute grasp generation and visualization
    
    Args:
        config: Configuration dictionary
        sim: Simulation object
        collected_point_clouds: List of collected point cloud data
    """
    print("Merging point clouds and calculating centroid...")
    merged_pcd = iterative_closest_point(collected_point_clouds)
    centre_point = np.asarray(merged_pcd.points)
    centre_point = centre_point.mean(axis=0)
    print(f"Point cloud centroid coordinates: {centre_point}")
    
    # Visualize point cloud centroid in PyBullet
    print("Visualizing point cloud centroid in PyBullet...")
    # Create a red sphere to represent the centroid
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.02,  # 2cm radius sphere
        rgbaColor=[1, 0, 0, 1]  # Red
    )
    centroid_id = p.createMultiBody(
        baseMass=0,  # Mass of 0 indicates a static object
        baseVisualShapeIndex=visual_id,
        basePosition=centre_point.tolist()
    )
    
    # Add coordinate axes for better visualization of centroid position
    axis_length = 0.1  # 10cm long axes
    p.addUserDebugLine(
        centre_point, 
        centre_point + np.array([axis_length, 0, 0]), 
        [1, 0, 0], 3, 0  # X-axis - Red
    )
    p.addUserDebugLine(
        centre_point, 
        centre_point + np.array([0, axis_length, 0]), 
        [0, 1, 0], 3, 0  # Y-axis - Green
    )
    p.addUserDebugLine(
        centre_point, 
        centre_point + np.array([0, 0, axis_length]), 
        [0, 0, 1], 3, 0  # Z-axis - Blue
    )
    
    # Add text label
    p.addUserDebugText(
        f"Centroid ({centre_point[0]:.3f}, {centre_point[1]:.3f}, {centre_point[2]:.3f})",
        centre_point + np.array([0, 0, 0.05]),  # Display text 5cm above centroid
        [1, 1, 1],  # White text
        1.0  # Text size
    )
    
    # Calculate point cloud bounding box and height
    points = np.asarray(merged_pcd.points)
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)
    object_height = max_point[2] - min_point[2]
    print(f"Object height: {object_height:.4f}m")
    print(f"Object bounding box min point: {min_point}")
    print(f"Object bounding box max point: {max_point}")
    
    # Visualize object bounding box
    bbox_lines = [
        # Bottom rectangle
        [min_point, [max_point[0], min_point[1], min_point[2]]],
        [[max_point[0], min_point[1], min_point[2]], [max_point[0], max_point[1], min_point[2]]],
        [[max_point[0], max_point[1], min_point[2]], [min_point[0], max_point[1], min_point[2]]],
        [[min_point[0], max_point[1], min_point[2]], min_point],
        # Top rectangle
        [[min_point[0], min_point[1], max_point[2]], [max_point[0], min_point[1], max_point[2]]],
        [[max_point[0], min_point[1], max_point[2]], max_point],
        [max_point, [min_point[0], max_point[1], max_point[2]]],
        [[min_point[0], max_point[1], max_point[2]], [min_point[0], min_point[1], max_point[2]]],
        # Connecting lines
        [min_point, [min_point[0], min_point[1], max_point[2]]],
        [[max_point[0], min_point[1], min_point[2]], [max_point[0], min_point[1], max_point[2]]],
        [[max_point[0], max_point[1], min_point[2]], max_point],
        [[min_point[0], max_point[1], min_point[2]], [min_point[0], max_point[1], max_point[2]]]
    ]
    
    for line in bbox_lines:
        p.addUserDebugLine(
            line[0], 
            line[1], 
            [0, 1, 1],  # Cyan
            1, 
            0
        )
    
    # Initialize IK solver
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    
    # Get current joint positions
    current_joints = sim.robot.get_joint_positions()
    
    # Initialize grasp generator
    print("Generating grasp candidates...")
    grasp_generator = GraspGeneration()
    
    # Modify centroid height to better suit low height objects
    adjusted_centre_point = centre_point.copy()
    
    # If object height is small, adjust centroid height to half of object height
    if object_height < 0.1:  # If object height is less than 10cm
        print("Detected low height object, adjusting centroid height...")
        # Set centroid height to object bottom plus half of object height
        adjusted_centre_point[2] = min_point[2] + object_height / 2
        print(f"Adjusted centroid coordinates: {adjusted_centre_point}")
        
        # Visualize adjusted centroid
        visual_id_adjusted = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,  # 2cm radius sphere
            rgbaColor=[0, 1, 0, 1]  # Green
        )
        adjusted_centroid_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id_adjusted,
            basePosition=adjusted_centre_point.tolist()
        )
        
        # Add text label
        p.addUserDebugText(
            f"Adjusted centroid ({adjusted_centre_point[0]:.3f}, {adjusted_centre_point[1]:.3f}, {adjusted_centre_point[2]:.3f})",
            adjusted_centre_point + np.array([0, 0, 0.05]),
            [0, 1, 0],  # Green text
            1.0
        )
    else:
        adjusted_centre_point = centre_point
        
    # Generate grasp candidates using adjusted centroid
    sampled_grasps = grasp_generator.sample_grasps(adjusted_centre_point, 100, radius=0.1)
    
    # Create meshes for each grasp
    all_grasp_meshes = []
    for grasp in sampled_grasps:
        R, grasp_center = grasp
        all_grasp_meshes.append(utils.create_grasp_mesh(center_point=grasp_center, rotation_matrix=R))

    # Create triangle mesh from point cloud for visualization
    print("Creating triangle mesh from point cloud...")
    obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=merged_pcd, 
                                                                                      alpha=0.08)
 
    # Evaluate grasp quality
    print("Evaluating grasp quality...")
    vis_meshes = [obj_triangle_mesh]
    highest_quality = 0
    highest_containment_grasp = None
    best_grasp = None
    
    for (pose, grasp_mesh) in zip(sampled_grasps, all_grasp_meshes):
        print(f"grasp mesh:{grasp_mesh}")
        if not grasp_generator.check_grasp_collision(grasp_mesh, merged_pcd, num_colisions=1):
            valid_grasp, grasp_quality, max_interception_depth = grasp_generator.check_grasp_containment(
                grasp_mesh[0].get_center(), 
                grasp_mesh[1].get_center(),
                finger_length=0.05,
                object_pcd=merged_pcd,
                num_rays=50,
                rotation_matrix=pose[0],
                visualize_rays=False # toggle visualization of ray casting
            )
            # # Convert tensor to float if needed
            # if hasattr(grasp_quality, 'item'):
            #     grasp_quality = float(grasp_quality.item())
            
            # Use new quality metric to select grasp
            if valid_grasp and grasp_quality > highest_quality:
                highest_quality = grasp_quality
                highest_containment_grasp = grasp_mesh
                best_grasp = pose
                print(f"Found better grasp, quality: {grasp_quality:.3f}")

    # Visualize best grasp
    if highest_containment_grasp is not None:
        print(f"Found valid grasp, highest quality: {highest_quality:.3f}")
        vis_meshes.extend(highest_containment_grasp)
        
        # Print best grasp pose information
        print("\n========== Best Grasp Pose Information ==========")
        R, grasp_center = best_grasp
        print(f"Grasp center position: {grasp_center}")
        
        # Convert rotation matrix to quaternion
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w] format
        euler = rot.as_euler('xyz', degrees=True)  # Euler angles (degrees)
        
        print(f"Grasp rotation matrix:\n{R}")
        print(f"Grasp quaternion [x, y, z, w]: {quat}")
        print(f"Grasp Euler angles [x, y, z] (degrees): {euler}")
        
        # Calculate end effector target pose
        # Note: Here we assume the grasp center is the target position for the end effector,
        # and the rotation matrix is the target orientation for the end effector
        # In actual applications, this may need to be adjusted based on the specific robot configuration
        ee_target_pos = grasp_center
        ee_target_orn = p.getQuaternionFromEuler([euler[0]/180*np.pi, euler[1]/180*np.pi, euler[2]/180*np.pi])
        
        print("\n========== End Effector Target Pose ==========")
        print(f"Position: {ee_target_pos}")
        print(f"Quaternion [x, y, z, w]: {ee_target_orn}")
        
        # Add coordinate system transformation to make printed pose consistent with Open3D visualization
        # In Open3D, the grasp direction is vertically downward, which may be due to coordinate system differences
        
        # Analyze coordinate systems in grasp generation and visualization:
        # In GraspGeneration.sample_grasps:
        # - Created rotation matrix R = Rx @ Ry @ Rx_again
        # - Where Rx rotated by π/2, making the grasp direction possibly along the y-axis
        
        # This matrix combines a rotation mapping y-axis to -z-axis and a 90-degree rotation around z-axis
        combined_transform = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        
        # Apply the combined transformation directly
        R_world = R @ combined_transform
        
        rot_world = Rotation.from_matrix(R_world)
        quat_world = rot_world.as_quat()
        euler_world = rot_world.as_euler('xyz', degrees=True)
        
        print("\n========== Transformed End Effector Pose ==========")
        print(f"Position: {ee_target_pos}")
        print(f"Rotation matrix:\n{R_world}")
        print(f"Quaternion [x, y, z, w]: {quat_world}")
        print(f"Euler angles [x, y, z] (degrees): {euler_world}")
        
        # ===== Add robot arm motion planning code =====
        print("\n========== Planning Robot Arm Trajectory ==========")
        
        # Define pose 2 (final grasp pose)
        pose2_pos = ee_target_pos
        pose2_orn = p.getQuaternionFromEuler([euler_world[0]/180*np.pi, euler_world[1]/180*np.pi, euler_world[2]/180*np.pi])
        
        # Calculate pose 1 (pre-grasp position) - move back 0.05m along pose 2's own z-axis
        # Get pose 2's rotation matrix to calculate z-axis direction
        pose2_rot_matrix = R_world
        
        # Extract the third column of the rotation matrix, which is the z-axis direction
        z_axis = pose2_rot_matrix[:, 2]
        
        # Move back 0.15m along z-axis direction
        pose1_pos = pose2_pos - 0.15 * z_axis
        pose1_orn = pose2_orn  # Keep the same orientation
        
        print(f"Pose 2 (final grasp pose) - Position: {pose2_pos}, Orientation: {pose2_orn}")
        print(f"Pose 1 (pre-grasp position) - Position: {pose1_pos}, Orientation: {pose1_orn}")
        
        # ===== Visualize pose 1 and pose 2 coordinate axes in PyBullet =====
        print("\n========== Visualizing Grasp Pose Coordinate Axes ==========")
        
        # Coordinate axis length
        axis_length = 0.1
        
        # Get rotation matrices from quaternions
        pose1_rot = np.array(p.getMatrixFromQuaternion(pose1_orn)).reshape(3, 3)
        pose2_rot = np.array(p.getMatrixFromQuaternion(pose2_orn)).reshape(3, 3)
        
        # Extract direction vectors for each axis
        pose1_x_axis = pose1_rot[:, 0] * axis_length
        pose1_y_axis = pose1_rot[:, 1] * axis_length
        pose1_z_axis = pose1_rot[:, 2] * axis_length
        
        pose2_x_axis = pose2_rot[:, 0] * axis_length
        pose2_y_axis = pose2_rot[:, 1] * axis_length
        pose2_z_axis = pose2_rot[:, 2] * axis_length
        
        # Visualize Pose 1 coordinate axes
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_x_axis, [1, 0, 0], 3, 0)  # X-axis - Red
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_y_axis, [0, 1, 0], 3, 0)  # Y-axis - Green
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_z_axis, [0, 0, 1], 3, 0)  # Z-axis - Blue
        
        # Visualize Pose 2 coordinate axes
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_x_axis, [1, 0, 0], 3, 0)  # X-axis - Red
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_y_axis, [0, 1, 0], 3, 0)  # Y-axis - Green
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_z_axis, [0, 0, 1], 3, 0)  # Z-axis - Blue
        
        # Add text labels
        p.addUserDebugText("Pose 1", pose1_pos + [0, 0, 0.05], [1, 1, 1], 1.5)
        p.addUserDebugText("Pose 2", pose2_pos + [0, 0, 0.05], [1, 1, 1], 1.5)
        
        # Get current robot arm joint angles (last point of point cloud collection)
        start_joints = sim.robot.get_joint_positions()
        print(f"Starting joint angles: {start_joints}")
        
        # Solve IK for Pose 1
        target_joints = ik_solver.solve(pose1_pos, pose1_orn, start_joints, max_iters=50, tolerance=0.001)
        
        if target_joints is not None:
            print(f"Target joint angles: {target_joints}")
            
            # Generate trajectory from current position to Pose 1
            print("Generating trajectory...")
            trajectory = generate_trajectory(start_joints, target_joints, steps=100)
            
            if trajectory:
                print(f"Generated trajectory with {len(trajectory)} points")
                
                # Execute trajectory
                print("Executing trajectory...")
                for joint_target in trajectory:
                    sim.robot.position_control(joint_target)
                    for _ in range(1):
                        sim.step()
                        time.sleep(1/240.)
                
                print("Robot arm has moved to pre-grasp position")
                
                # ===== Open gripper =====
                print("\n========== Opening Gripper ==========")
                # According to the configuration file, gripper_idx is [9, 10], default value is [0.02, 0.02]
                # Set larger values to open the gripper
                open_gripper_width = 0.04  # Width to open the gripper
                p.setJointMotorControlArray(
                    sim.robot.id,
                    jointIndices=sim.robot.gripper_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[open_gripper_width, open_gripper_width]
                )
                
                # Wait for gripper to open
                for _ in range(int(0.5 * 240)):  # Wait 0.5 seconds
                    sim.step()
                    time.sleep(1/240.)
                
                print("Gripper opened")
                
                # ===== Move to Pose 2 (final grasp position) =====
                print("\n========== Moving to Final Grasp Position ==========")
                # Get current joint angles
                current_joints = sim.robot.get_joint_positions()
                
                # Solve IK for Pose 2
                pose2_target_joints = ik_solver.solve(pose2_pos, pose2_orn, current_joints, max_iters=50, tolerance=0.001)
                
                if pose2_target_joints is not None:
                    print(f"Pose 2 target joint angles: {pose2_target_joints}")
                    
                    # Generate trajectory from Pose 1 to Pose 2
                    print("Generating trajectory to Pose 2...")
                    pose2_trajectory = generate_trajectory(current_joints, pose2_target_joints, steps=50)
                    
                    if pose2_trajectory:
                        print(f"Generated trajectory with {len(pose2_trajectory)} points")
                        
                        # Execute trajectory
                        print("Executing trajectory...")
                        for joint_target in pose2_trajectory:
                            sim.robot.position_control(joint_target)
                            for _ in range(1):
                                sim.step()
                                time.sleep(1/240.)
                        
                        print("Robot arm has moved to final grasp position")
                        
                        # ===== Close gripper to grasp =====
                        print("\n========== Closing Gripper to Grasp Object ==========")
                        # Set smaller values to close the gripper
                        close_gripper_width = 0.01  # Width to close the gripper
                        p.setJointMotorControlArray(
                            sim.robot.id,
                            jointIndices=sim.robot.gripper_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=[close_gripper_width, close_gripper_width]
                        )
                        
                        # Wait for gripper to close
                        for _ in range(int(2.0 * 240)):  # Wait 1 second
                            sim.step()
                            time.sleep(1/240.)
                        
                        print("Gripper closed, object grasped")
                        
                        # ===== Lift object after grasping =====
                        print("\n========== Lifting Object ==========")
                        # Get current end effector position and orientation
                        current_ee_pos, current_ee_orn = sim.robot.get_ee_pose()
                        print(f"Current end effector position: {current_ee_pos}")
                        
                        # Calculate position after lifting (in world coordinate system z-axis direction)
                        lift_height = 0.5  # Lift height (meters)
                        lift_pos = current_ee_pos.copy()
                        lift_pos[2] += lift_height  # Increase height in z-axis direction
                        
                        print(f"Target position after lifting: {lift_pos}")
                        
                        # Get current joint angles
                        current_joints = sim.robot.get_joint_positions()
                        
                        # Solve IK for lift position
                        lift_target_joints = ik_solver.solve(lift_pos, current_ee_orn, current_joints, max_iters=50, tolerance=0.001)
                        
                        if lift_target_joints is not None:
                            print(f"Target joint angles for lift position: {lift_target_joints}")
                            
                            # Generate trajectory from current position to lift position
                            print("Generating lift trajectory...")
                            lift_trajectory = generate_trajectory(current_joints, lift_target_joints, steps=100)
                            
                            if lift_trajectory:
                                print(f"Generated lift trajectory with {len(lift_trajectory)} points")
                                
                                # Execute trajectory
                                print("Executing lift trajectory...")
                                for joint_target in lift_trajectory:
                                    sim.robot.position_control(joint_target)
                                    for _ in range(1):
                                        sim.step()
                                        time.sleep(1/240.)
                                
                                print("Object successfully lifted")
                            else:
                                print("Unable to generate lift trajectory")
                        else:
                            print("Unable to solve IK for lift position, cannot lift object")
                    else:
                        print("Unable to generate trajectory to Pose 2")
                else:
                    print("Unable to solve IK for Pose 2, cannot move to final grasp position")
            else:
                print("Unable to generate trajectory")
        else:
            print("Unable to solve IK, cannot move to pre-grasp position")
    else:
        print("No valid grasp found!")

    # Display visualization results
    print("Displaying grasp visualization results...")
    utils.visualize_3d_objs(vis_meshes)

def run_planning(config, sim):
    """
    Use RRT* planning to plan from grasping position to tray position
    
    Args:
        config: Configuration dictionary
        sim: Simulation object
    """
    print("\n========== Starting Path Planning ==========")
    
    # Initialize obstacle tracker
    obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
    
    # Use static camera to get obstacle positions
    print("Getting obstacle information...")
    rgb_static, depth_static, seg_static = sim.get_static_renders()
    detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
    tracked_positions = obstacle_tracker.update(detections)
    
    # Visualize obstacle bounding boxes
    bounding_box_ids = obstacle_tracker.visualize_tracking_3d(tracked_positions)
    if bounding_box_ids:
            for debug_line in bounding_box_ids:
                p.removeUserDebugItem(debug_line)
                
    print(f"Detected {len(tracked_positions)} obstacles")
    
    # Get current robot position (grasping position)
    current_joints = sim.robot.get_joint_positions()
    current_ee_pos, current_ee_orn = sim.robot.get_ee_pose()
    print(f"Current end effector position: {current_ee_pos}")
    
    # Calculate tray position (target position)
    min_lim, max_lim = sim.goal._get_goal_lims()
    goal_pos = np.array([
        (min_lim[0] + max_lim[0])/2,
        (min_lim[1] + max_lim[1])/2,
        max_lim[2] + 0.2
    ])
    goal_pos[0] -= 0.2
    goal_pos[1] -= 0.2
    print(f"Tray target position: {goal_pos}")
    
    # Visualize tray target position in PyBullet
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.03,  # 3cm radius sphere
        rgbaColor=[0, 0, 1, 0.7]  # Blue semi-transparent
    )
    goal_marker_id = p.createMultiBody(
        baseMass=0,  # Mass of 0 indicates a static object
        baseVisualShapeIndex=visual_id,
        basePosition=goal_pos.tolist()
    )
    
    # Add tray target position coordinate axes
    axis_length = 0.1  # 10cm long axes
    p.addUserDebugLine(
        goal_pos, 
        goal_pos + np.array([axis_length, 0, 0]), 
        [1, 0, 0], 3, 0  # X-axis - Red
    )
    p.addUserDebugLine(
        goal_pos, 
        goal_pos + np.array([0, axis_length, 0]), 
        [0, 1, 0], 3, 0  # Y-axis - Green
    )
    p.addUserDebugLine(
        goal_pos, 
        goal_pos + np.array([0, 0, axis_length]), 
        [0, 0, 1], 3, 0  # Z-axis - Blue
    )
    
    # Add tray target position text label
    p.addUserDebugText(
        f"Target position ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f})",
        goal_pos + np.array([0, 0, 0.05]),  # Display text 5cm above target position
        [1, 1, 1],  # White text
        1.0  # Text size
    )
    
    # Initialize IK solver
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    
    # Solve target position IK
    goal_orn = current_ee_orn  # Keep current direction
    goal_joints = ik_solver.solve(goal_pos, goal_orn, current_joints, max_iters=50, tolerance=0.001)
    
    if goal_joints is None:
        print("Unable to solve IK for target position, cannot perform path planning")
        return
    
    print("Initializing RRT* planner...")
    # Initialize RRT* planner
    rrt_planner = RRTStarPlanner(
        robot_id=sim.robot.id,
        joint_indices=sim.robot.arm_idx,
        lower_limits=sim.robot.lower_limits,
        upper_limits=sim.robot.upper_limits,
        ee_link_index=sim.robot.ee_idx,
        obstacle_tracker=obstacle_tracker,
        max_iterations=3000,  # Increase iterations for better path
        step_size=0.2,
        goal_sample_rate=0.1,  # Increase goal sampling rate
        search_radius=0.6,
        goal_threshold=0.15,
        collision_check_step=0.05
    )
    
    print("Starting RRT* path planning...")
    # Plan path
    trajectory = generate_rrt_star_trajectory(sim, rrt_planner, current_joints, goal_joints, visualize=True)
    
    if not trajectory:
        print("RRT* planning failed, trying simple joint space planning...")
        trajectory = generate_trajectory(current_joints, goal_joints, steps=200)
    
    if not trajectory:
        print("Unable to generate path to target position, planning failed")
        return
    
    print(f"Generated path with {len(trajectory)} points")
    
    # Execute path
    print("Executing planned path...")
    for joint_target in trajectory:
        # Update obstacle tracking
        rgb_static, depth_static, seg_static = sim.get_static_renders()
        detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
        tracked_positions = obstacle_tracker.update(detections)
        
        # Move robot
        sim.robot.position_control(joint_target)
        for _ in range(1):
            sim.step()
            time.sleep(1/240.)
    
    print("Robot arm has successfully moved to tray position")
    
    # Place object
    print("\n========== Placing Object ==========")
    # Open gripper
    open_gripper_width = 0.04  # Width to open the gripper
    p.setJointMotorControlArray(
        sim.robot.id,
        jointIndices=sim.robot.gripper_idx,
        controlMode=p.POSITION_CONTROL,
        targetPositions=[open_gripper_width, open_gripper_width]
    )
    
    # Wait for gripper to open
    for _ in range(int(1.0 * 240)):  # Wait 1 second
        sim.step()
        time.sleep(1/240.)
    
    print("Gripper opened, object placed on tray")
    
    # ===== Robot arm returns to grasping position =====
    print("\n========== Robot Arm Returns ==========")
    
    # Get current joint angles (tray position, now as new start)
    current_joints_at_goal = sim.robot.get_joint_positions()
    current_ee_pos_at_goal = current_ee_pos  # Use previously saved grasping position as target
    
    print(f"Returning from tray position {goal_pos} to grasping position {current_ee_pos}")
    
    # Again use RRT* planning to plan from tray position back to grasping position
    print("Starting return path planning...")
    
    # Update obstacle positions
    rgb_static, depth_static, seg_static = sim.get_static_renders()
    detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
    tracked_positions = obstacle_tracker.update(detections)
    
    # Plan return path
    return_trajectory = generate_rrt_star_trajectory(sim, rrt_planner, current_joints_at_goal, current_joints, visualize=True)
    
    if not return_trajectory:
        print("RRT* return path planning failed, trying simple joint space planning...")
        return_trajectory = generate_trajectory(current_joints_at_goal, current_joints, steps=200)
    
    if not return_trajectory:
        print("Unable to generate return path, planning failed")
    else:
        print(f"Generated return path with {len(return_trajectory)} points")
        
        # Execute return path
        print("Executing return path...")
        for joint_target in return_trajectory:
            # Update obstacle tracking
            rgb_static, depth_static, seg_static = sim.get_static_renders()
            detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
            tracked_positions = obstacle_tracker.update(detections)
            
            # Move robot
            sim.robot.position_control(joint_target)
            for _ in range(1):
                sim.step()
                time.sleep(1/240.)
        
        print("Robot arm has successfully returned to grasping position")
    
    print("\n========== Path Planning Completed ==========")

def run_pcd(config):
    """
    main function to run point cloud collection from multiple viewpoints
    """
    print("Starting point cloud collection ...")
    
    # initialize PyBullet simulation
    
    # randomly select an object from YCB dataset
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [os.path.basename(file) for file in files]
    # target_obj_name = random.choice(obj_names)
    # print(f"Resetting simulation with random object: {target_obj_name}")
    # All objects: 
    # Low objects: YcbBanana, YcbFoamBrick, YcbHammer, YcbMediumClamp, YcbPear, YcbScissors, YcbStrawberry, YcbTennisBall, 
    # Medium objects: YcbGelatinBox, YcbMasterChefCan, YcbPottedMeatCan, YcbTomatoSoupCan
    # High objects: YcbCrackerBox, YcbMustardBottle, 
    # Unstable objects: YcbChipsCan, YcbPowerDrill
    target_obj_name = "YcbGelatinBox" 
    
    # reset simulation with target object
    sim.reset(target_obj_name)
    
    # Initialize obstacle tracker
    obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
    
    # Initialize point cloud collection list
    collected_data = []
    
    # Get and save initial position of simulation
    initial_joints = sim.robot.get_joint_positions()
    print("Saving initial simulation joint positions")
    
    # Initialize object height variable, default value
    object_height_with_offset = 1.6
    # Initialize object centroid coordinates, default value
    object_centroid_x = -0.02
    object_centroid_y = -0.45

    pause_time = 2.0  # Pause for 2 seconds
    print(f"\nPause {pause_time} seconds...")
    for _ in range(int(pause_time * 240)):  # Assume simulation frequency is 240Hz
        sim.step()
        time.sleep(1/240.)
        
    # ===== Move to specified position and get point cloud =====
    print("\nMoving to high point observation position...")
    # Define high point observation position and orientation
    z_observe_pos = np.array([-0.02, -0.45, 1.9])
    z_observe_orn = p.getQuaternionFromEuler([0, np.radians(-180), 0])  # Looking down
    
    # Solve IK
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    high_point_target_joints = ik_solver.solve(z_observe_pos, z_observe_orn, initial_joints, max_iters=50, tolerance=0.001)
    
    # Generate trajectory
    print("Generating trajectory for high point observation position...")
    # high_point_trajectory = generate_cartesian_trajectory(sim, ik_solver, initial_joints, z_observe_pos, z_observe_orn, steps=100)
    high_point_trajectory = generate_trajectory(initial_joints, high_point_target_joints, steps=100)
    if not high_point_trajectory:
        print("Unable to generate trajectory to high point observation position, skipping high point cloud collection")
    else:
        print(f"Generated trajectory with {len(high_point_trajectory)} points")
        
        # Reset to initial position
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, initial_joints[i])
        
        # Move robot along trajectory to high point
        for joint_target in high_point_trajectory:
            # sim.get_ee_renders()
            sim.robot.position_control(joint_target)
            for _ in range(1):
                sim.step()
                time.sleep(1/240.)
        
        # Get point cloud at high point observation position
        rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
        camera_pos, camera_R = get_ee_camera_params(sim.robot, config)
        print(f"High point observation position camera position:", camera_pos)
        print(f"High point observation position end effector position:", sim.robot.get_ee_pose()[0])
        
        # Build point cloud
        target_mask_id = sim.object.id
        print(f"Target object ID: {target_mask_id}")
        
        try:
            if target_mask_id not in np.unique(seg_ee):
                print("Warning: Target object ID not found in segmentation mask.")
                print("Available IDs in segmentation mask:", np.unique(seg_ee))
                
                non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                if len(non_zero_ids) > 0:
                    target_mask_id = non_zero_ids[0]
                    print(f"Using first non-zero ID instead: {target_mask_id}")
                else:
                    raise ValueError("No valid objects found in segmentation mask")
            
            high_point_pcd = build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, config, camera_pos, camera_R)
            
            # Process point cloud
            high_point_pcd = high_point_pcd.voxel_down_sample(voxel_size=0.005)
            high_point_pcd, _ = high_point_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Store point cloud data
            high_point_cloud_data = {
                'point_cloud': high_point_pcd,
                'camera_position': camera_pos,
                'camera_rotation': camera_R,
                'ee_position': sim.robot.get_ee_pose()[0],
                'timestamp': time.time(),
                'target_object': target_obj_name,
                'viewpoint_idx': 'high_point'
            }
            
            # Get coordinates of all points in the cloud
            points_array = np.asarray(high_point_pcd.points)
            if len(points_array) > 0:
                # Find the point with maximum z-axis value
                max_z_idx = np.argmax(points_array[:, 2])
                max_z_point = points_array[max_z_idx]
                print(f"Maximum z-axis point in high point cloud: {max_z_point}")
                high_point_cloud_data['max_z_point'] = max_z_point
                
                # Extract maximum z-axis value, add offset
                object_max_z = max_z_point[2]
                object_height_with_offset = max(object_max_z + 0.2, 1.65)
                print(f"Object height with offset: {object_height_with_offset}")
                
                # Calculate centroid of x and y coordinates for all points in the cloud
                object_centroid_x = np.mean(points_array[:, 0])
                object_centroid_y = np.mean(points_array[:, 1])
                print(f"Object point cloud centroid coordinates (x, y): ({object_centroid_x:.4f}, {object_centroid_y:.4f})")
                high_point_cloud_data['centroid'] = np.array([object_centroid_x, object_centroid_y, 0])
            else:
                print("No points in high point cloud")
            
            # # Visualize high point cloud
            # print("\nVisualizing high point cloud...")
            # visualize_point_clouds([high_point_cloud_data], show_merged=False)
            
            # Add high point cloud to collected data
            collected_data.append(high_point_cloud_data)
            print(f"Collected {len(high_point_pcd.points)} points from high observation position")
            
        except ValueError as e:
            print(f"Error building point cloud for high observation position:", e)

    target_positions = [
        # option 1:
        np.array([object_centroid_x + 0.15, object_centroid_y, object_height_with_offset]),
        np.array([object_centroid_x, object_centroid_y + 0.15, object_height_with_offset]),
        np.array([object_centroid_x - 0.15, object_centroid_y, object_height_with_offset]),
        np.array([object_centroid_x, object_centroid_y - 0.15, object_height_with_offset]),
        # # option 2:
        # np.array([object_centroid_x + 0.1, object_centroid_y + 0.1, object_height_with_offset]),
        # np.array([object_centroid_x - 0.1, object_centroid_y + 0.1, object_height_with_offset]),
        # np.array([object_centroid_x - 0.1, object_centroid_y - 0.1, object_height_with_offset]),
        # np.array([object_centroid_x + 0.1, object_centroid_y - 0.1, object_height_with_offset]),
    ]
    target_orientations = [
        # option 1:
        p.getQuaternionFromEuler([0, np.radians(-150), 0]),
        p.getQuaternionFromEuler([np.radians(150), 0, 0]),
        p.getQuaternionFromEuler([0, np.radians(150), 0]),
        p.getQuaternionFromEuler([np.radians(-150), 0, 0]),
        # # option 2:
        # p.getQuaternionFromEuler([0, np.radians(-150), np.radians(45)]),
        # p.getQuaternionFromEuler([np.radians(150), 0, np.radians(45)]),
        # p.getQuaternionFromEuler([0, np.radians(150), np.radians(45)]),
        # p.getQuaternionFromEuler([np.radians(-150), 0, np.radians(45)]),
        
    ]
    
    print(f"\nUsing based on object centroid collection positions:")
    print(f"Object centroid coordinates (x, y): ({object_centroid_x:.4f}, {object_centroid_y:.4f})")
    print(f"Object height with offset: {object_height_with_offset:.4f}")
    for i, pos in enumerate(target_positions):
        print(f"Collection point {i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    # For each viewpoint
    for viewpoint_idx, (target_pos, target_orn) in enumerate(zip(target_positions, target_orientations)):
        print(f"\nMoving to viewpoint {viewpoint_idx + 1}")
        sim.get_ee_renders()
        # Get initial static camera view to setup obstacle tracking
        # rgb_static, depth_static, seg_static = sim.get_static_renders()
        # detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
        # tracked_positions = obstacle_tracker.update(detections)
        
        # Get current joint positions
        current_joints = sim.robot.get_joint_positions()
        # Save current joint positions
        saved_joints = current_joints.copy()
        
        # Solve IK for target end-effector pose
        ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
        target_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.001)
        
        # Reset to saved start position
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, saved_joints[i])
        
        # # Initialize RRT* planner
        # rrt_planner = RRTStarPlanner(
        #     robot_id=sim.robot.id,
        #     # joint_indices=ik_solver.joint_indices,
        #     joint_indices=sim.robot.arm_idx,
        #     lower_limits=sim.robot.lower_limits,
        #     upper_limits=sim.robot.upper_limits,
        #     ee_link_index=sim.robot.ee_idx,
        #     obstacle_tracker=obstacle_tracker,
        #     max_iterations=1000,
        #     step_size=0.2,
        #     goal_sample_rate=0.05,
        #     search_radius=0.5,
        #     goal_threshold=0.1,
        #     collision_check_step=0.05
        # )
        
        choice = 2  # Change this to test different methods
        
        trajectory = []
        if choice == 1:
            print("Generating linear Cartesian trajectory...")
            trajectory = generate_cartesian_trajectory(sim, ik_solver, saved_joints, target_pos, target_orn, steps=100)
        elif choice == 2:
            print("Generating linear joint space trajectory...")
            trajectory = generate_trajectory(saved_joints, target_joints, steps=100)
        
        if not trajectory:
            print(f"Failed to generate trajectory for viewpoint {viewpoint_idx + 1}. Skipping...")
            continue
        
        print(f"Generated trajectory with {len(trajectory)} points")
        
        # Reset to saved start position again before executing trajectory
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, saved_joints[i])
        
        # Move robot along trajectory to target position
        for joint_target in trajectory:
            # sim.get_ee_renders()
            # Update obstacle tracking
            # rgb_static, depth_static, seg_static = sim.get_static_renders()
            # detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
            # tracked_positions = obstacle_tracker.update(detections)
            
            # Visualize tracked obstacles
            # bounding_box = obstacle_tracker.visualize_tracking_3d(tracked_positions)
            # if bounding_box:
            #     for debug_line in bounding_box:
            #         p.removeUserDebugItem(debug_line)
            
            # Move robot
            sim.robot.position_control(joint_target)
            for _ in range(1):
                sim.step()
                time.sleep(1/240.)
        
        # Capture point cloud at this viewpoint
        rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
        camera_pos, camera_R = get_ee_camera_params(sim.robot, config)
        print(f"Viewpoint {viewpoint_idx + 1} camera position:", camera_pos)
        print(f"Viewpoint {viewpoint_idx + 1} end effector position:", sim.robot.get_ee_pose()[0])
        
        # Build point cloud
        target_mask_id = sim.object.id
        print(f"Target object ID: {target_mask_id}")
        
        try:
            if target_mask_id not in np.unique(seg_ee):
                print("Warning: Target object ID not found in segmentation mask.")
                print("Available IDs in segmentation mask:", np.unique(seg_ee))
                
                non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                if len(non_zero_ids) > 0:
                    target_mask_id = non_zero_ids[0]
                    print(f"Using first non-zero ID instead: {target_mask_id}")
                else:
                    raise ValueError("No valid objects found in segmentation mask")
            
            pcd_ee = build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, config, camera_pos, camera_R)
            
            # Process point cloud
            pcd_ee = pcd_ee.voxel_down_sample(voxel_size=0.005)
            pcd_ee, _ = pcd_ee.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Store point cloud data
            point_cloud_data = {
                'point_cloud': pcd_ee,
                'camera_position': camera_pos,
                'camera_rotation': camera_R,
                'ee_position': sim.robot.get_ee_pose()[0],
                'timestamp': time.time(),
                'target_object': target_obj_name,
                'viewpoint_idx': viewpoint_idx
            }
            collected_data.append(point_cloud_data)
            print(f"Point cloud collected from viewpoint {viewpoint_idx + 1} with {len(pcd_ee.points)} points.")
            
        except ValueError as e:
            print(f"Error building point cloud for viewpoint {viewpoint_idx + 1}:", e)
    
    return collected_data

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    # Run simulation and collect point clouds
    sim = Simulation(config)
    collected_point_clouds = run_pcd(config)
    print(f"Successfully collected {len(collected_point_clouds)} point clouds.")
    
    # Check and print maximum z-axis point in high point cloud
    for data in collected_point_clouds:
        if data.get('viewpoint_idx') == 'high_point' and 'max_z_point' in data:
            print(f"\nMaximum z-axis point in high point cloud: {data['max_z_point']}")
    
    # Visualize the collected point clouds if any were collected
    # if collected_point_clouds:
    #     # First show individual point clouds
    #     print("\nVisualizing individual point clouds...")
    #     visualize_point_clouds(collected_point_clouds, show_merged=False)
        
    #     # Then show merged point cloud
    #     print("\nVisualizing merged point cloud...")
    #     visualize_point_clouds(collected_point_clouds, show_merged=True)
        
        # Execute grasp generation
        print("\nExecuting grasp generation...")
        run_grasping(config, sim, collected_point_clouds)
        
        # Execute path planning
        print("\nExecuting path planning...")
        run_planning(config, sim)
        
        # Close simulation
        sim.close()