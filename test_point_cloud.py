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
from src.grasping.grasp_generation import GraspGeneration
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
    
    Parameters:
    config: Configuration dictionary
    sim: Simulation object
    collected_point_clouds: List of collected point cloud data
    """
    print("Merging point clouds and calculating centroid...")
    merged_pcd = iterative_closest_point(collected_point_clouds)
    centre_point = np.asarray(merged_pcd.points)
    centre_point = centre_point.mean(axis=0)
    print(f"Point cloud centroid coordinates: {centre_point}")
    
    # Initialize IK solver
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    
    # Get current joint positions
    current_joints = sim.robot.get_joint_positions()
    
    # Initialize grasp generator
    print("Generating grasp candidates...")
    grasp_generator = GraspGeneration()
    sampled_grasps = grasp_generator.sample_grasps(centre_point, 50, offset=0.1)
    
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
        if not grasp_generator.check_grasp_collision(grasp_mesh, merged_pcd, num_colisions=1):
            valid_grasp, grasp_quality = grasp_generator.check_grasp_containment(
                grasp_mesh[0].get_center(), 
                grasp_mesh[1].get_center(),
                finger_length=0.05,
                object_pcd=merged_pcd,
                num_rays=50,
                rotation_matrix=pose[0],
            )
            # Convert tensor to float if needed
            if hasattr(grasp_quality, 'item'):
                grasp_quality = float(grasp_quality.item())
            
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
    else:
        print("No valid grasp found!")

    # Display visualization results
    print("Displaying grasp visualization results...")
    utils.visualize_3d_objs(vis_meshes)
    
    # If valid grasp found, move robot to grasp position
    if best_grasp is not None:
        rot_mat, translation = best_grasp
        goal_pos = merged_pcd.get_center() + translation
        print(f"Target position: {goal_pos}")
        rot = Rotation.from_matrix(rot_mat)
        rot_quat = rot.as_quat()
        
        # Solve IK to get joint targets
        joint_goals = ik_solver.solve(merged_pcd.get_center(), rot_quat, sim.robot.get_joint_positions())
        
        # Move robot to grasp position
        print("Moving robot to grasp position...")
        sim.robot.position_control(joint_goals)
        
        # Open gripper
        print("Opening gripper...")
        sim.robot.control_gripper()
        
        # Wait for physics to stabilize
        for _ in range(100):
            sim.step()
            time.sleep(1/240.)

def run(config):
    """
    main function to run point cloud collection from multiple viewpoints
    """
    print("Starting point cloud collection ...")
    
    # initialize PyBullet simulation
    sim = Simulation(config)
    
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
    target_obj_name = "YcbMustardBottle" 
    
    # reset simulation with target object
    sim.reset(target_obj_name)
    
    # Initialize obstacle tracker
    obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
    
    # Initialize point cloud collection list
    collected_data = []
    
    # Get and save initial position at the start of simulation
    initial_joints = sim.robot.get_joint_positions()
    print("Saving initial joint positions of simulation environment")
    
    # Initialize object height variable, default value
    object_height_with_offset = 1.6
    # Initialize object centroid coordinates, default values
    object_centroid_x = -0.02
    object_centroid_y = -0.45

    pause_time = 2.0  # Pause for 2 seconds
    print(f"\nPausing for {pause_time} seconds...")
    for _ in range(int(pause_time * 240)):  # Assuming simulation frequency of 240Hz
        sim.step()
        time.sleep(1/240.)
        
    # ===== Move to specified position and get point cloud =====
    print("\nMoving to high observation point...")
    # Define high observation position and orientation
    z_observe_pos = np.array([-0.02, -0.45, 1.9])
    z_observe_orn = p.getQuaternionFromEuler([0, np.radians(-180), 0])  # Looking down
    
    # Solve IK
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    high_point_target_joints = ik_solver.solve(z_observe_pos, z_observe_orn, initial_joints, max_iters=50, tolerance=0.001)
    
    # Generate trajectory
    print("Generating trajectory for high observation point...")
    high_point_trajectory = generate_cartesian_trajectory(sim, ik_solver, initial_joints, z_observe_pos, z_observe_orn, steps=100)
    
    if not high_point_trajectory:
        print("Unable to generate trajectory to high observation point, skipping high point cloud collection")
    else:
        print(f"Generated trajectory with {len(high_point_trajectory)} points")
        
        # Reset to initial position
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, initial_joints[i])
        
        # Move robot along trajectory to high point
        for joint_target in high_point_trajectory:
            # sim.get_ee_renders()
            sim.robot.position_control(joint_target)
            for _ in range(5):
                sim.step()
                time.sleep(1/240.)
        
        # Get point cloud at high observation position
        rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
        camera_pos, camera_R = get_ee_camera_params(sim.robot, config)
        print(f"High observation point camera position:", camera_pos)
        print(f"High observation point end effector position:", sim.robot.get_ee_pose()[0])
        
        # Build point cloud
        target_mask_id = sim.object.id
        print(f"Target object ID: {target_mask_id}")
        
        try:
            if target_mask_id not in np.unique(seg_ee):
                print("Warning: Target object ID not found in segmentation mask")
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
            
            # Get coordinates of all points in the point cloud
            points_array = np.asarray(high_point_pcd.points)
            if len(points_array) > 0:
                # Find point with maximum z value
                max_z_idx = np.argmax(points_array[:, 2])
                max_z_point = points_array[max_z_idx]
                print(f"Maximum z-axis point in high point cloud: {max_z_point}")
                high_point_cloud_data['max_z_point'] = max_z_point
                
                # Extract maximum z value, add offset
                object_max_z = max_z_point[2]
                object_height_with_offset = max(object_max_z + 0.2, 1.65)
                print(f"Object height with offset: {object_height_with_offset}")
                
                # Calculate centroid of x and y coordinates of all points in the cloud
                object_centroid_x = np.mean(points_array[:, 0])
                object_centroid_y = np.mean(points_array[:, 1])
                print(f"Object point cloud centroid coordinates (x, y): ({object_centroid_x:.4f}, {object_centroid_y:.4f})")
                high_point_cloud_data['centroid'] = np.array([object_centroid_x, object_centroid_y, 0])
            else:
                print("No points in high point cloud")
            
            # Visualize high point cloud
            print("\nVisualizing high point cloud...")
            visualize_point_clouds([high_point_cloud_data], show_merged=False)
            
            # Add high point cloud to collected data
            collected_data.append(high_point_cloud_data)
            print(f"Point cloud collected from high observation position has {len(high_point_pcd.points)} points")
            
        except ValueError as e:
            print(f"Error building point cloud for high observation position:", e)
        
    #     # Return from high point to initial position
    #     print("\nReturning from high point to initial position...")
    #     # Generate trajectory from high point back to initial position
    #     return_trajectory = generate_trajectory(sim.robot.get_joint_positions(), initial_joints, steps=100)
        
    #     if not return_trajectory:
    #         print("Unable to generate trajectory back to initial position")
    #     else:
    #         print(f"Generated return trajectory with {len(return_trajectory)} points")
            
    #         # Move robot along trajectory back to initial position
    #         for joint_target in return_trajectory:
    #             sim.robot.position_control(joint_target)
    #             for _ in range(1):
    #                 sim.step()
    #                 time.sleep(1/240.)
            
    #         print("Returned to initial position")
    
    # # Ensure robot is back at initial position
    # for i, joint_idx in enumerate(sim.robot.arm_idx):
    #     p.resetJointState(sim.robot.id, joint_idx, initial_joints[i])
    
    # ===== Original 4 point cloud collection positions =====
    # Define target positions and orientations
    # Use object centroid as base point for x and y, add offsets
    target_positions = [
        
        np.array([object_centroid_x + 0.15, object_centroid_y, object_height_with_offset]),
        np.array([object_centroid_x, object_centroid_y + 0.15, object_height_with_offset]),
        np.array([object_centroid_x - 0.15, object_centroid_y, object_height_with_offset]),
        np.array([object_centroid_x, object_centroid_y - 0.15, object_height_with_offset])
        
    ]
    target_orientations = [
        
        p.getQuaternionFromEuler([0, np.radians(-150), 0]),
        p.getQuaternionFromEuler([np.radians(150), 0, 0]),
        p.getQuaternionFromEuler([0, np.radians(150), 0]),
        p.getQuaternionFromEuler([np.radians(-150), 0, 0])
        
    ]
    
    print(f"\nUsing collection positions based on object centroid:")
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
        
        # Initialize RRT* planner
        rrt_planner = RRTStarPlanner(
            robot_id=sim.robot.id,
            # joint_indices=ik_solver.joint_indices,
            joint_indices=sim.robot.arm_idx,
            lower_limits=sim.robot.lower_limits,
            upper_limits=sim.robot.upper_limits,
            ee_link_index=sim.robot.ee_idx,
            obstacle_tracker=obstacle_tracker,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.05,
            search_radius=0.5,
            goal_threshold=0.1,
            collision_check_step=0.05
        )
        
        choice = 2  # Change this to test different methods
        
        trajectory = []
        if choice == 1:
            print("Generating linear Cartesian trajectory...")
            trajectory = generate_cartesian_trajectory(sim, ik_solver, saved_joints, target_pos, target_orn, steps=100)
        elif choice == 2:
            print("Generating linear joint space trajectory...")
            trajectory = generate_trajectory(saved_joints, target_joints, steps=100)
        else:
            print("Generating RRT* trajectory...")
            trajectory = generate_rrt_star_trajectory(sim, rrt_planner, saved_joints, target_joints)
        
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
            for _ in range(5):
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
    
    # sim.close()
    return collected_data, sim

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    # Run simulation and collect point clouds
    collected_point_clouds, sim = run(config)
    print(f"Successfully collected {len(collected_point_clouds)} point clouds.")
    
    # Check and print maximum z-axis point of high point cloud
    for data in collected_point_clouds:
        if data.get('viewpoint_idx') == 'high_point' and 'max_z_point' in data:
            print(f"\nMaximum z-axis point in high observation position point cloud: {data['max_z_point']}")
    
    # Visualize the collected point clouds if any were collected
    if collected_point_clouds:
        # First show individual point clouds
        print("\nVisualizing individual point clouds...")
        visualize_point_clouds(collected_point_clouds, show_merged=False)
        
        # Then show merged point cloud
        print("\nVisualizing merged point cloud...")
        visualize_point_clouds(collected_point_clouds, show_merged=True)
        
        # Execute grasp generation
        print("\nExecuting grasp generation...")
        run_grasping(config, sim, collected_point_clouds)
        
        # Close simulation
        sim.close()