import os
import glob
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.perception.pose_estimation import PoseEstimation
from src.perception.position_estimation import PositionEstimation
import src.perception.object_detection as object_detection
from src.controllers.ik_controller import IKController

import pybullet as p
import cv2


def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    # for obj_name in obj_names:
    for obj_name in ["YcbHammer", "YcbPowerDrill", "YcbBanana", "YcbStrawberry"]:
        for tstep in range(10):
            sim.reset(obj_name)
            print((f"Object: {obj_name}, Timestep: {tstep},"
                   f" pose: {sim.get_ground_tuth_position_object}"))
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot inital pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocites: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")

            robot = sim.get_robot()
            ik_controller = IKController(
                    robot_id=robot.id,
                    joint_indices=robot.arm_idx,
                    ee_index=robot.ee_idx
                )

            target_pos, _ = robot.get_ee_pose()
            target_pos = target_pos + np.array([0.0, 0.0, 0.7])
            # move the robot out of the way so that the camera can view the object properly
            joint_positions = ik_controller.solve_ik(
                target_pos,
                max_iters=10,     
                tolerance=1e-2)
            
            # control robot
            robot.position_control(joint_positions)
 
            for i in range(10000):
                sim.step()
                ee_pos, ee_ori = sim.robot.get_ee_pose()
                print(f"[{i}] End Effector Position: {ee_pos}")
                print(f"[{i}] End Effector Orientation: {ee_ori}")
                
                if sim.obstacles_flag: # obstacle set to True in config
                    obs_position_guess = sim.predicted_positions
                else: 
                    obs_position_guess = np.zeros((2, 3))
                    
                print(f"[{i}] Obstacle Position-Diff: {sim.check_obstacle_position(obs_position_guess)}")
                # for getting renders
                # rgb, depth, seg = sim.get_ee_renders()
                # rgb, depth, seg = sim.get_static_renders()
                goal_guess = np.zeros((7,))

                (rgb, depth, seg) = sim.get_static_renders()

                unique_ids = np.unique(seg)
                id_to_label = {}
                for obj_id in unique_ids:
                    if obj_id >= 0:  # Ignore background (-1)
                        body_info = p.getBodyInfo(obj_id)
                        body_name = body_info[1].decode('utf-8')  # Decode name from bytes
                        id_to_label[obj_id] = body_name
                for obj_id, label in id_to_label.items():
                    print(f"ID: {obj_id}, Label: {label}")

                segmented_depth = object_detection.segment_depth_image(
                    depth, 
                    object_detection.segmentation_mask(seg, id=5)
                    )

                position_est = PositionEstimation(config["world_settings"]["camera"])
                posi = position_est.determine_position(segmented_depth)
                print(f"the position of the object is {posi}")


                # camera_projection_matrix = np.array(sim.projection_matrix).reshape(4, 4)
                # camera_projection_matrix = np.delete(camera_projection_matrix, 2, axis=0)

                # position_est = PositionEstimation(config["camera"], camera_projection_matrix)
                # camera_intrinsics = position_est.compute_camera_intrinsics()
                # print(f"the camera intrinsics are {camera_intrinsics}")

                # print(f"the projection matrix is {camera_projection_matrix}")

                # cv2.imwrite("mask.jpg", segmentation_mask*100)
                # cv2.imwrite("depth.jpg", segmented_depth*10)

                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)
