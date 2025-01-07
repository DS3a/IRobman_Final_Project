import os
import glob
import yaml
import time
import numpy as np
from typing import Dict, Any
from src.simulation import Simulation
from src.controllers.ik_controller import IKController

def test_ik_movement(sim: Simulation, duration: int = 100):
    """test IK solver by moving the end effector in a sinusoidal pattern"""
    print("\nInitializing IK movement test...")
    
    # acquire robot instance
    print("Getting robot instance...")
    robot = sim.get_robot()
    if robot is None:
        raise RuntimeError("Failed to get robot instance")
    
    # print robot configuration
    print("\nRobot Configuration:")
    print(f"Robot ID: {robot.id}")
    print(f"Joint indices: {robot.arm_idx}")
    print(f"End effector index: {robot.ee_idx}")
    
    # create IK controller
    print("\nCreating IK controller...")
    ik_controller = IKController(
        robot_id=robot.id,
        joint_indices=robot.arm_idx,
        ee_index=robot.ee_idx
    )
    
    # acquire initial end-effector position
    initial_pos, _ = robot.get_ee_pose()
    print(f"\nInitial end-effector position: {initial_pos}")
    
    # motion parameters
    amplitude = 0.2  # larger amplitude for more visible motion
    period = 50     # smaller period for faster motion
    time_step = 1.0/240.0  # default pybullet timestep in yaml config
    
    print(f"\nMovement parameters:")
    print(f"Amplitude: {amplitude}m")
    print(f"Period: {period} steps")
    
    try:
        # main loop
        for step in range(duration):
            start_time = time.time()
            
            # generate target position
            phase = step * 2 * np.pi / period
            target_pos = initial_pos.copy()
            target_pos[0] += amplitude * np.sin(phase)  # X axis motion
            target_pos[1] += amplitude * 0.5 * np.cos(phase)  # Y axis motion
            
            # solver IK
            joint_positions = ik_controller.solve_ik(
                target_pos,
                max_iters=10,     
                tolerance=1e-2     
            )
            
            # control robot
            robot.position_control(joint_positions)
            
            # step simulation
            sim.step()
            
            # print status
            if step % 50 == 0:
                current_pos, _ = robot.get_ee_pose()
                print(f"\nStep {step}:")
                print(f"Target position: {target_pos}")
                print(f"Current position: {current_pos}")
                error = np.linalg.norm(target_pos - current_pos)
                print(f"Position error: {error:.4f}m")
            
            # control sim step rate
            elapsed = time.time() - start_time
            if elapsed < time_step:
                time.sleep(time_step - elapsed)
                
    except Exception as e:
        print(f"\nError during IK movement test: {str(e)}")
        raise

def run_exp(config: Dict[str, Any]):
    print("Starting IK Test Simulation...")
    
    # show GUI
    config['world_settings']['mode'] = 1
    
    # create simulation instance
    sim = Simulation(config)
    
    # init simulation
    print("\nInitializing simulation...")
    sim.reset()
    
    # wait for simulation to stabilize
    print("Waiting for simulation to stabilize...")
    for _ in range(10):
        sim.step()
        time.sleep(1.0/240.0)  # default pybullet timestep in yaml config
    
    # run IK movement test
    try:
        test_ik_movement(sim)
    except Exception as e:
        print(f"Error in experiment: {str(e)}")
    finally:
        # close simulation
        print("\nTest completed. Keeping simulation window open for 5 seconds...")
        time.sleep(5)
        sim.close()

if __name__ == "__main__":
    # load configuration
    config_path = "configs/test_config.yaml"
    print(f"\nLoading configuration from {config_path}")
    
    try:
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        raise
        
    # run experiment
    run_exp(config)