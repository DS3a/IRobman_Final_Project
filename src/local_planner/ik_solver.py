import numpy as np
import pybullet as p
from typing import List, Tuple, Optional
from src.local_planner.panda_forward_dynamics.velocity_inputs.system_model import SystemModel
from scipy.spatial.transform import Rotation as R

class IKController:
    """Inverse Kinematics Controller using Pseudo-inverse Method"""
    
    def __init__(self, robot_id: int, joint_indices: List[int], ee_index: int):
        """
        Args:
            robot_id: PyBullet robot ID
            joint_indices: list of joint indices to control
            ee_index: end effector link index
        """
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.ee_index = ee_index
        self.num_joints = len(joint_indices)
        self.damping = 0.001  # damping factor for pseudo-inverse
        
        # print init message
        print(f"Initializing IK Controller:")
        print(f"Robot ID: {robot_id}")
        print(f"Joint indices: {joint_indices}")
        print(f"End effector index: {ee_index}")
        print(f"Number of controlled joints: {self.num_joints}")
        

        self.system_model = SystemModel()
        self._verify_robot_setup()
    
    def _verify_robot_setup(self):
        """verify robot setup"""
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Total number of joints in robot: {num_joints}")
        
        # print joint information
        print("\nJoint Information:")
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")
            
        # verify end effector index
        if self.ee_index >= num_joints:
            raise ValueError(f"End effector index {self.ee_index} is out of range. Max joint index is {num_joints-1}")
    
    def _get_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix using numerical differentiation
        """
        delta = 1e-6  # small value for numerical differentiation
        jac = np.zeros((6, len(self.joint_indices)))  # 6×n jacobi matrix
        
        # save original joint positions
        original_positions = np.copy(joint_positions.copy())
        print(f"THE original positions are {original_positions}")
        
        # acquire current end effector pose
        current_pos, current_orn = self._get_current_ee_pose()
        
        # partial differentiation for each joint
        for i in range(len(self.joint_indices)):
            # perturb joint position
            joint_positions_ = np.copy(original_positions)
            joint_positions_[i] += delta
            
           
            # TODO use the forward kinematics function instead of this to find the new position and orientation
            # update joint states



            # frames = self.system_model.forward_kinematics_fast(joint_positions)
            # ee_frame = frames[6]
            # numpy_matrix = joint_positions.reshape((7, 1))
            # sympy_matrix = self.system_model.X
        
            # mapping = {sympy_matrix[i, 0]: numpy_matrix[i, 0] for i in range(7)}
 
            # ee_frame = ee_frame.subs(mapping)


            for idx, pos in zip(self.joint_indices, joint_positions_):
                p.resetJointState(self.robot_id, idx, pos)
                # print(f"ORIGINAL joints {original_positions} current {joint_positions}")



            # new_pos = np.array(ee_frame[:3, 3])
            # new_rot_mat = np.array(ee_frame[:3, :3])
            # rot_mat = R.from_matrix(new_rot_mat)
            # new_orn = rot_mat.as_quat()


            # acquire new end effector pose
            new_pos, new_orn = self._get_current_ee_pose()
            
            # print(f"THE new frame is {ee_frame[0, 0:3]}")
            # print(f"VS newpos fwd {new_pos_}, {new_pos}")
            # position jacobian
            jac[:3, i] = (new_pos - current_pos) / delta
            
            # orientation jacobian
            # quaternion difference as angular velocity
            orn_diff = self._quat_difference(current_orn, new_orn)
            jac[3:, i] = np.array(orn_diff[:3]) / delta  # only use imaginary part of quaternion
            
        # restore original joint positions
        
        print(f"THE original positions are {original_positions}")
        for idx, pos in zip(self.joint_indices, original_positions):
            # print(f"ORIGINAL joints {original_positions} current {joint_positions}")
            p.resetJointState(self.robot_id, idx, pos)
            
        return jac
        
    def _quat_difference(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Compute quaternion difference
        """
        return p.getDifferenceQuaternion(q1.tolist(), q2.tolist())
    
    def solve_ik(self, 
                target_pos: np.ndarray, 
                target_orn: Optional[np.ndarray] = None,
                max_iters: int = 50,
                tolerance: float = 1e-3) -> np.ndarray:
        """pseudo-inverse IK solver"""
        print(f"\nSolving IK:")
        print(f"Target position: {target_pos}")
        print(f"Target orientation: {target_orn if target_orn is not None else 'None'}")

        original_joint_positions = None
        
        try:
            # acquire initial joint positions
            current_joint_positions = np.array([p.getJointState(self.robot_id, idx)[0] 
                                            for idx in self.joint_indices])
            print(f"Initial joint positions: {current_joint_positions}")

            original_joint_positions = current_joint_positions.copy() 
            for iter in range(max_iters):
                # acquire current end effector pose
                current_pos, current_orn = self._get_current_ee_pose()
                
                # calculate position error
                pos_error = target_pos - current_pos
                
                # if orientation is provided, calculate orientation error
                if target_orn is not None:
                    orn_error = self._compute_orientation_error(current_orn, target_orn)
                    error = np.concatenate([pos_error, orn_error])
                else:
                    error = pos_error
                
                error_norm = np.linalg.norm(error)
                print(f"Iteration {iter}, Error: {error_norm}")
                
                # check convergence
                if error_norm < tolerance:
                    print("IK solved successfully")
                    break
                
                # compute Jacobian matrix
                J = self._get_jacobian(current_joint_positions)
                
                # if orientation is not provided, use only position part of Jacobian
                if target_orn is None:
                    J = J[:3]
                
                # compute pseudo-inverse of Jacobian
                J_pinv = np.linalg.pinv(J, rcond=self.damping)
                
                # update joint positions
                delta_q = J_pinv @ error
                current_joint_positions += delta_q
                
                # set joint states for next iteration
                for i, idx in enumerate(self.joint_indices):
                    p.resetJointState(self.robot_id, idx, current_joint_positions[i])

            for i, idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, idx, original_joint_positions[i])
            
            return current_joint_positions
            
        except Exception as e:
            print(f"Error in IK solution: {str(e)}")
            raise
    
    def _get_current_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """acquire current end effector pose"""
        ee_state = p.getLinkState(self.robot_id, self.ee_index)
        return np.array(ee_state[0]), np.array(ee_state[1])
    
    def _compute_orientation_error(self, 
                                current_orn: np.ndarray, 
                                target_orn: np.ndarray) -> np.ndarray:
        """compute orientation error by quaternion difference"""
        current_mat = np.array(p.getMatrixFromQuaternion(current_orn)).reshape(3, 3)
        target_mat = np.array(p.getMatrixFromQuaternion(target_orn)).reshape(3, 3)
        
        error_mat = target_mat @ current_mat.T
        axis, theta = rotation_matrix_to_axis_angle(error_mat)
        # angle_axis = p.getAxisAngleFromMatrix(error_mat.flatten().tolist())
        
        if theta < 0:
            axis = [-x for x in axis]
            
        return np.array(axis[:3]) * theta


def rotation_matrix_to_axis_angle(R: np.ndarray):
    """
    Convert a 3x3 rotation matrix to axis-angle representation.

    Args:
        R (np.ndarray): 3x3 rotation matrix

    Returns:
        axis (np.ndarray): 3D unit vector representing the rotation axis
        angle (float): Rotation angle in radians
    """
    # Ensure R is a numpy array
    R = np.array(R)

    # Compute rotation angle
    theta = np.arccos((np.trace(R) - 1) / 2.0)

    # Handle small angles to avoid division by zero
    if np.isclose(theta, 0):
        return np.array([1, 0, 0]), 0  # No rotation

    # Compute rotation axis
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))

    return axis, theta
