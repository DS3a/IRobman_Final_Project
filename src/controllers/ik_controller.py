import numpy as np
import pybullet as p
from typing import List, Tuple, Optional

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
        original_positions = joint_positions.copy()
        
        # acquire current end effector pose
        current_pos, current_orn = self._get_current_ee_pose()
        
        # partial differentiation for each joint
        for i in range(len(self.joint_indices)):
            # perturb joint position
            joint_positions = original_positions.copy()
            joint_positions[i] += delta
            
            # update joint states
            for idx, pos in zip(self.joint_indices, joint_positions):
                p.resetJointState(self.robot_id, idx, pos)
            
            # acquire new end effector pose
            new_pos, new_orn = self._get_current_ee_pose()
            
            # position jacobian
            jac[:3, i] = (new_pos - current_pos) / delta
            
            # orientation jacobian
            # quaternion difference as angular velocity
            orn_diff = self._quat_difference(current_orn, new_orn)
            jac[3:, i] = np.array(orn_diff[:3]) / delta  # only use imaginary part of quaternion
            
        # restore original joint positions
        for idx, pos in zip(self.joint_indices, original_positions):
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
        
        try:
            # acquire initial joint positions
            current_joint_positions = np.array([p.getJointState(self.robot_id, idx)[0] 
                                            for idx in self.joint_indices])
            print(f"Initial joint positions: {current_joint_positions}")
            
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
        angle_axis = p.getAxisAngleFromMatrix(error_mat.flatten().tolist())
        
        if angle_axis[3] < 0:
            angle_axis = [-x for x in angle_axis]
            
        return np.array(angle_axis[:3]) * angle_axis[3]
