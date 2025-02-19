import numpy as np

class EEVelocityController:
    def __init__(self, robot, ik_solver):
        self.robot = robot
        self.ik_solver = ik_solver
        self.velocity = np.array([0, 0, 0])

    def step(self, velocity, t=1/240):
        current_pos, _current_orientation = self.robot.get_ee_pose()
        self.velocity = velocity
        target_pos = current_pos + self.velocity*t
        joint_positions = self.ik_solver.solve_ik(
            target_pos,
            max_iters=10,     
            tolerance=1e-2)
        
        self.robot.position_control(joint_positions)
        print(f"the new joint positions are {joint_positions}")
        return joint_positions
 