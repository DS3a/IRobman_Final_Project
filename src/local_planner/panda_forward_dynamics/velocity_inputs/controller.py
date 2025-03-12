import sympy as sp
import numpy as np
from src.local_planner.panda_forward_dynamics.velocity_inputs.system_model import SystemModel
from src.local_planner.panda_forward_dynamics.velocity_inputs.reach_cbf import ReachCBF
from src.local_planner.panda_forward_dynamics.velocity_inputs.reach_clf import ReachCLF
from cvxopt import solvers, matrix
# from src.controllers.ik_controller import IKController
from src.local_planner.ik_solver import IKController

### TODO instead of using a reach cbf, I could just get the 
# final position using the IK solver, and make the cbf 
# try to reach that instead using a control lyapunav function, 
# and can use the collision cone cbf 


class Controller:
    def __init__(self, robot, step_time=1/240):
        self.robot = robot
        # self.velocity_controller = velocity_controller
        self.step_time = step_time
        self.system_model = SystemModel()
        # self.reach_cbf = ReachCBF(self.system_model, np.array([0.65, 0.8, 1.24]))
        self.goal_position = np.array([0.65, 0.4, 1.24])
        self.goal_orientation = np.array([1, 0.0, 0, 0])
        self.ik_controller = IKController(
                    robot_id=robot.id,
                    joint_indices=robot.arm_idx,
                    ee_index=robot.ee_idx
                )
        # self.joint_angle_goals = self.ik_controller.solve_ik(self.goal_position, target_orn=self.goal_orientation)
        self.joint_angle_goals = self.ik_controller.solve_ik(self.goal_position)


        # self.reach_cbf = ReachCBF(self.system_model, np.array([-0.65, -0.8, 1.84]))
        self.reach_clf = ReachCLF(self.system_model, self.joint_angle_goals)
        self.alpha = 0.99


    def eval_expr(self, expr, time):
        current_pos, _current_orientation = self.robot.get_ee_pose()
        numpy_matrix = self.robot.get_joint_positions().reshape((7, 1))
        sympy_matrix = self.system_model.X
        
        mapping = {sympy_matrix[i, 0]: numpy_matrix[i, 0] for i in range(7)}
        # mapping[self.reach_cbf.t] = time

        return expr.subs(mapping)


    def set_goals(self, goal_position, goal_orientation):
        self.goal_position = goal_position
        self.goal_orientation = goal_orientation
        # self.joint_angle_goals = self.ik_controller.solve_ik(self.goal_position, target_orn=self.goal_orientation)
        self.joint_angle_goals = self.ik_controller.solve_ik(self.goal_position)
        self.reach_clf = ReachCLF(self.system_model, self.joint_angle_goals)

    def step(self, time):
        # t = self.reach_cbf.t
        # b = self.reach_cbf.barrier_fn
        v = self.reach_clf.get_lf()
        # f = self.system_model.f
        g = self.system_model.g
        alpha = self.alpha

        lhs_a1 = self.lie_derivative(v, g.col(0))
        lhs_a2 = self.lie_derivative(v, g.col(1))
        lhs_a3 = self.lie_derivative(v, g.col(2))
        lhs_a4 = self.lie_derivative(v, g.col(3))
        lhs_a5 = self.lie_derivative(v, g.col(4))
        lhs_a6 = self.lie_derivative(v, g.col(5))
        lhs_a7 = self.lie_derivative(v, g.col(6))
        rhs = -self.Lf(v) - alpha * v
        # rhs = self.Lf(b, 2) + sp.diff(b, t, 2) + alpha*sp.diff(b, t) + alpha*alpha*b + self.Lf(alpha*b) + sp.diff(alpha*b, t)
        # rhs = self.Lf(b, 2) + sp.diff(b, t, 2) + 2*b*self.Lf(b) + 2*b*sp.diff(b, t) + alpha*self.Lf(b) + alpha*b*b
        # rhs = self.Lf(b, 2) + sp.diff(b, t, 2) + 2*b*self.Lf(b) + 2*b*sp.diff(b, t) + (self.Lf(b))**2 + (sp.diff(b, t))**2 + b**4 + 2*self.Lf(b)*sp.diff(b, t) + 2*(b**2)*self.Lf(b) + 2*(b**2)*sp.diff(b, t)

        vel_lower_constraints_g = -np.eye(7)
        vel_higher_constraints_g = np.eye(7)

        G = np.array([
            [self.eval_expr(lhs_a1, time), 
             self.eval_expr(lhs_a2, time), 
             self.eval_expr(lhs_a3, time), 
             self.eval_expr(lhs_a4, time),
             self.eval_expr(lhs_a5, time),
             self.eval_expr(lhs_a6, time),
             self.eval_expr(lhs_a7, time),],
        ], dtype=float)


        # G = np.concatenate([G, vel_lower_constraints_g, vel_higher_constraints_g])


        # print(f"the rhs is {self.eval_expr(rhs, time)}")
        delta = 0
        h = np.array([
            [self.eval_expr(rhs, time)],
        ], dtype=float)
        vel_constraints_lower_h = -np.ones((7, 1))*2.175
        vel_constraints_higher_h = np.ones((7, 1))*2.175

        # h = np.concatenate([h, vel_constraints_lower_h, vel_constraints_higher_h])

        P = 0.5*np.eye(7)
        Q = np.array([0., 0., 0., 0., 0., 0., 0.], dtype=float).reshape((7,))
        
        dist = self.eval_expr(v, time=0)
        # at the moment, the qp becomes infeasible when you go near the setpoint, IDK why
        if dist > 0.01:
            print(f"THE angles are {dist} away")
            sol = solvers.qp(matrix(P), matrix(Q), matrix(G), matrix(h))

            v = np.array(sol['x']).reshape((7,))*100
        else:
            v = np.zeros((7,))


        # print(f"the distnance is {self.eval_expr(self.reach_cbf.dist, time)} at time {time}")

        return v

    def lie_derivative(self, barrier_fn, vector_field, n=1):
        fn_ = barrier_fn
        X = self.system_model.X
        for n in range(0, n):
            grad = sp.diff(fn_, X)
            fn_ = grad.dot(vector_field)

        return fn_

    def Lf(self, barrier_fn, order=1):
        return self.lie_derivative(barrier_fn, self.system_model.f, order)