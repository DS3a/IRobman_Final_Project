import sympy as sp
import numpy as np
from src.local_planner.panda_forward_dynamics.velocity_inputs.system_model import SystemModel
from src.local_planner.panda_forward_dynamics.velocity_inputs.reach_cbf import ReachCBF
from cvxopt import solvers, matrix


class Controller:
    def __init__(self, robot, step_time=1/240):
        self.robot = robot
        # self.velocity_controller = velocity_controller
        self.step_time = step_time
        self.system_model = SystemModel()
        self.reach_cbf = ReachCBF(self.system_model, np.array([0.65, 0.8, 1.24]))
        self.alpha = 0.99

    def eval_expr(self, expr, time):
        current_pos, _current_orientation = self.robot.get_ee_pose()
        numpy_matrix = self.robot.get_joint_positions().reshape((7, 1))
        sympy_matrix = self.system_model.X
        
        mapping = {sympy_matrix[i, 0]: numpy_matrix[i, 0] for i in range(7)}
        mapping[self.reach_cbf.t] = time

        return expr.subs(mapping)

        

    def step(self, time):
        t = self.reach_cbf.t
        b = self.reach_cbf.barrier_fn
        # f = self.system_model.f
        g = self.system_model.g
        alpha = self.alpha

        lhs_a1 = -self.lie_derivative(b, g.col(0))
        lhs_a2 = -self.lie_derivative(b, g.col(1))
        lhs_a3 = -self.lie_derivative(b, g.col(2))
        lhs_a4 = -self.lie_derivative(b, g.col(3))
        lhs_a5 = -self.lie_derivative(b, g.col(4))
        lhs_a6 = -self.lie_derivative(b, g.col(5))
        lhs_a7 = -self.lie_derivative(b, g.col(6))
        rhs = self.Lf(b) + alpha*b
        # rhs = self.Lf(b, 2) + sp.diff(b, t, 2) + alpha*sp.diff(b, t) + alpha*alpha*b + self.Lf(alpha*b) + sp.diff(alpha*b, t)
        # rhs = self.Lf(b, 2) + sp.diff(b, t, 2) + 2*b*self.Lf(b) + 2*b*sp.diff(b, t) + alpha*self.Lf(b) + alpha*b*b
        # rhs = self.Lf(b, 2) + sp.diff(b, t, 2) + 2*b*self.Lf(b) + 2*b*sp.diff(b, t) + (self.Lf(b))**2 + (sp.diff(b, t))**2 + b**4 + 2*self.Lf(b)*sp.diff(b, t) + 2*(b**2)*self.Lf(b) + 2*(b**2)*sp.diff(b, t)

        # barrier functions to enforce velocity constraints
        vel_max = 8

        nu_max = vel_max # max velocity can be nu_max
        # b_vx = (self.system_model.v_x * np.tan(nu_max))**2 - (self.system_model.v_x)**2
        # b_vy = (self.system_model.v_y * np.tan(nu_max))**2 - (self.system_model.v_y)**2
        # b_vz = (self.system_model.v_z * np.tan(nu_max))**2 - (self.system_model.v_z)**2

        # b_vx = nu_max**2 - (self.system_model.v_x)**2
        # b_vy = nu_max**2 - (self.system_model.v_y)**2
        # b_vz = nu_max**2 - (self.system_model.v_z)**2


        # lhs_vx = -self.lie_derivative(b_vx, g.col(0))
        # lhs_vy = -self.lie_derivative(b_vy, g.col(1))
        # lhs_vz = -self.lie_derivative(b_vz, g.col(2))

        # rhs_vx = self.Lf(b_vx) + alpha*b_vx
        # rhs_vy = self.Lf(b_vy) + alpha*b_vy
        # rhs_vz = self.Lf(b_vz) + alpha*b_vz

        G = np.array([
            [self.eval_expr(lhs_a1, time), 
             self.eval_expr(lhs_a2, time), 
             self.eval_expr(lhs_a3, time), 
             self.eval_expr(lhs_a4, time),
             self.eval_expr(lhs_a5, time),
             self.eval_expr(lhs_a6, time),
             self.eval_expr(lhs_a7, time),],
            # [self.eval_expr(lhs_vx, time), 0, 0],
            # [0, self.eval_expr(lhs_vy, time), 0],
            # [0, 0, self.eval_expr(lhs_vz, time)],
            # # [-1, 0, 0],
            # [0, -1, 0],
            # [0, 0, -1],
            # [1, 0, 0],
            # [0, 1, 0],
            # [0, 0, 1],
        ], dtype=float)



        print(f"the rhs is {self.eval_expr(rhs, time)}")
        delta = 0
        h = np.array([
            [self.eval_expr(rhs, time)],
            # [self.eval_expr(rhs_vx, time) + delta],
            # [self.eval_expr(rhs_vy, time) + delta],
            # [self.eval_expr(rhs_vz, time) + delta],
            # # [-vel_max],
            # [-vel_max],
            # [-vel_max],
            # [vel_max],
            # [vel_max],
            # [vel_max],
        ], dtype=float)

        P = 0.5*np.eye(7)
        Q = np.array([0., 0., 0., 0., 0., 0., 0.], dtype=float).reshape((7,))

        sol = solvers.qp(matrix(P), matrix(Q), matrix(G), matrix(h))

        a = np.array(sol['x']).reshape((7,))
        # print(f"the solution has been calculated and it's {self.velocity_controller.velocity.shape}")
        # velocity = self.velocity_controller.velocity + a*self.step_time
        # self.velocity_controller.step(velocity)

        return a

    def lie_derivative(self, barrier_fn, vector_field, n=1):
        fn_ = barrier_fn
        X = self.system_model.X
        for n in range(0, n):
            grad = sp.diff(fn_, X)
            fn_ = grad.dot(vector_field)

        return fn_

    def Lf(self, barrier_fn, order=1):
        return self.lie_derivative(barrier_fn, self.system_model.f, order)