import sympy as sp
import numpy as np
import math

class SystemModel:
    def __init__(self):
        self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7 = sp.symbols("q_1 q_2 q_3 q_4 q_5 q_6 q_7")
        self.q1_dot, self.q2_dot, self.q3_dot, self.q4_dot, self.q5_dot, self.q6_dot, self.q7_dot = sp.symbols("q_1_dot q_2_dot q_3_dot q_4_dot q_5_dot q_6_dot q_7_dot")

        self.X = sp.Matrix([
            self.q1,
            self.q2,
            self.q3,
            self.q4,
            self.q5,
            self.q6,
            self.q7,
            # self.q1_dot,
            # self.q2_dot,
            # self.q3_dot,
            # self.q4_dot,
            # self.q5_dot,
            # self.q6_dot,
            # self.q7_dot,
        ])

        self.f = sp.Matrix([
            # self.q1_dot,
            # self.q2_dot,
            # self.q3_dot,
            # self.q4_dot,
            # self.q5_dot,
            # self.q6_dot,
            # self.q7_dot,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ])

        # self.g = sp.Matrix(np.concatenate([np.zeros((7, 1)), np.eye(7)]))
        self.g = sp.Matrix(np.eye(7))
        
        
        M_PI = math.pi
        # Create DH parameters (data given by maker franka-emika)
        self.dh = [[ 0,      0,        0.333,   self.q1],
            [-M_PI/2,   0,        0,       self.q2],
            [ M_PI/2,   0,        0.316,   self.q3],
            [ M_PI/2,   0.0825,   0,       self.q4],
            [-M_PI/2,  -0.0825,   0.384,   self.q5],
            [ M_PI/2,   0,        0,       self.q6],
            [ M_PI/2,   0.088,    0.107,   self.q7]]
 

    def TF_matrix(self, i):
        # Define Transformation matrix based on DH params
        dh = self.dh
        alpha = dh[i][0]
        a = dh[i][1]
        d = dh[i][2]
        q = dh[i][3]
        
        TF = sp.Matrix([
            [sp.cos(q),-sp.sin(q), 0, a],
            [sp.sin(q)*sp.cos(alpha), sp.cos(q)*sp.cos(alpha), -sp.sin(alpha), -sp.sin(alpha)*d],
            [sp.sin(q)*sp.sin(alpha), sp.cos(q)*sp.sin(alpha),  sp.cos(alpha),  sp.cos(alpha)*d],
            [   0,  0,  0,  1]])
        return TF
    
    def forward_kinematics(self):
        frames = []

        frame = sp.Matrix(np.eye(4))
        for i in range(7):
            frame = frame.dot(self.TF_matrix(i))
            frames.append(frame)

        return frames
        
    def TF_matrix_with_angle(self, i, q):
        # print(f"THE angle is {q}")
        # Define Transformation matrix based on DH params
        dh = self.dh
        alpha = dh[i][0]
        a = dh[i][1]
        d = dh[i][2]
        
        TF = np.array([
            [sp.cos(q),-sp.sin(q), 0, a],
            [sp.sin(q)*sp.cos(alpha), sp.cos(q)*sp.cos(alpha), -sp.sin(alpha), -sp.sin(alpha)*d],
            [sp.sin(q)*sp.sin(alpha), sp.cos(q)*sp.sin(alpha),  sp.cos(alpha),  sp.cos(alpha)*d],
            [   0,  0,  0,  1]])
        return TF
 
    def forward_kinematics_fast(self, joint_angles):
        # print(f"THE angles are {joint_angles}")
        frames = []
        frame = np.eye(4)
        for i in range(7):
            frame = frame.dot(self.TF_matrix_with_angle(i, joint_angles[i]))
            frames.append(frame)

        # print(f"THE angles are {frame}")
        return frames