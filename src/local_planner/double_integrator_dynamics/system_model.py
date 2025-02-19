
import sympy as sp

class SystemModel:
    """
    This is the system model for the double integrator
    the input is the acceleration
    The system is going to take the form of a control affine model
    X_dot = f(X) + g(X)*u
    """
    def __init__(self):
        self.x, self.y, self.z = sp.symbols("x y z") # positions of the end effector
        self.v_x, self.v_y, self.v_z = sp.symbols("v_x v_y v_z") # velocities of the end effector
        self.a_x, self.a_y, self.a_z = sp.symbols("a_x a_y a_z") # accelerations of the end effector along the principal axes
        # this is also the input of the system

        self.X = sp.Matrix([[self.x], [self.y], [self.z], [self.v_x], [self.v_y], [self.v_z]])
        self.u = sp.Matrix([[self.a_x], [self.a_y], [self.a_z]])

        self.f = sp.Matrix([[self.v_x], [self.v_y], [self.v_z], [0], [0], [0]])
        self.g = sp.Matrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])