import sympy as sp
import numpy as np

class ReachCLF:
    def __init__(self, system, destination):
        self.destination = destination
        self.system = system


    def set_destination(self, destination):
        self.destination = destination

    def get_lf(self):
        x_sys = sp.Matrix([self.system.q1, 
                            self.system.q2, 
                            self.system.q3, 
                            self.system.q4, 
                            self.system.q5, 
                            self.system.q6, 
                            self.system.q7])
        
        dest = sp.Matrix(self.destination)

        diff = x_sys - dest
        return 0.5*(diff.transpose() @ diff)

                            