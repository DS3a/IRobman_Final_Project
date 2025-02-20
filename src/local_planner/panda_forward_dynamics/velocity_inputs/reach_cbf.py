import sympy as sp

class ReachCBF:
    def __init__(self, system, destination):
        self.t = sp.symbols("t")
        self.system = system
        ee_ht = self.system.forward_kinematics()[6]

        x = ee_ht.col(3)[0]
        y = ee_ht.col(3)[1]
        z = ee_ht.col(3)[2]
        diff = sp.Matrix([x, y, z]) - sp.Matrix(destination)
        self.dist = diff.dot(diff)


        self.temporal_fn = 1.2*sp.exp(-2*self.t) + 0.005
        # Start 3 metres away from the goal reticle, and get within 5cm of it in 12 seconds
        
        self.barrier_fn = self.temporal_fn - self.dist