import sympy as sp

class ReachCBF:
    def __init__(self, system, destination):
        self.t = sp.symbols("t")
        diff = sp.Matrix([system.x, system.y, system.z]) - sp.Matrix(destination)
        self.dist = diff.dot(diff)


        self.temporal_fn = 1.2*sp.exp(-0.5*self.t) + 0.05
        # Start 3 metres away from the goal reticle, and get within 5cm of it in 12 seconds
        
        self.barrier_fn = self.temporal_fn - self.dist