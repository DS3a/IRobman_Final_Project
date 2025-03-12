
import sympy as sp
import numpy as np

class CollsionConeCBF:
    """
        This class makes barrier functions to dynamic obstacles.

        It's going to be hard for a velocity controlled model
        because the velocity is one of the variables used in the barrier function.
        With a constant velocity model we can't treat the velocity as a variable, 
        but rather a constant, which should be interesting to see in  
    
    """
    def __init__(self):
        pass