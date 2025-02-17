class TrajectoryFollower:
    def __init__(self, trajectory, ik_controller, robot):
        """
            The trajectory is merely a set of points
        """
        self.trajectory = trajectory
        self.ik_controller = ik_controller
        self.robot = robot

    def step(self, t=1/240):
        pass