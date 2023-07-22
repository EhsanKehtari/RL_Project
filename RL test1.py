import gym
from gym import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    def __init__(self, size=5):
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space =Box(low=0, high=np.infty, shape=(6,), dtype=np.float32)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(9)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def reset(self):
        pass

    def step(self, action):
        pass



class Patient:
    """

    """
    def __init__(self):
        pass

    def tmp(self):
        pass


class Resource:
    """

    """
    def __init__(self):
        pass

    def tmp(self):
        pass

    