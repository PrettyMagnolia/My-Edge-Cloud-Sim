import gym
from gym import spaces
import numpy as np

class Car2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.xth = 0
        self.target_x = 0
        self.target_y = 0
        self.L = 10
        self.action_space = spaces.Discrete(5)  # 0, 1, 2，3，4: 不动，上下左右
        self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))
        self.state = None
        pass

    def step(self, action):
        if action == 0:
            pass
        elif action == 1:
            self.target_y += 1
        elif action == 2:
            self.target_y -= 1
        elif action == 3:
            self.target_x -= 1
        elif action == 4:
            self.target_x += 1
        else:
            raise ValueError("Invalid action")

        return self.state, reward, done, {}

    def reset(self):
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None
