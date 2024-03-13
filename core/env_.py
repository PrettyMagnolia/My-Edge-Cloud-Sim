import gym
import simpy
import numpy as np
from gym import spaces
from core.base_scenario import BaseScenario
from core.task import Task


class Env(gym.Env):
    def __init__(self, scenario: BaseScenario, user_node_num: int, edge_node_num: int):
        super(Env, self).__init__()
        self.scenario = scenario
        self.controller = simpy.Environment()

        # 更新状态空间大小：任务大小（1）+ 任务源节点（user_node_num）+ 边缘节点信息（edge_node_num * 3）
        self.state_size = 1 + user_node_num + edge_node_num * 3
        self.action_space = spaces.Discrete(edge_node_num + 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

    def step(self, action):
        # 示例更新逻辑
        self.total_delay += np.random.random()  # 更新延迟
        self.total_energy += np.random.random()  # 更新能耗

        state = self._get_next_state()
        done = self._check_done()
        reward = self._calculate_reward()

        return state, reward, done, {}

    def reset(self):
        self.total_delay = 0
        self.total_energy = 0
        return self._get_initial_state()

    def _get_initial_state(self):
        # 生成初始状态
        return np.zeros(self.state_size, dtype=np.float32)

    def _get_next_state(self):
        # 构建状态向量
        state = np.zeros(self.state_size)
        # 假设这里已经有了当前任务信息和边缘节点信息
        task_size_normalized = np.random.random()  # 示例：任务大小
        task_source_node_one_hot = np.eye(self.scenario.user_node_num)[np.random.randint(0, self.scenario.user_node_num)]  # 示例：任务源节点
        edge_node_info = np.concatenate([np.random.random(size=(self.scenario.edge_node_num, 3))])  # 示例：边缘节点信息

        state[0] = task_size_normalized
        state[1:self.scenario.user_node_num + 1] = task_source_node_one_hot
        state[self.scenario.user_node_num + 1:] = edge_node_info.flatten()

        return state

    def _calculate_reward(self):
        # 计算奖励函数
        alpha = 0.5  # 时延权重
        beta = 0.5  # 能耗权重
        return -(alpha * self.total_delay + beta * self.total_energy)

    def _check_done(self):
        # 检查是否完成
        return False
