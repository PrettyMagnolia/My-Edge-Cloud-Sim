import math
import random

import gym
import simpy
import numpy as np
from gym import spaces
from core.base_scenario import BaseScenario
from zoo.task import Task
from zoo.scenario import Scenario1
from core.utils import min_max_normalization, real_time_normalize


class EnvLogger:
    def __init__(self, controller):
        self.controller = controller

    def log(self, content):
        print("[{:.2f}]: {}".format(self.controller.now, content))
        pass


class Env(gym.Env):
    def __init__(self, user_node_num: int, edge_node_num: int):
        super(Env, self).__init__()
        random.seed(1)
        self.scenario = Scenario1(user_node_num=user_node_num, edge_node_num=edge_node_num, dtd_node_num=6, random_seed=1)

        self.controller = simpy.Environment()
        self.logger = EnvLogger(self.controller)

        self.task_list = []
        self.task_index = 0
        self.current_task = None

        self.sys_time_list = []
        self.sys_energy_list = []
        self.sys_total_time = 0
        self.sys_total_energy = 0
        # 任务超时数
        self.task_time_out_num = 0

        # 更新状态空间大小：任务大小（1）+ 任务源节点（user_node_num）+ 源节点计算功耗 + 边缘节点信息（edge_node_num * 5：距离，计算功耗，上行传输功耗，下行传输功耗）
        self.state_size = 1 + user_node_num + 1 + edge_node_num * 4
        self.action_space = spaces.Discrete(edge_node_num + 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

    def step(self, action):
        if action == 0:
            # 选择本地执行
            task_time, task_energy = self.local_execute(self.current_task)
            dst_node = None
        else:
            dst_node = self.scenario.get_node(f'e{action - 1}')
            if dst_node is None or dst_node.is_available is False:
                # 选择了不可卸载的远程节点 需要进行惩罚
                task_time, task_energy = self.local_execute(self.current_task)
            else:
                task_time, task_energy = self.edge_execute(self.current_task, dst_name=f'e{action - 1}')
        # 任务时间单位转换
        task_time *= 1e3
        time_out_flag = False
        if task_time > self.current_task.max_time * 1e-3:
            print(f"Task {self.current_task.task_id} timeout! Time: {task_time}ms")
            self.task_time_out_num += 1
            # 超时惩罚
            time_out_flag = True

        self.sys_total_time += task_time
        self.sys_total_energy += task_energy
        self.sys_time_list.append(task_time)
        self.sys_energy_list.append(task_energy)

        # 分配任务到线程
        self.controller.process(self.task_exec(task_time, dst_node))
        # 每100ms更新一次状态
        self.controller.run(self.controller.now + 100 * 1e-3)

        state = self._get_next_state()
        done = self._check_done()
        reward = self._calculate_reward(time_out_flag=time_out_flag)

        return state, reward, done, {}

    def task_exec(self, exec_time, dst_node=None):
        if dst_node is not None:
            dst_node.is_available = False
        yield self.controller.timeout(exec_time)
        if dst_node is not None:
            dst_node.is_available = True

    def reset(self):
        self.controller = simpy.Environment()
        self.logger = EnvLogger(self.controller)
        self.generate_tasks(task_num=100, task_size=[500, 1000])
        self.task_index = 0
        self.sys_total_energy = 0
        self.sys_total_time = 0
        self.current_task = self.task_list[self.task_index]
        return self._get_initial_state()

    def _get_initial_state(self):
        # 生成初始状态
        return np.zeros(self.state_size, dtype=np.float32)

    def _get_next_state(self):
        # 构建状态向量
        state = np.zeros(self.state_size)
        # 假设这里已经有了当前任务信息和边缘节点信息
        task_size_normalized = min_max_normalization(self.current_task.task_size, 500, 1000)  # 示例：任务大小归一化
        task_source_node_one_hot = np.eye(self.scenario.user_node_num)[self.task_index % self.scenario.user_node_num]  # 示例：任务源节点

        src_node = self.scenario.get_node(self.current_task.src_name)
        user_cal_power = min_max_normalization(src_node.power_loc, 400, 500)

        edge_node_info = []
        for i in range(self.scenario.edge_node_num):
            dst_node = self.scenario.get_node(f'e{i}')
            up_link = self.scenario.get_link(self.current_task.src_name, dst_node.name)
            down_link = self.scenario.get_link(dst_node.name, self.current_task.src_name)
            up_power = min_max_normalization(up_link.power_up, 0.1, 1)  # 示例：上行传输功耗归一化
            down_power = min_max_normalization(down_link.power_down, 1, 10)
            edge_cal_power = min_max_normalization(dst_node.power_mec, 40, 50)
            distance = min_max_normalization(up_link.distance, 0, 2000 * 2 ** 0.5)
            edge_node_info.extend([distance, edge_cal_power, up_power, down_power])
            # print(edge_node_info)
        state[0] = task_size_normalized
        state[1:self.scenario.user_node_num + 1] = task_source_node_one_hot
        state[self.scenario.user_node_num + 1] = user_cal_power
        state[self.scenario.user_node_num + 2:] = np.array(edge_node_info)

        self.task_index += 1
        if self.task_index < len(self.task_list):
            self.current_task = self.task_list[self.task_index]

        return state

    def _calculate_reward(self, time_out_flag):
        # 计算奖励函数
        alpha = 0.5  # 时延权重
        beta = 0.5  # 能耗权重
        time = real_time_normalize(self.sys_time_list)
        energy = real_time_normalize(self.sys_energy_list)

        return math.log(1.0 / (alpha * time + beta * energy)) - 100 if not time_out_flag else -1

    def _check_done(self):
        # 检查是否完成
        return self.task_index == len(self.task_list)

    def local_execute(self, task: Task, dst_name=None):
        """本地执行任务"""
        src_node = self.scenario.get_node(task.src_name)

        # 本地计算时延 = 任务大小 / 本地计算能力
        exc_time = (task.task_size * 1024) / (src_node.calculate_loc * 1e9)
        # 本地计算能耗 = 本地计算功耗 * 本地计算时延
        exc_energy = src_node.power_loc * exc_time
        self.logger.log(f"Task {task.task_id} Local Execute: "f"{task.src_name} Time: {exc_time}s Energy: {exc_energy}J")

        # 总时延 = 本地计算时延
        total_time = exc_time

        # 总能耗 = 本地计算能耗
        total_energy = exc_energy

        return total_time, total_energy

    def edge_execute(self, task: Task, dst_name=None):
        """远程执行任务"""
        # 判断目标卸载节点为空的情况
        if dst_name is None:
            raise ValueError("dst_name cannot be None!")

        dst_node = self.scenario.get_node(dst_name)
        dst_node.available = False

        # 获取上行/下行传输链路
        up_stream_link = self.scenario.get_link(task.src_name, dst_name)
        down_stream_link = self.scenario.get_link(dst_name, task.src_name)

        # 1. 上行传输时延 = 发送时延 + 传播时延 = 任务大小 / 上行传输速率 + 信道长度 / 电磁波的传播速率
        up_stream_time = (task.task_size * 1024) / (up_stream_link.trans_up * 1e9) + up_stream_link.distance / up_stream_link.signal_speed
        # 1. 上行传输能耗 = 上行传输功率 * 上行传输时延
        up_stream_energy = up_stream_link.power_up * up_stream_time

        # 2. 边缘计算时延 = 任务大小 / 边缘计算能力
        exc_time = (task.task_size * 1024) / (self.scenario.get_node(dst_name).calculate_mec * 1e9)
        # 2. 边缘计算能耗 = 边缘计算功率 * 边缘计算时延
        exc_energy = self.scenario.get_node(dst_name).power_mec * exc_time

        # 3. 下行传输时延 = 发送时延 + 传播时延 = 任务大小 / 下行传输速率 + 信道长度 / 电磁波的传播速率
        down_stream_time = (task.task_size * 1024) / (down_stream_link.trans_down * 1e9) + down_stream_link.distance / down_stream_link.signal_speed
        # 3. 下行传输能耗 = 下行传输功率 * 下行传输时延
        down_stream_energy = down_stream_link.power_down * down_stream_time

        # 4. 总时延 = 上行传输时延 + 边缘计算时延 + 下行传输时延
        total_time = up_stream_time + exc_time + down_stream_time

        # 4. 总能耗 = 上行传输能耗 + 边缘计算能耗 + 下行传输能耗
        total_energy = up_stream_energy + exc_energy + down_stream_energy

        return total_time, total_energy

    def generate_tasks(self, task_num: int, task_size: list):
        self.task_list = []
        for i in range(task_num):
            task = Task(task_id=i, task_size=random.uniform(task_size[0], task_size[1]), src_name='u' + str(i % self.scenario.user_node_num), max_time=150)
            self.task_list.append(task)

    def close(self):
        # self.vis_graph(save_as=None)
        self.logger.log("Simulation completed!")
        self.logger.log(f"Total time: {self.sys_total_time}s Total energy: {self.sys_total_energy}J")
        self.logger.log(f"Task timeout: {self.task_time_out_num}")
        return self.sys_total_time, self.sys_total_energy
