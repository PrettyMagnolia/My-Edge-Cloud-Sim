import random

from core.base_scenario import BaseScenario
from core.infrastructure import Node, Location
from zoo.node import UserNode, EdgeNode
from zoo.link import UpStreamLink, DownStreamLink


class Scenario1(BaseScenario):
    def __init__(self, user_node_num: int, edge_node_num: int, random_seed: int) -> None:
        random.seed(random_seed)

        self.user_node_num = user_node_num
        self.edge_node_num = edge_node_num

        # 用户节点本地计算能力为10GHZ 计算功耗为 [1-500]W
        self.user_node_calculate_loc = 1  # todo modify
        self.user_node_power_loc = random.uniform(400, 500)

        # 边缘节点边缘计算能力为100GHZ 计算功耗为 [1-50]W
        self.edge_node_calculate_loc = 10  # todo modify
        self.edge_node_power_loc = random.uniform(40, 50)

        # 上行链路传输功率为15Gbps 传输功耗为 [0.1-1]W
        self.trans_up = 15
        self.power_up = random.uniform(0.1, 1)

        # 下行链路传输功率为200Gbps 传输功耗为 [1-10]W
        self.trans_down = 200
        self.power_down = random.uniform(1, 10)

        super().__init__()

    def init_infrastructure_nodes(self):
        # 添加用户节点
        for i in range(self.user_node_num):
            self.node_id2name[i] = f'u{i}'
            self.infrastructure.add_node(
                UserNode(
                    node_id=i,
                    name=f'u{i}',
                    location=Location(random.uniform(-1000, 1000), random.uniform(-1000, 1000)),
                    calculate_loc=self.user_node_calculate_loc,
                    power_loc=self.user_node_power_loc
                )
            )
        # 添加边缘节点
        for i in range(self.edge_node_num):
            self.node_id2name[i + self.user_node_num] = f'e{i}'
            self.infrastructure.add_node(
                EdgeNode(
                    node_id=i,
                    name=f'e{i}',
                    location=Location(random.uniform(-1000, 1000), random.uniform(-1000, 1000)),
                    calculate_mec=self.edge_node_calculate_loc,
                    power_mec=self.edge_node_power_loc
                )
            )

    def init_infrastructure_links(self):
        # 创建上行链路 用户节点到每一个边缘节点
        for i in range(self.user_node_num):
            for j in range(self.edge_node_num):
                self.infrastructure.add_link(
                    UpStreamLink(
                        src=self.infrastructure.get_node(f'u{i}'),
                        dst=self.infrastructure.get_node(f'e{j}'),
                        bandwidth=100,
                        trans_up=self.trans_up,
                        power_up=self.power_up
                    )
                )

        # 创建下行链路 边缘节点到每一个用户节点
        for i in range(self.edge_node_num):
            for j in range(self.user_node_num):
                self.infrastructure.add_link(
                    DownStreamLink(
                        src=self.infrastructure.get_node(f'e{i}'),
                        dst=self.infrastructure.get_node(f'u{j}'),
                        bandwidth=100,
                        trans_down=self.trans_down,
                        power_down=self.power_down
                    )
                )

    def status(self, node_name=None, link_args=None):
        # 返回节点和链路状态
        pass

    def run(self):
        print("Running scenario 1")
        print("Scenario 1 completed")
