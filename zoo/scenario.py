import random

from core.base_scenario import BaseScenario
from core.infrastructure import Node, Location
from zoo.node import UserNode, EdgeNode, DTDNode
from zoo.link import UpStreamLink, DownStreamLink, DTDUpStreamLink, DTDDownStreamLink


class Scenario1(BaseScenario):
    def __init__(self, user_node_num: int, edge_node_num: int, dtd_node_num: int, random_seed: int) -> None:
        random.seed(random_seed)

        # --节点信息设置--
        self.user_node_num = user_node_num
        self.edge_node_num = edge_node_num
        self.dtd_node_num = edge_node_num

        # 用户节点本地计算能力为5GHZ 计算功耗为 [400-500]W
        self.user_node_calculate_loc = 5
        self.user_node_power_loc = random.uniform(400, 500)

        # 边缘节点边缘计算能力为50GHZ 计算功耗为 [40-50]W
        self.edge_node_calculate_loc = 50
        self.edge_node_power_loc = random.uniform(40, 50)

        # DTD节点计算能力为8GHZ 计算功耗为 [90-100]W
        self.dtd_node_calculate_loc = 8
        self.dtd_node_power_loc = random.uniform(90, 100)

        # --链路信息设置--
        # 上行链路传输功率为15Gbps 传输功耗为 [900-1000]mW
        self.trans_up = 15
        self.power_up = random.uniform(0.9, 1)

        # 下行链路传输功率为200Gbps 传输功耗为 [10-20]mW
        self.trans_down = 200
        self.power_down = random.uniform(0.01, 0.02)

        # DTD上行链路传输功率为15Gbps 传输功耗为 [900-1000]mW
        self.trans_up_dtd = 15
        self.power_up_dtd = random.uniform(0.9, 1)

        # DTD下行链路传输功率为50Gbps 传输功耗为 [50-100]mW
        self.trans_down_dtd = 50
        self.power_down_dtd = random.uniform(0.05, 0.1)

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

        # 添加DTD节点
        for i in range(self.dtd_node_num):
            self.node_id2name[i + self.user_node_num + self.edge_node_num] = f'd{i}'
            self.infrastructure.add_node(
                DTDNode(
                    node_id=i,
                    name=f'd{i}',
                    location=Location(random.uniform(-1000, 1000), random.uniform(-1000, 1000)),
                    calculate_dtd=self.dtd_node_calculate_loc,
                    power_dtd=self.dtd_node_power_loc
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
        # 创建DTD上行链路 用户节点到每一个DTD节点
        for i in range(self.user_node_num):
            for j in range(self.dtd_node_num):
                self.infrastructure.add_link(
                    DTDUpStreamLink(
                        src=self.infrastructure.get_node(f'u{i}'),
                        dst=self.infrastructure.get_node(f'd{j}'),
                        bandwidth=100,
                        trans_dtd_up=self.trans_up_dtd,
                        power_dtd_up=self.power_up_dtd
                    )
                )
        # 创建DTD下行链路 DTD节点到每一个用户节点
        for i in range(self.dtd_node_num):
            for j in range(self.user_node_num):
                self.infrastructure.add_link(
                    DTDDownStreamLink(
                        src=self.infrastructure.get_node(f'd{i}'),
                        dst=self.infrastructure.get_node(f'u{j}'),
                        bandwidth=100,
                        trans_dtd_down=self.trans_down_dtd,
                        power_dtd_down=self.power_down_dtd
                    )
                )

    def status(self, node_name=None, link_args=None):
        # 返回节点和链路状态
        pass

    def run(self):
        print("Running scenario 1")
        print("Scenario 1 completed")
