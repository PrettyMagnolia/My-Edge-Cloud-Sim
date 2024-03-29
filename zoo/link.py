from core.infrastructure import Link, Node


class UpStreamLink(Link):
    def __init__(
            self,
            src: Node,
            dst: Node,
            bandwidth: int,
            trans_up: int,
            power_up: float

    ) -> None:
        super().__init__(src=src, dst=dst, bandwidth=bandwidth)
        self.trans_up = trans_up
        self.power_up = power_up

        # 计算两点之间的距离
        src_x, src_y = src.location.loc()
        dst_x, dst_y = dst.location.loc()
        self.distance = ((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2) ** 0.5

        # 信号传播速度 设置为光速的0.9倍
        self.signal_speed = 0.9 * 3 * 10 ** 8

    def __repr__(self):
        return f"Link: {self.src.name} -> {self.dst.name} || Bandwidth: {self.bandwidth} || Distance: {self.distance} || Transmission Up: {self.trans_up} || Power Up: {self.power_up}"


class DownStreamLink(Link):
    def __init__(
            self,
            src: Node,
            dst: Node,
            bandwidth: int,
            trans_down: int,
            power_down: float
    ) -> None:
        super().__init__(src=src, dst=dst, bandwidth=bandwidth)
        self.trans_down = trans_down
        self.power_down = power_down

        # 计算两点之间的距离
        src_x, src_y = src.location.loc()
        dst_x, dst_y = dst.location.loc()
        self.distance = ((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2) ** 0.5

        # 信号传播速度 设置为光速的0.9倍
        self.signal_speed = 0.9 * 3 * 10 ** 8

    def __repr__(self):
        return f"Link: {self.src.name} -> {self.dst.name} || Bandwidth: {self.bandwidth} || Distance: {self.distance} || Transmission Down: {self.trans_down} || Power Down: {self.power_down}"


class DTDUpStreamLink(Link):
    def __init__(
            self,
            src: Node,
            dst: Node,
            bandwidth: int,
            trans_dtd_up: int,
            power_dtd_up: float
    ) -> None:
        super().__init__(src=src, dst=dst, bandwidth=bandwidth)
        self.trans_dtd_up = trans_dtd_up
        self.power_dtd_up = power_dtd_up

        # 计算两点之间的距离
        src_x, src_y = src.location.loc()
        dst_x, dst_y = dst.location.loc()
        self.distance = ((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2) ** 0.5

        # 信号传播速度 设置为光速的0.9倍
        self.signal_speed = 0.9 * 3 * 10 ** 8


class DTDDownStreamLink(Link):
    def __init__(
            self,
            src: Node,
            dst: Node,
            bandwidth,
            trans_dtd_down: int,
            power_dtd_down: float
    ) -> None:
        super().__init__(src=src, dst=dst, bandwidth=bandwidth)
        self.trans_dtd_down = trans_dtd_down
        self.power_dtd_down = power_dtd_down

        # 计算两点之间的距离
        src_x, src_y = src.location.loc()
        dst_x, dst_y = dst.location.loc()
        self.distance = ((src_x - dst_x) ** 2 + (src_y - dst_y) ** 2) ** 0.5

        # 信号传播速度 设置为光速的0.9倍
        self.signal_speed = 0.9 * 3 * 10 ** 8
