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

    def __repr__(self):
        return f"Link: {self.src.name} -> {self.dst.name} || Bandwidth: {self.bandwidth} || Distance: {self.distance} || Transmission Up: {self.trans_up} || Power Up: {self.power_up}"


class DownStreamLink(Link):
    def __init__(
            self,
            src: Node,
            dst: Node,
            bandwidth: int,
            distance: float,
            trans_down: int,
            power_down: float
    ) -> None:
        super().__init__(src=src, dst=dst, bandwidth=bandwidth)
        self.distance = distance
        self.trans_down = trans_down
        self.power_down = power_down

    def __repr__(self):
        return f"Link: {self.src.name} -> {self.dst.name} || Bandwidth: {self.bandwidth} || Distance: {self.distance} || Transmission Down: {self.trans_down} || Power Down: {self.power_down}"
