from core.infrastructure import Node, Location


class UserNode(Node):
    def __init__(
            self,
            node_id: int,
            name: str,
            location: Location,
            calculate_loc: int,
            power_loc: float
    ) -> None:
        super().__init__(node_id=node_id, name=name, location=location)
        self.calculate_loc = calculate_loc
        self.power_loc = power_loc

    def __repr__(self) -> str:
        return f"Node Name: {self.name} || Location: {self.location} || Calculate Location: {self.calculate_loc} || Power Location: {self.power_loc}"


class EdgeNode(Node):
    def __init__(
            self,
            node_id: int,
            name: str,
            location: Location,
            calculate_mec: int,
            power_mec: float
    ) -> None:
        super().__init__(node_id=node_id, name=name, location=location)
        self.calculate_mec = calculate_mec
        self.power_mec = power_mec

    def __repr__(self) -> str:
        return f"Node Name: {self.name} || Location: {self.location} || Calculate MEC: {self.calculate_mec} || Power MEC: {self.power_mec}"
