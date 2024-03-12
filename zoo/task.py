class Task:
    def __init__(
            self,
            task_id: int,
            task_size: float,
            src_name: str,
    ) -> None:
        self.task_id = task_id
        self.task_size = task_size # Kb
        self.src_name = src_name
