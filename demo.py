from core.env import Env
from zoo.scenario import Scenario1
from zoo.task import Task


def main():
    print("Hello, World!")
    # Create the Env
    env = Env(scenario=Scenario1())

    # Begin Simulation
    task = Task(task_id=0,
                task_size=1000,  # 单位Kb
                src_name='u0')

    env.process(task=task, exc_type='local', dst_name='n1')

    env.run(until=10)  # execute the simulation all at once

    env.close()


if __name__ == '__main__':
    main()
