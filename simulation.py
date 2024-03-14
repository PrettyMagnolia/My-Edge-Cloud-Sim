from zoo.env import Env
from zoo.scenario import Scenario1
from zoo.task import Task
import random
import pandas as pd

user_node_num = 20
edge_node_num = 6


def get_task(exp_type: str, task_num: int, task_size: int, user_node_num: int, edge_node_num: int):
    task_info_list = []
    if exp_type == 'all local':
        for i in range(task_num):
            task_info = {
                'generate_time': i,
                'task_id': i,
                'task_size': task_size,
                'src_name': 'u' + str(i % user_node_num),
                'exc_type': 'local',
                'dst_name': None
            }
            task_info_list.append(task_info)
    elif exp_type == 'all edge':
        for i in range(task_num):
            task_info = {
                'generate_time': i,
                'task_id': i,
                'task_size': task_size,
                'src_name': 'u' + str(i % user_node_num),
                'exc_type': 'edge',
                'dst_name': 'e' + str(random.randint(0, edge_node_num - 1))
            }
            task_info_list.append(task_info)
    elif exp_type == 'random':
        for i in range(task_num):
            task_info = {
                'generate_time': i,
                'task_id': i,
                'task_size': task_size,
                'src_name': 'u' + str(i % user_node_num),
                'exc_type': 'edge' if random.random() > 0.5 else 'local',
                'dst_name': 'e' + str(random.randint(0, 5))
            }
            task_info_list.append(task_info)
    return task_info_list


def exp_by_task_size():
    start_task_size = 100
    end_task_size = 1000
    task_num = 400
    df_time = pd.DataFrame({'task_size': ['all local', 'all edge', 'random']})
    df_energy = pd.DataFrame({'task_size': ['all local', 'all edge', 'random']})
    for task_size in range(start_task_size, end_task_size + 1, 50):
        random.seed(task_size)
        # 创建新的一列
        df_time[str(task_size)] = [0, 0, 0]
        df_energy[str(task_size)] = [0, 0, 0]

        for exp_type in ['all local', 'all edge', 'random']:
            scenario = Scenario1(user_node_num=user_node_num, edge_node_num=edge_node_num, random_seed=task_size)
            env = Env(scenario=scenario)
            task_info_list = get_task(exp_type=exp_type, task_num=task_num, task_size=task_size, user_node_num=user_node_num, edge_node_num=edge_node_num)
            until = 1
            for task_info in task_info_list:
                generated_time = task_info['generate_time']
                task = Task(task_id=task_info['task_id'],
                            task_size=task_info['task_size'],
                            src_name=task_info['src_name'])

                while True:
                    if env.now == generated_time:
                        env.process(task=task, exc_type=task_info['exc_type'], dst_name=task_info['dst_name'])
                        break

                    try:
                        env.run(until=until)  # execute the simulation step by step
                    except Exception as e:
                        print(e)
                    until += 1
            time, energy = env.close()
            df_time.loc[df_time['task_size'] == exp_type, str(task_size)] = time
            df_energy.loc[df_energy['task_size'] == exp_type, str(task_size)] = energy
    df_time.to_excel(f'task_num_{task_num}_time.xlsx', index=False)
    df_energy.to_excel(f'task_num_{task_num}_energy.xlsx', index=False)


def exp_by_task_num():
    start_task_num = 100
    end_task_num = 400
    task_size = 1000
    df_time = pd.DataFrame({'task_num': ['all local', 'all edge', 'random']})
    df_energy = pd.DataFrame({'task_num': ['all local', 'all edge', 'random']})

    for task_num in range(start_task_num, end_task_num + 1, 20):
        # 创建新的一列
        df_time[str(task_num)] = [0, 0, 0]
        df_energy[str(task_num)] = [0, 0, 0]

        user_node_num = task_num

        for exp_type in ['all local', 'all edge', 'random']:
            scenario = Scenario1(user_node_num=user_node_num, edge_node_num=edge_node_num, random_seed=task_num)
            env = Env(scenario=scenario)
            task_info_list = get_task(exp_type=exp_type, task_num=task_num, task_size=task_size, user_node_num=user_node_num, edge_node_num=edge_node_num)
            until = 1
            for task_info in task_info_list:
                generated_time = task_info['generate_time']
                task = Task(task_id=task_info['task_id'],
                            task_size=task_info['task_size'],
                            src_name=task_info['src_name'])

                while True:
                    if env.now == generated_time:
                        env.process(task=task, exc_type=task_info['exc_type'], dst_name=task_info['dst_name'])
                        break

                    try:
                        env.run(until=until)  # execute the simulation step by step
                    except Exception as e:
                        print(e)
                    until += 1
            time, energy = env.close()
            df_time.loc[df_time['task_num'] == exp_type, str(task_num)] = time
            df_energy.loc[df_energy['task_num'] == exp_type, str(task_num)] = energy
    df_time.to_excel(f'task_size_{task_size}_time.xlsx', index=False)
    df_energy.to_excel(f'task_size_{task_size}_energy.xlsx', index=False)


if __name__ == '__main__':
    exp_by_task_num()
