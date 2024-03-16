import pandas as pd
import torch
from zoo.env import Env
from network.Dueling_DDQN import D3QN_agent
from network.DQN import DQN_agent


def main():
    # 环境参数
    user_node_num = 20  # 用户节点数量
    edge_node_num = 6  # 边缘节点数量

    state_dim = 1 + user_node_num + 1 + edge_node_num * 4  # 假设状态空间维度为51
    action_dim = edge_node_num + 1  # 假设动作空间大小为5
    hid_shape = (64, 64)  # 隐藏层维度
    episodes = 200  # 训练回合数
    batch_size = 64  # 批次大小
    lr = 3e-2  # 学习率
    update_every = 10  # 更新频率

    env = Env(user_node_num=user_node_num, edge_node_num=edge_node_num)  # 创建环境实例

    # 创建D3QN agent实例
    agent = DQN_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        net_width=64,
        lr=lr,
        batch_size=batch_size,
        dvc='cuda' if torch.cuda.is_available() else 'cpu',
        Duel=True,
        Double=True,
        exp_noise=0.2,
        gamma=0.99
    )

    df = pd.DataFrame(columns=['Episode', 'Total Reward'])

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_step = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)

            # 训练模型
            if total_step % update_every == 0:
                for _ in range(update_every):
                    agent.train()

            state = next_state
            total_reward += reward
            total_step += 1

        print(f'Episode: {episode + 1}, Total Reward: {total_reward}')
        df.loc[len(df)] = {'Episode': episode + 1, 'Total Reward': total_reward}
    df.to_excel('DQN.xlsx', index=False)

if __name__ == "__main__":
    main()
