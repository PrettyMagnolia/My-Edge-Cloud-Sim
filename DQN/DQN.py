import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy
from utils import demo1


# 定义一个函数用于构建神经网络层
def build_net(layer_shape, activation, output_activation):
    '''Build networks with For loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


# 定义 Q 网络类
class Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q


# 定义 Dueling Q 网络类
class Duel_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Duel_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape)
        self.hidden = build_net(layers, nn.ReLU, nn.ReLU)
        self.V = nn.Linear(hid_shape[-1], 1)
        self.A = nn.Linear(hid_shape[-1], action_dim)

    def forward(self, s):
        s = self.hidden(s)
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


# 定义 DQN agent 类
class DQN_agent(object):
    def __init__(self, **kwargs):
        # 使用kwargs初始化超参数
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))
        # 初始化 Q 网络
        if self.Duel:
            self.q_net = Duel_Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        else:
            self.q_net = Q_Net(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.q_target = copy.deepcopy(self.q_net)
        # 冻结目标网络的参数
        for p in self.q_target.parameters():
            p.requires_grad = False

    # 选择动作
    def select_action(self, state, deterministic):  # 仅用于与环境交互时使用
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
            if deterministic:
                a = self.q_net(state).argmax().item()
            else:
                if np.random.rand() < self.exp_noise:
                    a = np.random.randint(0, self.action_dim)
                else:
                    a = self.q_net(state).argmax().item()
        return a

    # 训练方法
    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        '''计算目标 Q 值'''
        with torch.no_grad():
            if self.Double:
                argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
                max_q_next = self.q_target(s_next).gather(1, argmax_a)
            else:
                max_q_next = self.q_target(s_next).max(1)[0].unsqueeze(1)
            target_Q = r + (~dw) * self.gamma * max_q_next  # dw: die or win

        # 获取当前 Q 估计值
        current_q = self.q_net(s)
        current_q_a = current_q.gather(1, a)

        q_loss = F.mse_loss(current_q_a, target_Q)
        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()

        # 更新冻结的目标网络参数
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # 保存模型
    def save(self, algo, EnvName, steps):
        torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.pth".format(algo, EnvName, steps))

    # 加载模型
    def load(self, algo, EnvName, steps):
        self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo, EnvName, steps)))
        self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo, EnvName, steps)))


# 定义重放缓冲区类
class ReplayBuffer(object):
    def __init__(self, state_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    # 添加样本到重放缓冲区
    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # 从重放缓冲区中采样样本
    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


if __name__ == '__main__':
    demo1()