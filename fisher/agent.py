import torch 
from torch import nn 
from copy import deepcopy 
import numpy as np 

class DQN: 
    def __init__(self, base_net, batch_size, n_states, n_actions, memory_capacity=2000, epsilon=0.9, gamma=0.9, rep_frep=100, lr=0.01, reg=False): 
        self.eval_net = base_net 
        self.target_net = deepcopy(base_net) 

        self.batch_size=batch_size 
        self.epsilon=epsilon 
        self.gamma=gamma 
        self.n_states=n_states 
        self.n_actions=n_actions 
        self.memory_capacity=memory_capacity 
        self.rep_frep=rep_frep 
        self.reg=reg 

        self.learn_step_counter = 0  # 记录学习过程的步数
        self.memory_counter = 0  # 经验回放缓冲区的计数器

        # 经验回放缓冲区的列数取决于4个元素，s, a, r, s_，总共是N_STATES*2 + 2
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2 + 1)) 

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr) 
        self.loss_func = nn.MSELoss() 

    def choose_action(self, x): 
        # 根据epsilon贪婪法进行决策
        x = torch.FloatTensor(x).unsqueeze(0)  # 给输入状态x添加一个维度
        # 只输入一个样本
        if np.random.uniform() < self.epsilon:  # 贪婪
            # 使用epsilon贪婪法选择动作
            actions_value = self.eval_net.forward(x) 
            # torch.max()返回一个由沿着dim轴的最大值和相应的索引组成的张量
            # 我们需要的是索引，表示小车的动作
            action = actions_value if self.reg else torch.argmax(actions_value, dim=1).numpy()  # 返回argmax索引
        else:  # 随机
            action = np.random.rand(self.n_actions)*2-1 if self.reg else np.random.randint(0, self.n_actions) 
        return action 

    def store_transition(self, s, a, r, s_, done): 
        # 这个函数作为经验回放缓冲区
        transition = np.hstack((s, [a, r], s_, done))  # 水平堆叠这些向量
        # 如果容量已满，则使用索引替换旧的记忆
        index = self.memory_counter % self.memory_capacity 
        self.memory[index, :] = transition 
        self.memory_counter += 1 

    def train_step(self): 
        # 定义整个DQN的工作方式，包括采样一批经验，何时以及如何更新目标网络的参数，以及如何实现反向传播。

        # 每隔固定步数更新目标网络
        if self.learn_step_counter % self.rep_frep == 0: 
            # 将eval_net的参数赋值给target_net
            self.target_net.load_state_dict(self.eval_net.state_dict()) 
        self.learn_step_counter += 1 

        # 从缓冲区中确定采样批次的索引
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)  # 从缓冲区中随机选择一些数据
        # 从缓冲区中提取批次大小的经验
        b_memory = self.memory[sample_index, :] 
        # 从批次经验中提取向量或矩阵s,a,r,s_并将其转换为方便进行反向传播的torch变量
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]) 
        # 将长整型转换为张量
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)) 
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]) 
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states-1:-1]) 
        b_d = torch.FloatTensor(b_memory[:, -1]) # done 

        # 计算状态-动作对的Q值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1) 
        # 计算下一个状态的Q值
        q_next = self.target_net(b_s_).detach()  # 从计算图中分离出来，不进行反向传播
        # 选择最大的Q值
        q_target = b_r + self.gamma * (1-b_d) * q_next.max(dim=1)[0].view(self.batch_size, 1)  # (batch_size, 1) 
        loss = self.loss_func(q_eval, q_target) 

        self.optimizer.zero_grad()  # 将梯度重置为零
        loss.backward() 
        self.optimizer.step()  # 执行一步反向传播

class DDQN(DQN): 
    def __init__(self, base_net, batch_size, n_states, n_actions, memory_capacity=2000, epsilon=0.9, gamma=0.9, rep_frep=100, lr=0.01, reg=False): 
        super(DDQN, self).__init__(base_net, batch_size, n_states, n_actions, memory_capacity, epsilon, gamma, rep_frep, lr, reg) 

    def train_step(self): 
        # 定义整个DQN的工作方式，包括采样一批经验，何时以及如何更新目标网络的参数，以及如何实现反向传播。

        # 每隔固定步数更新目标网络
        if self.learn_step_counter % self.rep_frep == 0: 
            # 将eval_net的参数赋值给target_net
            self.target_net.load_state_dict(self.eval_net.state_dict()) 
        self.learn_step_counter += 1 

        # 从缓冲区中确定采样批次的索引
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)  # 从缓冲区中随机选择一些数据
        # 从缓冲区中提取批次大小的经验
        b_memory = self.memory[sample_index, :] 
        # 从批次经验中提取向量或矩阵s,a,r,s_并将其转换为方便进行反向传播的torch变量
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]) 
        # 将长整型转换为张量
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)) 
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]) 
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states-1:-1]) 
        b_d = torch.FloatTensor(b_memory[:, -1]) 

        # 计算状态-动作对的Q值
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1) 
        # 计算下一个状态的Q值
        #q_next = self.target_net(b_s_).detach()  # 从计算图中分离出来，不进行反向传播
        # 选择最大的Q值
        #q_target = b_r + self.gamma * q_next.max(dim=1)[0].view(self.batch_size, 1)  # (batch_size, 1) 

        # 双DQN
        q_eval_next = self.eval_net(b_s_).detach() 
        b_a_ = q_eval_next if self.reg else torch.argmax(q_eval_next, dim=1, keepdim=True)  # 获取eval_net的argmax_a'(Q(s', a')) 
        q_target_next = self.target_net(b_s_).detach() 
        if self.reg: 
            q_target = b_r + self.gamma * (1-b_d) * q_target_next*b_a_ 
        else: 
            q_target = b_r + self.gamma * (1-b_d) * q_target_next.gather(1, b_a_) 

        loss = self.loss_func(q_eval, q_target) 

        self.optimizer.zero_grad()  # 将梯度重置为零
        loss.backward() 
        self.optimizer.step()  # 执行一步反向传播