import torch
import argparse
import os
from fisher.agent import DQN
from fisher.models import FishNet
from fisher.environment import Fishing_sim
from utils.render import PltRender

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train Genshin fishing simulation with DQN')
parser.add_argument('--batch_size', default=32, type=int, help='批大小')
parser.add_argument('--n_states', default=3, type=int, help='状态空间维度')
parser.add_argument('--n_actions', default=2, type=int, help='动作空间维度')
parser.add_argument('--step_tick', default=12, type=int, help='每个步骤的时间间隔')
parser.add_argument('--n_episode', default=400, type=int, help='训练的总回合数')
parser.add_argument('--save_dir', default='./output', type=str, help='保存模型的目录')
parser.add_argument('--resume', default=None, type=str, help='恢复训练时的模型路径')
args = parser.parse_args()

# 创建保存模型的目录
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# 创建FishNet模型
net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
if args.resume:
    net.load_state_dict(torch.load(args.resume))

# 创建DQN代理
agent = DQN(net, args.batch_size, args.n_states, args.n_actions, memory_capacity=2000)

# 创建环境
env = Fishing_sim(step_tick=args.step_tick, drawer=PltRender())

if __name__ == '__main__':
    # 开始训练
    print("\n收集经验...")
    net.train()
    for i_episode in range(args.n_episode):
        s = env.reset()  # 重置环境并获取初始状态
        ep_r = 0  # 记录当前回合的总奖励
        while True:
            if i_episode > 200 and i_episode % 20 == 0:
                env.render()  # 每隔20个回合渲染一次环境

            a = agent.choose_action(s)  # 根据当前状态选择动作
            s_, r, done = env.step(a)  # 执行动作并获取奖励、下一个状态和是否结束的标志

            agent.store_transition(s, a, r, s_, int(done))  # 存储状态转换

            ep_r += r  # 更新当前回合的总奖励

            if agent.memory_counter > agent.memory_capacity:
                agent.train_step()  # 当经验回放缓冲区满时，开始训练更新网络参数
                if done:
                    print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

            if done:
                break  # 如果游戏结束，则跳出循环

            s = s_  # 更新当前状态为下一个状态

    torch.save(net.state_dict(), os.path.join(args.save_dir, f'fish_sim_net_{i_episode}.pth'))  # 保存训练好的模型
