from fisher.agent import DQN
from fisher.models import FishNet
from fisher.environment import *
import torch
import argparse
import os
import keyboard
import winsound

parser = argparse.ArgumentParser(description='Train Genshin fishing with DQN')
parser.add_argument('--batch_size', default=32, type=int)  # 批大小
parser.add_argument('--n_states', default=3, type=int)  # 状态数量
parser.add_argument('--n_actions', default=2, type=int)  # 动作数量
parser.add_argument('--step_tick', default=12, type=int)  # 步长
parser.add_argument('--n_episode', default=400, type=int)  # 训练轮数
parser.add_argument('--save_dir', default='./output', type=str)  # 模型保存路径
parser.add_argument('--resume', default='./weights/fish_sim_net_399.pth', type=str)  # 恢复训练时的模型路径
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)  # 创建FishNet模型
if args.resume:
    net.load_state_dict(torch.load(args.resume))  # 加载预训练模型

agent = DQN(net, args.batch_size, args.n_states, args.n_actions, memory_capacity=1000)  # 创建DQN agent
#env = Fishing_sim(step_tick=args.step_tick)  # 创建环境
env = Fishing(delay=0.1, max_step=150)

if __name__ == '__main__':
    # 开始训练
    print("\nCollecting experience...")
    net.train()
    for i_episode in range(args.n_episode):
        winsound.Beep(500, 500)  # 发出提示音
        keyboard.wait('r')  # 等待按下'r'键开始训练
        # 进行400轮的训练
        s = env.reset()  # 重置环境
        ep_r = 0  # 记录每轮的总奖励
        while True:
            if i_episode > 200 and i_episode % 20 == 0:
                env.render()  # 每隔20轮渲染一次环境

            a = agent.choose_action(s)  # 根据当前状态选择动作
            s_, r, done = env.step(a)  # 执行动作，获取奖励和下一个状态

            agent.store_transition(s, a, r, s_, int(done))  # 存储状态转移信息

            ep_r += r  # 更新总奖励
            if agent.memory_counter > agent.memory_capacity:
                agent.train_step()  # 当经验回放缓冲区满时，开始训练更新模型参数
                if done:
                    print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))  # 打印每轮的总奖励

            if done:
                break  # 如果游戏结束，则跳出循环

            s = s_  # 更新当前状态为下一个状态

        torch.save(net.state_dict(), os.path.join(args.save_dir, f'fish_ys_net_{i_episode}.pth'))  # 保存模型参数
