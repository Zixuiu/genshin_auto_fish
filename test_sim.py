from utils.render import *
from fisher.models import FishNet
from fisher.environment import *
import torch
import argparse
from matplotlib.animation import FFMpegWriter

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Test Genshin fishing with DQN')
parser.add_argument('--n_states', default=3, type=int)  # 状态空间维度
parser.add_argument('--n_actions', default=2, type=int)  # 动作空间维度
parser.add_argument('--step_tick', default=12, type=int)  # 步长
parser.add_argument('--model_dir', default='./output/fish_sim_net_399.pth', type=str)  # 模型保存路径
args = parser.parse_args()

if __name__ == '__main__':
    # 创建视频写入器
    writer = FFMpegWriter(fps=60)
    # 创建渲染器
    render = PltRender(call_back=writer.grab_frame)

    # 创建FishNet模型
    net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    # 创建Fishing_sim环境
    env = Fishing_sim(step_tick=args.step_tick, drawer=render, stop_tick=10000)

    # 加载模型参数
    net.load_state_dict(torch.load(args.model_dir))

    # 设置为评估模式
    net.eval()
    # 重置环境并获取初始状态
    state = env.reset()
    # 使用视频写入器保存动画
    with writer.saving(render.fig, 'out.mp4', 100):
        for i in range(2000):
            # 渲染环境
            env.render()

            # 将状态转换为张量并添加一个维度
            state = torch.FloatTensor(state).unsqueeze(0)
            # 使用模型选择动作
            action = net(state)
            action = torch.argmax(action, dim=1).numpy()
            # 执行动作并获取下一个状态、奖励和结束标志
            state, reward, done = env.step(action)
            # 如果游戏结束，则退出循环
            if done:
                break
