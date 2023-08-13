import keyboard  # 导入keyboard模块，用于监听键盘事件
import winsound  # 导入winsound模块，用于播放声音
from fisher.models import FishNet  # 导入FishNet模型，用于进行预测
from fisher.environment import *  # 导入环境模块，用于创建游戏环境
import torch  # 导入torch模块，用于进行模型加载和预测
import argparse  # 导入argparse模块，用于解析命令行参数

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Test Genshin fishing with DQN')
parser.add_argument('--n_states', default=3, type=int)  # 添加n_states参数，表示状态的数量，默认为3
parser.add_argument('--n_actions', default=2, type=int)  # 添加n_actions参数，表示动作的数量，默认为2
parser.add_argument('--step_tick', default=12, type=int)  # 添加step_tick参数，表示每一步的时间间隔，默认为12
parser.add_argument('--model_dir', default='./weights/fish_genshin_net.pth', type=str)  # 添加model_dir参数，表示模型的保存路径，默认为'./weights/fish_genshin_net.pth'
args = parser.parse_args()  # 解析命令行参数

if __name__ == '__main__':
    # 创建FishNet模型实例
    net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    # 创建游戏环境实例
    env = Fishing(delay=0.1, max_step=10000, show_det=True)

    # 加载模型参数
    net.load_state_dict(torch.load(args.model_dir))
    # 设置为评估模式
    net.eval()

    while True:
        # 播放声音提示用户按下'r'键开始钓鱼
        winsound.Beep(500, 500)
        keyboard.wait('r')  # 等待用户按下'r'键
        while True:
            if env.is_bite():  # 判断是否有鱼咬钩
                break
            time.sleep(0.5)  # 等待0.5秒
        # 播放声音提示用户有鱼咬钩
        winsound.Beep(700, 500)
        env.drag()  # 拖动鱼竿
        time.sleep(1)  # 等待1秒

        state = env.reset()  # 重置游戏环境
        for i in range(10000):
            state = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为张量，并添加一个维度
            action = net(state)  # 使用模型预测动作
            action = torch.argmax(action, dim=1).numpy()  # 获取预测动作的索引
            state, reward, done = env.step(action)  # 执行动作并获取下一个状态、奖励和是否结束的标志
            if done:  # 如果游戏结束，则跳出循环
                break
