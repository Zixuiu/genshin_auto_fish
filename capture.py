import time 
from utils import * 

import keyboard 
import winsound 
import cv2 

# 设置截图计数器
i=0 
while True: 
    # 等待按下键盘上的 't' 键
    keyboard.wait('t') 
    # 使用pyautogui进行屏幕截图
    img = pyautogui.screenshot() 
    # 保存截图到指定路径
    img.save(f'img_tmp/{i}.png') 
    i+=1

# 读取退出按钮的图像
im_exit = cv2.imread('./imgs/exit.png') 

print('ok') 
# 等待按下键盘上的 't' 键
keyboard.wait('t') 

# 在当前屏幕上匹配退出按钮的位置
exit_pos = match_img(cap_raw(), im_exit) 
# 设置原神窗口的位置
gvars.genshin_window_rect_img = (exit_pos[0] - 32, exit_pos[1] - 19, DEFAULT_MONITOR_WIDTH, DEFAULT_MONITOR_HEIGHT) 

# 进行鱼类数据集的截图
for i in range(56,56+20): 
    img = cap() 
    img.save(f'fish_dataset/{i}.png') 
    time.sleep(0.5) 

# 发出声音提示截图完成
winsound.Beep(500, 500)