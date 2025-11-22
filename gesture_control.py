import math
import os
import time
import pyautogui
import cv2
from pynput.keyboard import Key, Controller
import numpy as np
import mouse_controller as mc

# 初始化键盘控制器
keyboard = Controller()


class VolumeController:
    def __init__(self):
        self.min_dist = 0.02  # 手指闭合的最小距离 (归一化坐标)
        self.max_dist = 0.20  # 手指张开的最大距离 (归一化坐标)
        self.last_volume = -1  # 记录上一次音量，防止重复发送命令

    def set_volume(self, hand_landmarks):
        """
        计算大拇指(4)和食指(8)的距离，并映射到系统音量
        """
        if not hand_landmarks:
            return

        # 获取大拇指指尖 (Landmark 4) 和 食指指尖 (Landmark 8) 的坐标
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]

        # 计算欧几里得距离
        length = math.hypot(index.x - thumb.x, index.y - thumb.y)

        # 将距离映射到音量范围 (0 - 100)
        # np.interp(当前值, [最小距离, 最大距离], [最小音量, 最大音量])
        vol = np.interp(length, [self.min_dist, self.max_dist], [0, 100])

        # 取整
        vol = int(vol)

        # 只有当音量变化超过一定阈值（例如2）时才执行系统命令，减少系统负担
        if abs(vol - self.last_volume) > 2:
            # macOS 设置音量的命令 (0-100)
            os.system(f"osascript -e 'set volume output volume {vol}'")
            print(f"音量调节: {vol}% (距离: {length:.3f})")
            self.last_volume = vol

        # 可选：在指尖之间画线（如果在主循环中传递了img，可以在这里画，但为了解耦暂不包含）


# 实例化音量控制器
VOLUME_CONTROLLER = VolumeController()

# 定义手势动作执行函数
def execute_gesture_action(gesture, cap, display_img,hand_landmarks=None):
    if gesture == 'right_mouse':
        # 只有在有手部关键点数据时才执行移动操作
        if hand_landmarks:
            mc.move_mouse(hand_landmarks)
            # 打印信息，以便在终端中追踪操作
            print("执行鼠标光标移动操作")
        return

    elif gesture == 'right_mouse_left_click':
        mc.left_click()
        print("执行鼠标左键点击操作")
        # 添加短暂延迟，防止手势短暂波动造成的连续点击
        time.sleep(0.2)
        return

    elif gesture == 'right_mouse_right_click':
        mc.right_click()
        print("执行鼠标右键点击操作")
        time.sleep(0.2)
        return
    try:
        if gesture == 'right_thumb_up' or 'left_thumb_up':  # 大拇指上
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            print("上一页/向上滚动")

        elif gesture == 'right_thumb_down' or 'left_thumb_down':  # 大拇指下
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            print("下一页/向下滚动")

        elif gesture == 'right_thumb_right' or 'left_thumb_right':  # 大拇指右
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            print("右一页/快进")

        elif gesture == 'right_thumb_left' or 'left_thumb_left':  # 大拇指左
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            print("左一页/快退")

        elif gesture == 'volume_control':
            if hand_landmarks:
                VOLUME_CONTROLLER.set_volume(hand_landmarks)
            return

        elif gesture == 'left_back':  #返回
            keyboard.press(Key.esc)
            keyboard.release(Key.esc)
            print("返回")

        elif gesture == 'left_palm' or 'right_palm':  # 音乐、视频暂停/继续
            # Space控制暂停/继续
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            print("执行暂停/继续操作")

        elif gesture == 'right_ok' or 'left_ok':    # 保存/确认
            with keyboard.pressed(Key.cmd):
                keyboard.press('s')
                keyboard.release('s')

            keyboard.press(Key.enter)
            keyboard.release(Key.enter)
            print("执行保存操作/确认")

        elif gesture == 'left_fist' or 'right_fist': # 拳头
            # 进入/退出全屏
            keyboard.press('f')
            keyboard.release('f')
            print("等待/全屏")

        elif gesture == 'left_L':  # 截屏
            # 截屏快捷键Command + Shift + 3
            with keyboard.pressed(Key.cmd):
                with keyboard.pressed(Key.shift):
                    keyboard.press('3')
                    keyboard.release('3')
            print("执行截屏操作")

        elif gesture == 'left_back':  # 返回
            # esc
            keyboard.press(Key.esc)
            keyboard.release(Key.esc)
            print("执行返回操作")




    except Exception as e:
        print(f"执行手势操作时出错: {e}")


