import os
import time
import pyautogui
import cv2
from pynput.keyboard import Key, Controller
import numpy as np


# 初始化键盘控制器
keyboard = Controller()


# 定义手势动作执行函数
def execute_gesture_action(gesture, cap, display_img):

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

        elif gesture == 'left_finger_up':  #左食指上
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
            print("音量大")

        elif gesture == 'left_finger_down':  #左食指下
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
            print("音量小")

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


    except Exception as e:
        print(f"执行手势操作时出错: {e}")


