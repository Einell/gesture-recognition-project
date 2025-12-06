# 根据手势进行相应操作

import math
import os
import time
from pynput.keyboard import Key, Controller
import numpy as np
import src.controllers.mouse_controller as mc # 鼠标控制
import src.controllers.playmusic as playmusic #音乐控制

keyboard = Controller()
# 音量控制
class VolumeController:
    def __init__(self):
        self.min_dist = 0.02 # 最小距离
        self.max_dist = 0.20 # 最大距离
        self.last_volume = -1 # 上一次音量

    def set_volume(self, hand_landmarks):
        if not hand_landmarks: return
        # 获取拇指和食指指尖的坐标
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        # 计算距离
        length = math.hypot(index.x - thumb.x, index.y - thumb.y)
        # 映射到音量范围
        vol = np.interp(length, [self.min_dist, self.max_dist], [0, 100])
        vol = int(vol)
        if abs(vol - self.last_volume) > 2:
            os.system(f"osascript -e 'set volume output volume {vol}'")
            self.last_volume = vol

VOLUME_CONTROLLER = VolumeController()

# 执行手势操作
def execute_gesture_action(gesture, cap, display_img, hand_landmarks=None):
    try:
        # 获取图像尺寸
        img_shape = display_img.shape if display_img is not None else (480, 640, 3)

        # 连续手势操作
        # 鼠标移动
        if gesture == 'right_mouse':
            if hand_landmarks:
                mc.move_mouse(hand_landmarks, img_shape)
            return
        # 鼠标滚轮
        elif gesture == 'right_mouse_roll':
            if hand_landmarks:
                mc.scroll_mouse(hand_landmarks)
            return
        # 音量控制
        elif gesture == 'volume_control':
            if hand_landmarks:
                VOLUME_CONTROLLER.set_volume(hand_landmarks)
            return

        # 离散手势操作
        # 鼠标左键
        elif gesture == 'right_mouse_left_click':
            mc.left_click()
            print("执行左键点击")
            time.sleep(0.3)
            return
        # 鼠标右键
        elif gesture == 'right_mouse_right_click':
            mc.right_click()
            print("执行右键点击")
            time.sleep(0.3)
            return

        # 快捷键控制
        # 向上
        elif gesture == 'right_thumb_up' or gesture == 'left_thumb_up':
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            print("上一页/向下滚动")
            time.sleep(0.8)
            return

        # 向下
        elif gesture == 'right_thumb_down' or gesture == 'left_thumb_down':
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            print("下一页/向下滚动")
            time.sleep(1.0)
            return

        # 向右
        elif gesture == 'right_thumb_right' or gesture == 'left_thumb_right':
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            print("右一页/快进")
            time.sleep(1.0)
            return

        # 向左
        elif gesture == 'right_thumb_left' or gesture == 'left_thumb_left':
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            print("左一页/快退")
            time.sleep(1.0)
            return

        # 返回
        elif gesture == 'left_back':
            keyboard.press(Key.esc)
            keyboard.release(Key.esc)
            print("返回")
            time.sleep(1.0)
            return

        # 播放/暂停
        elif gesture == 'left_palm':
            # 停止音乐播放
            playmusic.stop_playback()

            keyboard.press(Key.space)
            keyboard.release(Key.space)
            print("暂停/继续 + 停止音乐")
            time.sleep(1.0)
            return

        # 保存
        elif gesture == 'right_ok' or gesture == 'left_ok':
            with keyboard.pressed(Key.cmd):
                keyboard.press('s')
                keyboard.release('s')
            print("保存")
            time.sleep(1.0)
            return

        # 静默手势，待机动作
        elif gesture == 'left_fist' or gesture == 'right_fist':
            return

        # 截屏
        elif gesture == 'left_L':
            with keyboard.pressed(Key.cmd):
                with keyboard.pressed(Key.shift):
                    keyboard.press('3')
                    keyboard.release('3')
            print("截屏")
            time.sleep(1.0)
            return
        # 打个响指，播放音乐
        elif gesture in ("right_snap", "left_snap"):
            # playmusic.play()
            playmusic.play("../../resources/music.mp3", duration=10.0) # 自定义音乐路径与时长
            print("播放音乐")
            time.sleep(1.5)
            return

        # 放大
        elif gesture == "zoom_in":
            with keyboard.pressed(Key.cmd):
                keyboard.press('+')
                keyboard.release('+')
            print("放大")
            time.sleep(1.5)
            return

        # 缩小
        elif gesture == "zoom_out":
            with keyboard.pressed(Key.cmd):
                keyboard.press('-')
                keyboard.release('-')
            print("缩小")
            time.sleep(1.5)
            return

        # 全屏/退出全屏
        elif gesture == "open" or gesture == 'close':
            keyboard.press('f')
            keyboard.release('f')
            print("全屏/退出全屏")
            time.sleep(1.5)
            return

    except Exception as e:
        print(f"执行手势操作时出错: {e}")