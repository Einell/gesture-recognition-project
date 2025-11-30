import math
import os
import time
from pynput.keyboard import Key, Controller
import numpy as np
import mouse_controller as mc

keyboard = Controller()


class VolumeController:
    def __init__(self):
        self.min_dist = 0.02
        self.max_dist = 0.20
        self.last_volume = -1

    def set_volume(self, hand_landmarks):
        if not hand_landmarks: return
        thumb = hand_landmarks.landmark[4]
        index = hand_landmarks.landmark[8]
        length = math.hypot(index.x - thumb.x, index.y - thumb.y)
        vol = np.interp(length, [self.min_dist, self.max_dist], [0, 100])
        vol = int(vol)
        if abs(vol - self.last_volume) > 2:
            os.system(f"osascript -e 'set volume output volume {vol}'")
            # print(f"音量调节: {vol}%") # 减少打印频率
            self.last_volume = vol


VOLUME_CONTROLLER = VolumeController()


def execute_gesture_action(gesture, cap, display_img, hand_landmarks=None):
    try:
        # 获取图像尺寸 (height, width, channels)
        img_shape = display_img.shape if display_img is not None else (480, 640, 3)

        # --- 1. 实时/连续操作 ---
        if gesture == 'right_mouse':
            if hand_landmarks:
                # 【关键修改】传入 img_shape 以支持 Frame Reduction 计算
                mc.move_mouse(hand_landmarks, img_shape)
            return

        elif gesture == 'right_mouse_roll':
            if hand_landmarks:
                mc.scroll_mouse(hand_landmarks)
            return

        elif gesture == 'volume_control':
            #if hand_landmarks:
                #VOLUME_CONTROLLER.set_volume(hand_landmarks)
            return

        # --- 2. 鼠标点击 ---
        elif gesture == 'right_mouse_left_click':
            mc.left_click()
            print("执行左键点击")
            time.sleep(0.2)  # 稍微降低延迟，提升连点体验
            return

        elif gesture == 'right_mouse_right_click':
            mc.right_click()
            print("执行右键点击")
            time.sleep(0.3)
            return

        # --- 3. 键盘映射 (保持不变) ---
        elif gesture == 'right_thumb_up' or gesture == 'left_thumb_up':
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            print("上一页")
            time.sleep(0.8)

        elif gesture == 'right_thumb_up                        ' or gesture == 'left_thumb_up':
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            print("上一页/向上滚动")
            time.sleep(1.0)

        elif gesture == 'right_thumb_down' or gesture == 'left_thumb_down':
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            print("下一页/向下滚动")
            time.sleep(1.0)

        elif gesture == 'right_thumb_right' or gesture == 'left_thumb_right':
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            print("右一页/快进")
            time.sleep(1.0)

        elif gesture == 'right_thumb_left' or gesture == 'left_thumb_left':
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            print("左一页/快退")
            time.sleep(1.0)

        elif gesture == 'left_back':
            keyboard.press(Key.esc)
            keyboard.release(Key.esc)
            print("返回")
            time.sleep(1.0)

        # 【重要修复】
        elif gesture == 'left_palm' or gesture == 'right_palm':
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            print("暂停/继续")
            time.sleep(1.0)

        # 【重要修复】
        elif gesture == 'right_ok' or gesture == 'left_ok':
            # 示例：保存操作
            with keyboard.pressed(Key.cmd):
                keyboard.press('s')
                keyboard.release('s')
            print("执行保存")
            time.sleep(1.0)

        # 【重要修复】
        elif gesture == 'left_fist' or gesture == 'right_fist':
            keyboard.press('f')
            keyboard.release('f')
            print("全屏切换")
            time.sleep(1.0)

        elif gesture == 'left_L':
            with keyboard.pressed(Key.cmd):
                with keyboard.pressed(Key.shift):
                    keyboard.press('3')
                    keyboard.release('3')
            print("执行截屏")
            time.sleep(1.0)

    except Exception as e:
        print(f"执行手势操作时出错: {e}")






