import math
import os
import time
from pynput.keyboard import Key, Controller
import numpy as np
import mouse_controller as mc

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
        vol = np.interp(length, [self.min_dist, self.max_dist], [0, 100])
        vol = int(vol)

        # 只有当音量变化超过一定阈值（例如2）时才执行系统命令
        if abs(vol - self.last_volume) > 2:
            # macOS 设置音量的命令 (0-100)
            os.system(f"osascript -e 'set volume output volume {vol}'")
            print(f"音量调节: {vol}% (距离: {length:.3f})")
            self.last_volume = vol


# 实例化音量控制器
VOLUME_CONTROLLER = VolumeController()


def execute_gesture_action(gesture, cap, display_img, hand_landmarks=None):
    img_shape = display_img.shape if display_img is not None else (480, 640, 3)
    try:
        # ================= 1. 需要 hand_landmarks 的连续操作 =================
        if gesture == 'right_mouse':
            if hand_landmarks:
                mc.move_mouse(hand_landmarks, img_shape)
            return

        elif gesture == 'right_mouse_roll':
            if hand_landmarks:
                mc.scroll_mouse(hand_landmarks)
            return

        elif gesture == 'volume_control':
            # if hand_landmarks:
            #    VOLUME_CONTROLLER.set_volume(hand_landmarks)
            return

        # ================= 2. 新增：LSTM 动态手势控制 =================
        elif gesture == 'upglide':
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            print("LSTM: 上滑 (UpGlide)")
            time.sleep(0.5)

        elif gesture == 'downglide':
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            print("LSTM: 下滑 (DownGlide)")
            time.sleep(0.5)

        elif gesture == 'leftglide':
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            print("LSTM: 左滑 (LeftGlide)")
            time.sleep(0.5)

        elif gesture == 'rightglide':
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            print("LSTM: 右滑 (RightGlide)")
            time.sleep(0.5)

        elif gesture == 'open':
            keyboard.press('f')
            keyboard.release('f')
            print("LSTM: 张开 (Open) -> F键")
            time.sleep(0.5)

        elif gesture == 'zoom_in':
            with keyboard.pressed(Key.cmd):
                keyboard.press('+')  # 如果无效，尝试改成 '='
                keyboard.release('+')
            print("LSTM: 放大 (Zoom In)")
            time.sleep(2.0)

        elif gesture == 'zoom_out':
            with keyboard.pressed(Key.cmd):
                keyboard.press('-')
                keyboard.release('-')
            print("LSTM: 缩小 (Zoom Out)")
            time.sleep(2.0)

        # ================= 3. 原有：SVM 静态手势控制 (完全保留) =================

        # --- 鼠标点击 ---
        elif gesture == 'right_mouse_left_click':
            mc.left_click()
            print("执行左键点击")
            time.sleep(0.2)
            return

        elif gesture == 'right_mouse_right_click':
            mc.right_click()
            print("执行右键点击")
            time.sleep(0.3)
            return

        # --- 方向控制 (Thumb) ---
        elif gesture == 'right_thumb_up' or gesture == 'left_thumb_up':
            keyboard.press(Key.up)
            keyboard.release(Key.up)
            print("上一页/向上滚动 (Thumb Up)")
            time.sleep(1.0)

        elif gesture == 'right_thumb_down' or gesture == 'left_thumb_down':
            keyboard.press(Key.down)
            keyboard.release(Key.down)
            print("下一页/向下滚动 (Thumb Down)")
            time.sleep(1.0)

        # 这里是你刚才丢失的左右方向
        elif gesture == 'right_thumb_right' or gesture == 'left_thumb_right':
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            print("右一页/快进 (Thumb Right)")
            time.sleep(1.0)

        elif gesture == 'right_thumb_left' or gesture == 'left_thumb_left':
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            print("左一页/快退 (Thumb Left)")
            time.sleep(1.0)

        # --- 功能键 ---
        elif gesture == 'left_back':
            keyboard.press(Key.esc)
            keyboard.release(Key.esc)
            print("返回")
            time.sleep(1.0)

        elif gesture == 'left_palm' or gesture == 'right_palm':
            keyboard.press(Key.space)
            keyboard.release(Key.space)
            print("暂停/继续")
            time.sleep(1.0)

        elif gesture == 'right_ok' or gesture == 'left_ok':
            with keyboard.pressed(Key.cmd):
                keyboard.press('s')
                keyboard.release('s')
            print("执行保存")
            time.sleep(1.0)

        elif gesture == 'left_fist' or gesture == 'right_fist':
            #keyboard.press('f')
            #keyboard.release('f')
            print("待机动作")
            #time.sleep(1.0)

        elif gesture == 'left_L':
            with keyboard.pressed(Key.cmd):
                with keyboard.pressed(Key.shift):
                    keyboard.press('3')
                    keyboard.release('3')
            print("执行截屏")
            time.sleep(1.0)

    except Exception as e:
        print(f"执行手势操作时出错: {e}")