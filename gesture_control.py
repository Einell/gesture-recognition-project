import math
import os
import time
from pynput.keyboard import Key, Controller
import numpy as np
import mouse_controller as mc
import json
from pynput import keyboard as pynput_keyboard  # 引入pynput的Key

keyboard = Controller()

# --- 1. 配置加载 ---
CONFIG_FILE = 'gesture_config.json'
GESTURE_MAPPING = {}


def load_gesture_mapping():
    """加载手势配置文件"""
    global GESTURE_MAPPING
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

            # 转换配置中的字符串到 pynput.keyboard.Key 对象
            def get_key_object(key_str):
                # 尝试从 pynput.keyboard.Key 获取，否则返回普通字符
                key_map = {
                    "esc": pynput_keyboard.Key.esc,
                    "enter": pynput_keyboard.Key.enter,
                    "space": pynput_keyboard.Key.space,
                    "cmd": pynput_keyboard.Key.cmd,
                    "shift": pynput_keyboard.Key.shift,
                    "up": pynput_keyboard.Key.up,
                    "down": pynput_keyboard.Key.down,
                    "left": pynput_keyboard.Key.left,
                    "right": pynput_keyboard.Key.right,
                    # 更多可根据需要添加
                }
                # 如果是普通字符 (例如 'a', 'b', 'f' 等)，则直接返回
                return key_map.get(key_str.lower(), key_str)

            # 处理配置，确保所有键都是 pynput 可识别的 Key 对象或字符
            mapped_config = {}
            for gesture, action in config.items():
                if isinstance(action, list):
                    mapped_config[gesture] = [get_key_object(k) for k in action]
                elif isinstance(action, str):
                    mapped_config[gesture] = get_key_object(action)
                else:
                    mapped_config[gesture] = action

            GESTURE_MAPPING = mapped_config
            print(f"✅ 成功加载 {len(GESTURE_MAPPING)} 个手势映射。")

    except FileNotFoundError:
        print(f"❌ 警告: 未找到配置文件 '{CONFIG_FILE}'。将使用默认（硬编码）逻辑。")
        # 即使未找到，也继续执行，但 GESTURE_MAPPING 保持为空
    except json.JSONDecodeError:
        print(f"❌ 错误: 配置文件 '{CONFIG_FILE}' 格式错误。请检查 JSON 语法。")
        # 退出或设置空配置，这里选择设置空配置，防止崩溃
        GESTURE_MAPPING = {}


# 在模块加载时预先加载配置
load_gesture_mapping()


# --- 1. 配置加载结束 ---

class VolumeController:
    # ... (VolumeController 类保持不变)
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
            # 假定使用 macOS
            os.system(f"osascript -e 'set volume output volume {vol}'")
            self.last_volume = vol


VOLUME_CONTROLLER = VolumeController()


def perform_keyboard_action(action):
    """执行配置中的键盘动作"""
    if isinstance(action, list):
        # 组合键
        key_list = action
        with keyboard.pressed(key_list[0]):
            for key in key_list[1:]:
                # 注意：cmd/ctrl 键通常放在最前面
                if key != key_list[0]:
                    keyboard.press(key)
                    keyboard.release(key)
            # 由于 with 语句，第一个键在退出时才释放
        print(f"执行组合键: {' + '.join([str(k) for k in action])}")

    elif action is not None:
        # 单个键或字符
        keyboard.press(action)
        keyboard.release(action)
        print(f"执行按键: {action}")

    time.sleep(1.0)  # 维持原有的防抖/间隔时间


def execute_gesture_action(gesture, cap, display_img, hand_landmarks=None):
    try:
        # 获取图像尺寸 (height, width, channels)
        img_shape = display_img.shape if display_img is not None else (480, 640, 3)

        # --- 1. 实时/连续操作 (保持硬编码，保证稳定性和响应速度) ---
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
            # VOLUME_CONTROLLER.set_volume(hand_landmarks)
            return

        # --- 2. 鼠标点击 (保持硬编码) ---
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

        # --- 3. 自定义键盘映射操作 (新逻辑) ---
        elif gesture in GESTURE_MAPPING:
            action = GESTURE_MAPPING[gesture]
            perform_keyboard_action(action)
            return

        # --- 4. 兜底逻辑 (如果配置中没有，但原代码中有，则使用原逻辑) ---
        # 这一段可以删除，因为我们希望用户完全通过配置来定义离散手势
        # 但为了安全，保留一个未配置的提示
        else:
            print(f"警告: 手势 '{gesture}' 未在配置文件中找到，跳过执行。")


    except Exception as e:
        print(f"执行手势操作时出错: {e}")