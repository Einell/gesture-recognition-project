import pyautogui
import numpy as np
import time
import screeninfo

# 配置pyautogui的容错机制
pyautogui.FAILSAFE = False


class GestureMouseController:
    """
    一个专门用于手势控制鼠标的类。
    处理坐标映射、移动平滑、点击和滚动操作。
    """

    # 获取主屏幕分辨率
    try:
        W_SCREEN, H_SCREEN = screeninfo.get_monitors()[0].width, screeninfo.get_monitors()[0].height
    except:
        W_SCREEN, H_SCREEN = 1920, 1080

        # 移动平滑因子
    SMOOTHING_FACTOR = 4

    # 追踪上一帧的鼠标位置，用于平滑移动
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    # --- 新增：用于滚动的变量 ---
    # 追踪上一帧食指尖的归一化Y坐标用于滚动计算
    prev_scroll_y = 0
    # 滚动灵敏度乘数，越大滚动越快
    SCROLL_SENSITIVITY = 150
    # 滚动阈值，忽略微小的抖动
    SCROLL_THRESHOLD = 0.015

    # 摄像头画面有效区域 (ROI)
    CAM_X_MIN, CAM_X_MAX = 0.1, 0.9
    CAM_Y_MIN, CAM_Y_MAX = 0.1, 0.8

    def __init__(self):
        try:
            self.W_SCREEN, self.H_SCREEN = screeninfo.get_monitors()[0].width, screeninfo.get_monitors()[0].height
        except:
            pass

    # --- 现有的鼠标移动函数 (保持不变) ---
    def right_mouse(self, hand_landmarks):
        if not hand_landmarks:
            return
        index_tip = hand_landmarks.landmark[8]
        norm_x, norm_y = index_tip.x, index_tip.y

        target_x = np.interp(norm_x, [self.CAM_X_MIN, self.CAM_X_MAX], [0, self.W_SCREEN])
        target_y = np.interp(norm_y, [self.CAM_Y_MIN, self.CAM_Y_MAX], [0, self.H_SCREEN])
        target_x = np.clip(target_x, 0, self.W_SCREEN)
        target_y = np.clip(target_y, 0, self.H_SCREEN)

        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = target_x, target_y

        self.curr_x = self.prev_x + (target_x - self.prev_x) / self.SMOOTHING_FACTOR
        self.curr_y = self.prev_y + (target_y - self.prev_y) / self.SMOOTHING_FACTOR

        pyautogui.moveTo(int(self.curr_x), int(self.curr_y))
        self.prev_x, self.prev_y = self.curr_x, self.curr_y

    # --- 现有的点击函数 (保持不变) ---
    def right_mouse_left_click(self):
        pyautogui.click(button='left')

    def right_mouse_right_click(self):
        pyautogui.click(button='right')

    # --- 新增：鼠标滚动函数 ---
    def scroll_mouse(self, hand_landmarks):
        """
        根据食指指尖的垂直移动来控制鼠标滚动。
        手指向上移动 -> 向上滚动页面
        手指向下移动 -> 向下滚动页面
        """
        if not hand_landmarks:
            return

        # 使用食指指尖 (Landmark 8) 的Y坐标
        current_scroll_y = hand_landmarks.landmark[8].y

        # 如果是第一次调用，初始化上一帧位置
        if self.prev_scroll_y == 0:
            self.prev_scroll_y = current_scroll_y
            return

        # 计算Y轴移动差值 (注意：Y轴向下增加)
        # 如果现在Y < 之前Y，说明手指向上移动了，delta_y 为正
        delta_y = self.prev_scroll_y - current_scroll_y

        # 只有当移动距离超过阈值时才执行滚动，避免抖动
        if abs(delta_y) > self.SCROLL_THRESHOLD:
            # 计算滚动量：差值 * 灵敏度
            # pyautogui.scroll 中正值代表向上滚动，负值代表向下滚动
            scroll_amount = int(delta_y * self.SCROLL_SENSITIVITY)

            pyautogui.scroll(scroll_amount)
            # print(f"执行滚动: {scroll_amount}") # 调试用

        # 更新上一帧位置，用于下一次计算
        self.prev_scroll_y = current_scroll_y


# 实例化控制器
MOUSE_CONTROLLER = GestureMouseController()


# --- 模块对外接口 ---
def move_mouse(hand_landmarks):
    MOUSE_CONTROLLER.right_mouse(hand_landmarks)


def left_click():
    MOUSE_CONTROLLER.right_mouse_left_click()


def right_click():
    MOUSE_CONTROLLER.right_mouse_right_click()


# 新增接口
def scroll_mouse(hand_landmarks):
    MOUSE_CONTROLLER.scroll_mouse(hand_landmarks)