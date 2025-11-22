import pyautogui
import numpy as np
import time
import screeninfo

# 配置pyautogui的容错机制
# 禁用鼠标移到屏幕角落的保护，以允许全屏移动
pyautogui.FAILSAFE = False


class GestureMouseController:
    """
    一个专门用于手势控制鼠标的类。
    它处理坐标映射、移动平滑和点击操作。
    """

    # 获取主屏幕分辨率
    try:
        W_SCREEN, H_SCREEN = screeninfo.get_monitors()[0].width, screeninfo.get_monitors()[0].height
    except:
        # 备用方案，如果screeninfo无法工作
        W_SCREEN, H_SCREEN = 1920, 1080

        # 鼠标移动的平滑因子，值越大越平滑，响应越慢
    SMOOTHING_FACTOR = 7

    # 追踪上一帧的鼠标位置，用于平滑移动
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    # 用于将手部关键点映射到屏幕坐标的有效区域 (ROI)
    # MediaPipe landmarks 是归一化坐标 (0到1)。此范围定义了手部在摄像头画面中需要移动的区域。
    # 调整这些值可以改变鼠标移动的灵敏度和范围。
    # 摄像头画面X轴的有效范围
    CAM_X_MIN, CAM_X_MAX = 0.1, 0.9
    # 摄像头画面Y轴的有效范围
    CAM_Y_MIN, CAM_Y_MAX = 0.1, 0.8

    def __init__(self):
        # 重新初始化确保获取到正确的屏幕尺寸
        try:
            self.W_SCREEN, self.H_SCREEN = screeninfo.get_monitors()[0].width, screeninfo.get_monitors()[0].height
        except:
            pass  # 保持默认值

    def right_mouse(self, hand_landmarks):
        """
        根据手部关键点移动鼠标光标。
        使用食指尖 (Index Finger Tip, Landmark 8) 作为光标位置的依据。

        参数:
        hand_landmarks: MediaPipe返回的normalized HandLandmarks对象。
        """
        if not hand_landmarks:
            return

        # 获取食指尖的归一化坐标 (Landmark 8)
        index_tip = hand_landmarks.landmark[8]

        norm_x = index_tip.x
        norm_y = index_tip.y

        # 1. 映射：将归一化坐标映射到屏幕坐标
        # 使用 numpy.interp 将手部在摄像头ROI内的移动映射到整个屏幕
        target_x = np.interp(norm_x, [self.CAM_X_MIN, self.CAM_X_MAX], [0, self.W_SCREEN])
        target_y = np.interp(norm_y, [self.CAM_Y_MIN, self.CAM_Y_MAX], [0, self.H_SCREEN])

        # 限制坐标在屏幕范围内
        target_x = np.clip(target_x, 0, self.W_SCREEN)
        target_y = np.clip(target_y, 0, self.H_SCREEN)

        # 2. 平滑：使用平滑因子来减缓光标移动
        # 首次调用时初始化上一帧位置
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = target_x, target_y

        # 计算平滑后的当前位置
        self.curr_x = self.prev_x + (target_x - self.prev_x) / self.SMOOTHING_FACTOR
        self.curr_y = self.prev_y + (target_y - self.prev_y) / self.SMOOTHING_FACTOR

        # 3. 移动：执行鼠标移动
        pyautogui.moveTo(int(self.curr_x), int(self.curr_y))

        # 4. 更新：保存当前位置为下一帧的上一帧位置
        self.prev_x, self.prev_y = self.curr_x, self.curr_y

    def right_mouse_left_click(self):
        """ 执行一次鼠标左键点击 """
        pyautogui.click(button='left')

    def right_mouse_right_click(self):
        """ 执行一次鼠标右键点击 """
        pyautogui.click(button='right')


# 实例化控制器，确保只创建一次，并提供外部调用的简洁接口
MOUSE_CONTROLLER = GestureMouseController()


def move_mouse(hand_landmarks):
    """ 供外部调用的鼠标移动函数，传入 MediaPipe HandLandmarks 对象 """
    MOUSE_CONTROLLER.right_mouse(hand_landmarks)


def left_click():
    """ 供外部调用的左键点击函数 """
    MOUSE_CONTROLLER.right_mouse_left_click()


def right_click():
    """ 供外部调用的右键点击函数 """
    MOUSE_CONTROLLER.right_mouse_right_click()