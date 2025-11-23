import pyautogui
import numpy as np
import time
import screeninfo

# 配置pyautogui的容错机制
pyautogui.FAILSAFE = False


class GestureMouseController:
    """
    一个专门用于手势控制鼠标的类。
    已集成 Frame Reduction (帧区域缩减) 和平滑算法。
    """

    def __init__(self):
        # 获取主屏幕分辨率
        try:
            self.W_SCREEN, self.H_SCREEN = screeninfo.get_monitors()[0].width, screeninfo.get_monitors()[0].height
        except:
            self.W_SCREEN, self.H_SCREEN = 1920, 1080

        # ================= 虚拟鼠标核心参数 =================
        self.frameR = 200  # Frame Reduction: 帧缩减像素 (值越大，移动鼠标所需的动作幅度越小)
        self.smoothening = 5  # 平滑系数 (值越大越平滑，但延迟越高)
        # ==================================================

        # 追踪上一帧的鼠标位置 (用于平滑)
        self.plocX, self.plocY = 0, 0  # Previous Location
        self.clocX, self.clocY = 0, 0  # Current Location

        # 滚动相关变量
        self.prev_scroll_y = 0
        self.SCROLL_SENSITIVITY = 150
        self.SCROLL_THRESHOLD = 0.015

    def right_mouse(self, hand_landmarks, img_shape):
        """
        移动鼠标逻辑
        :param hand_landmarks: 手部关键点
        :param img_shape: 摄像头画面的形状 (height, width, channels) 用于计算像素坐标
        """
        if not hand_landmarks:
            return

        hCam, wCam, _ = img_shape

        # 1. 获取食指指尖 (Landmark 8) 的坐标
        # 注意：这里我们将其转换为像素坐标，以便使用 Frame Reduction 逻辑
        index_tip = hand_landmarks.landmark[8]
        x1 = int(index_tip.x * wCam)
        y1 = int(index_tip.y * hCam)

        # 2. 坐标转换 (核心逻辑：将 FrameR 区域映射到屏幕分辨率)
        # np.interp(当前值, [输入范围下限, 输入范围上限], [输出范围下限, 输出范围上限])

        # X轴映射：注意 wCam - self.frameR 是为了处理镜像翻转后的逻辑
        # 如果你觉得鼠标左右反了，可以调整 x1 的输入范围
        x3 = np.interp(x1, (self.frameR, wCam - self.frameR), (0, self.W_SCREEN))

        # Y轴映射
        y3 = np.interp(y1, (self.frameR, hCam - self.frameR), (0, self.H_SCREEN))

        # 3. 平滑处理 (Smoothen Values)
        # 当前位置 = 上次位置 + (目标位置 - 上次位置) / 平滑系数
        self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
        self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening

        # 4. 边界限制 (防止坐标溢出)
        self.clocX = np.clip(self.clocX, 0, self.W_SCREEN)
        self.clocY = np.clip(self.clocY, 0, self.H_SCREEN)

        # 5. 移动鼠标
        # pyautogui.moveTo(x, y)
        # 使用 int() 确保坐标是整数
        final_x = self.W_SCREEN - self.clocX
        pyautogui.moveTo(self.clocX, self.clocY)

        # 6. 更新上一帧位置
        self.plocX, self.plocY = self.clocX, self.clocY

    # --- 点击和滚动保持不变，或者根据需要微调 ---
    def right_mouse_left_click(self):
        pyautogui.click(button='left')

    def right_mouse_right_click(self):
        pyautogui.click(button='right')

    def scroll_mouse(self, hand_landmarks):
        if not hand_landmarks:
            return
        current_scroll_y = hand_landmarks.landmark[8].y
        if self.prev_scroll_y == 0:
            self.prev_scroll_y = current_scroll_y
            return
        delta_y = self.prev_scroll_y - current_scroll_y
        if abs(delta_y) > self.SCROLL_THRESHOLD:
            scroll_amount = int(delta_y * self.SCROLL_SENSITIVITY)
            pyautogui.scroll(scroll_amount)
        self.prev_scroll_y = current_scroll_y


# 实例化控制器
MOUSE_CONTROLLER = GestureMouseController()


# --- 模块对外接口 ---
def move_mouse(hand_landmarks, img_shape):
    # 注意：这里新增了 img_shape 参数
    MOUSE_CONTROLLER.right_mouse(hand_landmarks, img_shape)


def left_click():
    MOUSE_CONTROLLER.right_mouse_left_click()


def right_click():
    MOUSE_CONTROLLER.right_mouse_right_click()


def scroll_mouse(hand_landmarks):
    MOUSE_CONTROLLER.scroll_mouse(hand_landmarks)