# 基于pyautogui实现的鼠标控制模块
# 追踪食指指尖的位置并映射到屏幕

import pyautogui
import numpy as np
import time
import screeninfo


pyautogui.FAILSAFE = False # 禁用pyautogui的安全机制，避免意外触发

# 手势控制鼠标，利用屏幕映射
class GestureMouseController:

    def __init__(self):
        # 获取主屏幕分辨率
        try:
            self.W_SCREEN, self.H_SCREEN = screeninfo.get_monitors()[0].width, screeninfo.get_monitors()[0].height
        except:
            self.W_SCREEN, self.H_SCREEN = 1920, 1080

        self.frameR = 100  #帧缩减像素，修改控制框的大小，frameR越大框越小，鼠标控制越灵敏
        self.smoothening = 5  # 平滑系数，值越大越平滑，但延迟越高

        # 追踪上一帧的鼠标位置，用于平滑
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0

        # 鼠标滚动参数
        self.prev_scroll_y = 0 # 上一次滚动的y坐标
        self.SCROLL_SENSITIVITY = 150 # 滚动速度
        self.SCROLL_THRESHOLD = 0.015 # 滚动阈值，值越小越灵敏

    # 鼠标控制
    def right_mouse(self, hand_landmarks, img_shape):
        if not hand_landmarks:
            return

        hCam, wCam, _ = img_shape

        # 获取食指指尖lm8的坐标
        index_tip = hand_landmarks.landmark[8]
        x1 = int(index_tip.x * wCam)
        y1 = int(index_tip.y * hCam)

        # 坐标转换，将控制区域映射到屏幕分辨率
        x3 = np.interp(x1, (self.frameR, wCam - self.frameR), (0, self.W_SCREEN))
        y3 = np.interp(y1, (self.frameR, hCam - self.frameR), (0, self.H_SCREEN))

        # 平滑处理
        self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
        self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening

        # 边界限制
        self.clocX = np.clip(self.clocX, 0, self.W_SCREEN)
        self.clocY = np.clip(self.clocY, 0, self.H_SCREEN)

        # 移动鼠标
        pyautogui.moveTo(self.clocX, self.clocY)

        # 更新上一帧位置
        self.plocX, self.plocY = self.clocX, self.clocY

    # 鼠标左键点击
    def right_mouse_left_click(self):
        pyautogui.click(button='left')
    # 鼠标右键点击
    def right_mouse_right_click(self):
        pyautogui.click(button='right')
    # 鼠标滚轮控制
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

# 实例化
MOUSE_CONTROLLER = GestureMouseController()


# 接口
def move_mouse(hand_landmarks, img_shape):
    MOUSE_CONTROLLER.right_mouse(hand_landmarks, img_shape)
def left_click():
    MOUSE_CONTROLLER.right_mouse_left_click()
def right_click():
    MOUSE_CONTROLLER.right_mouse_right_click()
def scroll_mouse(hand_landmarks):
    MOUSE_CONTROLLER.scroll_mouse(hand_landmarks)