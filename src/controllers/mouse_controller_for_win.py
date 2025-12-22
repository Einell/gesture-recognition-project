# windows_mouse_controller.py
# 基于pyautogui实现的Windows鼠标控制模块
# 追踪食指指尖的位置并映射到屏幕

import pyautogui
import numpy as np
import screeninfo
import ctypes
from ctypes import wintypes
import time

# 禁用pyautogui的安全机制，避免意外触发
pyautogui.FAILSAFE = False

# Windows API 常量
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000  # 水平滚轮

# 加载 Windows API
try:
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    HAS_WINDOWS_API = True
except:
    HAS_WINDOWS_API = False
    print("警告: 无法加载 Windows API，使用 pyautogui 作为备用")


def get_windows_scaling():
    """获取 Windows 显示缩放比例"""
    try:
        # 方法1: 使用 ctypes 获取 DPI 缩放
        user32.SetProcessDPIAware()
        hdc = user32.GetDC(0)
        dpi_x = gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        user32.ReleaseDC(0, hdc)
        scaling = dpi_x / 96.0  # 96 DPI 是 100% 缩放
        return scaling
    except:
        return 1.0


def get_windows_screen_info():
    """获取 Windows 屏幕信息（考虑缩放）"""
    try:
        # 获取物理屏幕分辨率
        user32.SetProcessDPIAware()
        width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
        height = user32.GetSystemMetrics(1)  # SM_CYSCREEN

        # 获取虚拟屏幕分辨率（考虑多显示器）
        virtual_width = user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
        virtual_height = user32.GetSystemMetrics(79)  # SM_CYVIRTUALSCREEN
        virtual_left = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
        virtual_top = user32.GetSystemMetrics(77)  # SM_YVIRTUALSCREEN

        scaling = get_windows_scaling()

        return {
            'width': width,
            'height': height,
            'virtual_width': virtual_width,
            'virtual_height': virtual_height,
            'virtual_left': virtual_left,
            'virtual_top': virtual_top,
            'scaling': scaling
        }
    except:
        return {
            'width': 1920,
            'height': 1080,
            'virtual_width': 1920,
            'virtual_height': 1080,
            'virtual_left': 0,
            'virtual_top': 0,
            'scaling': 1.0
        }


class WindowsMouseAPI:
    """Windows 原生鼠标控制（更精确）"""

    @staticmethod
    def mouse_move(x, y):
        """使用 Windows API 移动鼠标"""
        if not HAS_WINDOWS_API:
            return False

        try:
            # 转换为绝对坐标 (0-65535)
            screen_info = get_windows_screen_info()
            abs_x = int(x * 65535 / screen_info['virtual_width'])
            abs_y = int(y * 65535 / screen_info['virtual_height'])

            user32.mouse_event(0x8000 | 0x0001, abs_x, abs_y, 0, 0)
            return True
        except:
            return False

    @staticmethod
    def mouse_click(button='left'):
        """使用 Windows API 点击鼠标"""
        if not HAS_WINDOWS_API:
            return False

        try:
            if button == 'left':
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.01)  # 短暂延迟
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif button == 'right':
                user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                time.sleep(0.01)
                user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
            elif button == 'middle':
                user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                time.sleep(0.01)
                user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
            return True
        except:
            return False

    @staticmethod
    def mouse_scroll(amount):
        """使用 Windows API 滚动"""
        if not HAS_WINDOWS_API:
            return False

        try:
            # amount 为正数向上滚动，负数向下滚动
            user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(amount * 120), 0)
            return True
        except:
            return False

    @staticmethod
    def mouse_hscroll(amount):
        """水平滚动"""
        if not HAS_WINDOWS_API:
            return False

        try:
            user32.mouse_event(MOUSEEVENTF_HWHEEL, 0, 0, int(amount * 120), 0)
            return True
        except:
            return False


class EnhancedMouseController:
    """增强的鼠标控制器，支持多种控制模式"""

    def __init__(self, mode='auto'):
        """
        初始化鼠标控制器

        参数:
        - mode: 'auto' (自动选择), 'windows' (Windows API), 'pyautogui' (pyautogui)
        """
        self.mode = mode
        self.windows_api = WindowsMouseAPI()

        # 获取屏幕信息
        screen_info = get_windows_screen_info()
        self.W_SCREEN = screen_info['width']
        self.H_SCREEN = screen_info['height']
        self.W_VIRTUAL = screen_info['virtual_width']
        self.H_VIRTUAL = screen_info['virtual_height']
        self.VIRTUAL_LEFT = screen_info['virtual_left']
        self.VIRTUAL_TOP = screen_info['virtual_top']
        self.scaling = screen_info['scaling']

        print(f"屏幕分辨率: {self.W_SCREEN}x{self.H_SCREEN}")
        print(f"虚拟屏幕: {self.W_VIRTUAL}x{self.H_VIRTUAL} (缩放: {self.scaling:.1f})")

        # 鼠标控制参数
        self.frameR = 100  # 帧缩减像素，控制框的大小
        self.smoothening = 5  # 平滑系数

        # 鼠标位置追踪
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0

        # 滚轮控制
        self.prev_scroll_y = 0
        self.SCROLL_SENSITIVITY = 150
        self.SCROLL_THRESHOLD = 0.015

        # 点击防抖
        self.last_click_time = 0
        self.CLICK_COOLDOWN = 0.3  # 点击冷却时间（秒）

        # 性能优化
        self.last_update_time = 0
        self.UPDATE_INTERVAL = 0.01  # 更新间隔（秒）

        # 鼠标加速
        self.enable_acceleration = True
        self.acceleration_factor = 1.5

        # 历史位置用于平滑
        self.position_history = []
        self.MAX_HISTORY = 3

    def _get_control_mode(self):
        """获取控制模式"""
        if self.mode == 'auto':
            return 'windows' if HAS_WINDOWS_API else 'pyautogui'
        return self.mode

    def _move_mouse_windows(self, x, y):
        """使用 Windows API 移动鼠标"""
        # 调整坐标到虚拟屏幕
        adjusted_x = x + self.VIRTUAL_LEFT
        adjusted_y = y + self.VIRTUAL_TOP

        # 应用缩放调整
        if self.scaling != 1.0:
            adjusted_x /= self.scaling
            adjusted_y /= self.scaling

        return self.windows_api.mouse_move(adjusted_x, adjusted_y)

    def _move_mouse_pyautogui(self, x, y):
        """使用 pyautogui 移动鼠标"""
        try:
            # pyautogui 会自动处理缩放
            pyautogui.moveTo(x, y, _pause=False)
            return True
        except:
            return False

    def _click_mouse_windows(self, button='left'):
        """使用 Windows API 点击"""
        if time.time() - self.last_click_time < self.CLICK_COOLDOWN:
            return False

        success = self.windows_api.mouse_click(button)
        if success:
            self.last_click_time = time.time()
        return success

    def _click_mouse_pyautogui(self, button='left'):
        """使用 pyautogui 点击"""
        if time.time() - self.last_click_time < self.CLICK_COOLDOWN:
            return False

        try:
            if button == 'left':
                pyautogui.click(button='left', _pause=False)
            elif button == 'right':
                pyautogui.click(button='right', _pause=False)
            elif button == 'middle':
                pyautogui.click(button='middle', _pause=False)

            self.last_click_time = time.time()
            return True
        except:
            return False

    def _scroll_mouse_windows(self, amount):
        """使用 Windows API 滚动"""
        return self.windows_api.mouse_scroll(amount)

    def _scroll_mouse_pyautogui(self, amount):
        """使用 pyautogui 滚动"""
        try:
            pyautogui.scroll(amount, _pause=False)
            return True
        except:
            return False

    def update_mouse_position(self, hand_landmarks, img_shape):
        """更新鼠标位置"""
        if not hand_landmarks:
            return False

        # 性能优化：限制更新频率
        current_time = time.time()
        if current_time - self.last_update_time < self.UPDATE_INTERVAL:
            return False
        self.last_update_time = current_time

        hCam, wCam, _ = img_shape

        # 获取食指指尖坐标
        index_tip = hand_landmarks.landmark[8]
        x1 = int(index_tip.x * wCam)
        y1 = int(index_tip.y * hCam)

        # 坐标转换到屏幕
        x3 = np.interp(x1, (self.frameR, wCam - self.frameR), (0, self.W_SCREEN))
        y3 = np.interp(y1, (self.frameR, hCam - self.frameR), (0, self.H_SCREEN))

        # 鼠标加速（可选）
        if self.enable_acceleration:
            distance = np.sqrt((x3 - self.plocX) ** 2 + (y3 - self.plocY) ** 2)
            if distance > 50:  # 快速移动时加速
                acceleration = 1 + (distance - 50) / 100 * self.acceleration_factor
                x3 = self.plocX + (x3 - self.plocX) * acceleration
                y3 = self.plocY + (y3 - self.plocY) * acceleration

        # 平滑处理（使用历史位置）
        self.position_history.append((x3, y3))
        if len(self.position_history) > self.MAX_HISTORY:
            self.position_history.pop(0)

        # 计算加权平均
        weights = np.linspace(0.5, 1.0, len(self.position_history))
        weights = weights / weights.sum()

        avg_x = sum(pos[0] * w for pos, w in zip(self.position_history, weights))
        avg_y = sum(pos[1] * w for pos, w in zip(self.position_history, weights))

        # 最终平滑
        self.clocX = self.plocX + (avg_x - self.plocX) / self.smoothening
        self.clocY = self.plocY + (avg_y - self.plocY) / self.smoothening

        # 边界限制
        self.clocX = np.clip(self.clocX, 0, self.W_SCREEN - 1)
        self.clocY = np.clip(self.clocY, 0, self.H_SCREEN - 1)

        # 移动鼠标
        mode = self._get_control_mode()
        success = False

        if mode == 'windows':
            success = self._move_mouse_windows(self.clocX, self.clocY)
        else:
            success = self._move_mouse_pyautogui(self.clocX, self.clocY)

        # 更新上一帧位置
        self.plocX, self.plocY = self.clocX, self.clocY

        return success

    def left_click(self):
        """左键点击"""
        mode = self._get_control_mode()
        if mode == 'windows':
            return self._click_mouse_windows('left')
        else:
            return self._click_mouse_pyautogui('left')

    def right_click(self):
        """右键点击"""
        mode = self._get_control_mode()
        if mode == 'windows':
            return self._click_mouse_windows('right')
        else:
            return self._click_mouse_pyautogui('right')

    def middle_click(self):
        """中键点击"""
        mode = self._get_control_mode()
        if mode == 'windows':
            return self._click_mouse_windows('middle')
        else:
            return self._click_mouse_pyautogui('middle')

    def double_click(self):
        """双击"""
        success1 = self.left_click()
        time.sleep(0.1)  # 双击之间的延迟
        success2 = self.left_click()
        return success1 and success2

    def drag_start(self):
        """开始拖拽（按下左键）"""
        if HAS_WINDOWS_API and self._get_control_mode() == 'windows':
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            return True
        else:
            # pyautogui 没有单独的拖拽开始函数
            return self.left_click()

    def drag_end(self):
        """结束拖拽（释放左键）"""
        if HAS_WINDOWS_API and self._get_control_mode() == 'windows':
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True
        else:
            # pyautogui 会自动处理
            return True

    def update_scroll(self, hand_landmarks):
        """更新滚轮控制"""
        if not hand_landmarks:
            return False

        current_scroll_y = hand_landmarks.landmark[8].y

        if self.prev_scroll_y == 0:
            self.prev_scroll_y = current_scroll_y
            return False

        delta_y = self.prev_scroll_y - current_scroll_y

        if abs(delta_y) > self.SCROLL_THRESHOLD:
            scroll_amount = int(delta_y * self.SCROLL_SENSITIVITY)

            mode = self._get_control_mode()
            if mode == 'windows':
                success = self._scroll_mouse_windows(scroll_amount)
            else:
                success = self._scroll_mouse_pyautogui(scroll_amount)

            self.prev_scroll_y = current_scroll_y
            return success

        self.prev_scroll_y = current_scroll_y
        return False

    def horizontal_scroll(self, delta_x):
        """水平滚动"""
        if abs(delta_x) > self.SCROLL_THRESHOLD:
            scroll_amount = int(delta_x * self.SCROLL_SENSITIVITY)

            if HAS_WINDOWS_API and self._get_control_mode() == 'windows':
                return self.windows_api.mouse_hscroll(scroll_amount)
            else:
                # pyautogui 不支持水平滚动，可以模拟 shift+滚轮
                try:
                    pyautogui.keyDown('shift')
                    pyautogui.scroll(scroll_amount, _pause=False)
                    pyautogui.keyUp('shift')
                    return True
                except:
                    return False
        return False

    def calibrate(self, frame_width, frame_height):
        """校准控制区域"""
        # 根据摄像头分辨率调整控制区域
        min_dim = min(frame_width, frame_height)
        self.frameR = max(50, min_dim // 8)  # 自适应调整框大小

        print(f"校准完成: 控制区域缩减 = {self.frameR}")

    def reset(self):
        """重置控制器状态"""
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.prev_scroll_y = 0
        self.position_history = []
        self.last_click_time = 0
        print("鼠标控制器已重置")


# 创建全局实例（默认使用自动模式）
try:
    MOUSE_CONTROLLER = EnhancedMouseController(mode='auto')
    print("鼠标控制器初始化成功")
except Exception as e:
    print(f"鼠标控制器初始化失败: {e}")
    # 降级到基本版本
    MOUSE_CONTROLLER = None


# 兼容性接口（保持与原有代码相同的接口）
def move_mouse(hand_landmarks, img_shape):
    """移动鼠标"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.update_mouse_position(hand_landmarks, img_shape)
    return False


def left_click():
    """左键点击"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.left_click()
    return False


def right_click():
    """右键点击"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.right_click()
    return False


def scroll_mouse(hand_landmarks):
    """滚轮控制"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.update_scroll(hand_landmarks)
    return False


# 新增功能接口
def middle_click():
    """中键点击"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.middle_click()
    return False


def double_click():
    """双击"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.double_click()
    return False


def drag_start():
    """开始拖拽"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.drag_start()
    return False


def drag_end():
    """结束拖拽"""
    if MOUSE_CONTROLLER:
        return MOUSE_CONTROLLER.drag_end()
    return False


def calibrate_controller(frame_width, frame_height):
    """校准控制器"""
    if MOUSE_CONTROLLER:
        MOUSE_CONTROLLER.calibrate(frame_width, frame_height)
        return True
    return False


def reset_controller():
    """重置控制器"""
    if MOUSE_CONTROLLER:
        MOUSE_CONTROLLER.reset()
        return True
    return False


if __name__ == "__main__":
    # 测试代码
    print("=== 鼠标控制器测试 ===")
    print(f"控制模式: {MOUSE_CONTROLLER.mode if MOUSE_CONTROLLER else '未初始化'}")
    print(f"屏幕分辨率: {MOUSE_CONTROLLER.W_SCREEN}x{MOUSE_CONTROLLER.H_SCREEN}")
    print(f"Windows API 可用: {HAS_WINDOWS_API}")

    # 测试点击
    print("\n5秒后测试左键点击...")
    time.sleep(5)

    if MOUSE_CONTROLLER:
        MOUSE_CONTROLLER.left_click()
        print("左键点击测试完成")

        print("\n5秒后测试移动到屏幕中心...")
        time.sleep(5)

        # 测试移动到中心
        MOUSE_CONTROLLER.plocX = MOUSE_CONTROLLER.W_SCREEN // 2
        MOUSE_CONTROLLER.plocY = MOUSE_CONTROLLER.H_SCREEN // 2

        mode = MOUSE_CONTROLLER._get_control_mode()
        if mode == 'windows':
            MOUSE_CONTROLLER._move_mouse_windows(MOUSE_CONTROLLER.plocX, MOUSE_CONTROLLER.plocY)
        else:
            MOUSE_CONTROLLER._move_mouse_pyautogui(MOUSE_CONTROLLER.plocX, MOUSE_CONTROLLER.plocY)

        print("鼠标移动测试完成")

    print("\n=== 测试完成 ===")