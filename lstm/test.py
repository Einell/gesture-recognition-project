import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import gesture_control as control
import threading
import mouse_controller as mc  # 用于获取 frameR 的上下文信息

# ================= 配置区域 =================
PROBABILITY_THRESHOLD = 0.8  # 只有当概率大于此值时才执行操作
MODEL_PATH = 'gesture_svm_model.pkl'  # 模型文件路径
CONSECUTIVE_THRESHOLD = 3  # 离散手势连续检测次数，用于防抖

# 定义哪些手势是“实时”的（需要在主线程运行以保证平滑）
CONTINUOUS_GESTURES = {'right_mouse', 'right_mouse_roll', 'volume_control'}

# 窗口设置
WINDOW_NAME = "Gesture Recognition (Scaled & Resizable)"
INITIAL_WINDOW_SCALE = 0.7  # 初始窗口大小比例 (1280x720 * 0.7)


# ================= 类定义 (简化版，确保功能完整) =================

class HandSmoother:
    """平滑滤波类，减少关键点抖动"""

    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks_proto):
        """简单指数移动平均 (EMA) 过滤"""
        current_data = np.array([[lm.x, lm.y] for lm in current_landmarks_proto.landmark])
        if self.prev_landmarks is None:
            self.prev_landmarks = current_data
            return current_landmarks_proto

        smoothed_data = self.alpha * current_data + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed_data

        for i, lm in enumerate(current_landmarks_proto.landmark):
            lm.x = smoothed_data[i, 0]
            lm.y = smoothed_data[i, 1]

        return current_landmarks_proto


class AsyncExecutor:
    """用于执行离散手势操作的异步执行器，防止操作卡顿和重复触发"""

    def __init__(self):
        self._running = False
        self._lock = threading.Lock()

    def run(self, task_func, args=()):
        if not self._running:
            with self._lock:
                if not self._running:
                    self._running = True
                    thread = threading.Thread(target=self._execute, args=(task_func, args))
                    thread.start()

    def _execute(self, task_func, args):
        try:
            task_func(*args)
        finally:
            self._running = False


# ================= 模型加载和MediaPipe初始化 =================

# 加载模型
try:
    svm_model = joblib.load(MODEL_PATH)
    print(f"成功加载模型: {MODEL_PATH}")
except Exception as e:
    print(f"错误: 无法加载模型文件 {MODEL_PATH}。请确保文件存在且正确训练。")
    print(e)
    exit()

# MediaPipe Hand 初始化
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDRaw = mp.solutions.drawing_utils
handLmsStyle = mpDRaw.DrawingSpec(color=(0, 0, 255), thickness=3)
handConnStyle = mpDRaw.DrawingSpec(color=(0, 255, 0), thickness=5)


# ================= 特征提取函数 (与训练时保持一致) =================

def extract_and_normalize_features(hand_landmarks):
    """
    从MediaPipe关键点中提取特征。
    将关键点坐标展平为一维数组，并相对于手掌根部 (Landmark 0) 进行归一化。
    """
    if not hand_landmarks:
        return np.array([])

    # 获取手掌根部坐标 (Landmark 0)
    root_lm = hand_landmarks.landmark[0]
    root_x, root_y = root_lm.x, root_lm.y

    features = []
    for lm in hand_landmarks.landmark:
        # 归一化：所有点坐标 - 根部坐标
        features.extend([lm.x - root_x, lm.y - root_y])

    return np.array([features])  # 转换为 (1, 63) 的 numpy 数组


# ================= 主程序初始化 =================

pTime = 0  # Previous time for FPS
cTime = 0  # Current time for FPS
current_gesture = None
consecutive_count = 0

# 初始化类
hand_smoother = HandSmoother(alpha=0.6)
async_executor = AsyncExecutor()

cap = cv2.VideoCapture(0)
# 【摄像头分辨率设置】
wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

# 【Frame Reduction 值】: 用于定义映射到全屏的矩形区域
frameR = 200

# 【窗口初始化】：创建可调整大小的窗口
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# 设置初始尺寸 (基于 INITIAL_WINDOW_SCALE)
initial_w = int(wCam * INITIAL_WINDOW_SCALE)
initial_h = int(hCam * INITIAL_WINDOW_SCALE)
cv2.resizeWindow(WINDOW_NAME, initial_w, initial_h)

print("系统启动完成。按 'q' 退出。")

# ================= 主循环 =================
while True:
    ret, img = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        time.sleep(0.1)
        continue

    # 【镜像翻转】：让用户感觉更自然，并修正鼠标方向
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ================== 绘制操作区域矩形框 ==================
    # 矩形框的坐标仍然基于原始分辨率 wCam, hCam 计算
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)
    # =======================================================

    # 手部检测
    results = hands.process(imgRGB)

    detected_gesture = None
    gesture_probability = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # 1. (可选) 平滑处理 (如果不需要，可注释掉)
            # hand_landmarks = hand_smoother.smooth(hand_landmarks)

            # 2. 特征提取与预测
            features = extract_and_normalize_features(hand_landmarks)
            probabilities = svm_model.predict_proba(features)[0]

            max_proba_index = np.argmax(probabilities)
            gesture_probability = np.max(probabilities)
            predicted_label = svm_model.classes_[max_proba_index]

            # 3. 阈值过滤与显示
            if gesture_probability >= PROBABILITY_THRESHOLD:
                detected_gesture = predicted_label

                # 颜色区分：连续操作显示绿色，否则紫色
                text_color = (0, 255, 0) if detected_gesture in CONTINUOUS_GESTURES else (255, 0, 255)

                cv2.putText(img, f'{detected_gesture} ({gesture_probability * 100:.1f}%)',
                            (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 3)
            else:
                cv2.putText(img, f'Unsure ({gesture_probability * 100:.1f}%)',
                            (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 3)

            # 4. 执行逻辑分流
            if detected_gesture:
                if detected_gesture in CONTINUOUS_GESTURES:
                    # 连续手势（如鼠标移动、滚动）：在主线程实时执行
                    # 传入 img_shape 和 frameR
                    control.execute_gesture_action(detected_gesture, cap, img, hand_landmarks, frameR=frameR)
                    consecutive_count = 0
                    current_gesture = detected_gesture

                else:
                    # 离散手势（如点击、键盘操作）：使用多线程防抖
                    if detected_gesture == current_gesture:
                        consecutive_count += 1
                        if consecutive_count >= CONSECUTIVE_THRESHOLD:
                            # 异步执行，防止操作中的 time.sleep 阻塞主循环
                            async_executor.run(
                                control.execute_gesture_action,
                                args=(detected_gesture, None, None, hand_landmarks, None)  # 离散手势不需要 frameR
                            )
                            consecutive_count = 0
                    else:
                        current_gesture = detected_gesture
                        consecutive_count = 1
            else:
                current_gesture = None
                consecutive_count = 0

            # 绘制骨架
            mpDRaw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS, handLmsStyle, handConnStyle)

    else:
        # 未检测到手时，重置状态
        current_gesture = None
        consecutive_count = 0
        # hand_smoother.prev_landmarks = None # 如果使用了平滑，这里也应该重置

    # 计算并显示 FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # 显示画面
    # 由于窗口已设置为 cv2.WINDOW_NORMAL，cv2.imshow 会自动缩放原始图像
    cv2.imshow(WINDOW_NAME, img)

    # 退出机制
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= 清理资源 =================
cap.release()
cv2.destroyAllWindows()
hands.close()