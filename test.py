import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import gesture_control as control
import threading

# ================= 配置区域 =================
PROBABILITY_THRESHOLD = 0.8  # 只有当概率大于此值时才执行操作
MODEL_PATH = 'gesture_svm_model.pkl'  # 模型文件路径

# 定义哪些手势是“实时”的（需要在主线程运行以保证平滑）
# 这些手势通常对应鼠标移动或音量调节
CONTINUOUS_GESTURES = {'right_mouse', 'right_mouse_roll', 'volume_control'}


# ================= 类定义 =================

class HandSmoother:
    """平滑滤波类，减少关键点抖动"""

    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks_proto):
        current_data = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks_proto.landmark])
        if self.prev_landmarks is None:
            self.prev_landmarks = current_data
            return current_landmarks_proto

        # 指数移动平均 (EMA)
        smoothed_data = self.alpha * current_data + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed_data

        for i, lm in enumerate(current_landmarks_proto.landmark):
            lm.x = smoothed_data[i, 0]
            lm.y = smoothed_data[i, 1]
            lm.z = smoothed_data[i, 2]
        return current_landmarks_proto


class AsyncExecutor:
    """
    多线程执行器
    用于处理包含 time.sleep 的耗时操作，防止阻塞视频主循环
    """

    def __init__(self):
        self.is_running = False  # 标记是否有后台任务正在运行
        self.lock = threading.Lock()

    def run(self, task_func, args=()):
        """
        尝试执行任务。
        如果当前没有任务在运行，则启动新线程。
        如果已有任务在运行（例如正在sleep），则忽略本次请求（防抖）。
        """
        # 检查锁状态，非阻塞式获取
        if not self.is_running:
            # 启动线程
            t = threading.Thread(target=self._worker, args=(task_func, args))
            t.daemon = True  # 设置为守护线程，主程序退出时自动结束
            t.start()

    def _worker(self, task_func, args):
        with self.lock:
            self.is_running = True

        try:
            # 执行具体的控制逻辑（包含sleep）
            task_func(*args)
        except Exception as e:
            print(f"后台任务执行出错: {e}")
        finally:
            # 任务结束，释放状态
            with self.lock:
                self.is_running = False


# ================= 主程序初始化 =================

# 1. 加载模型
try:
    svm_model = joblib.load(MODEL_PATH)
    print(f"成功加载模型: {MODEL_PATH}")
    print(f"支持的手势类别: {svm_model.classes_}")
except FileNotFoundError:
    print(f"错误: 未找到模型文件 '{MODEL_PATH}'")
    exit(1)

# 2. 初始化 MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDRaw = mp.solutions.drawing_utils
handLmsStyle = mpDRaw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
handConnStyle = mpDRaw.DrawingSpec(color=(0, 0, 255), thickness=2)

# 3. 初始化工具类
hand_smoother = HandSmoother(alpha=0.6)
async_executor = AsyncExecutor()


# 4. 特征提取函数
def extract_and_normalize_features(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    landmarks = np.array(landmarks)
    # 归一化
    wrist_coord = landmarks[0]
    translated = landmarks - wrist_coord
    distances = np.linalg.norm(translated, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        normalized = translated / max_distance
    else:
        normalized = translated
    return normalized.flatten().reshape(1, -1)


# 5. 状态变量
CONSECUTIVE_THRESHOLD = 5  # 连续帧阈值
current_gesture = None
consecutive_count = 0

# 6. 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

pTime = 0

print("系统启动完成。按 'q' 退出。")

# ================= 主循环 =================
while True:
    ret, img = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        continue

    # 镜像翻转 + 颜色转换
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 手部检测
    results = hands.process(imgRGB)

    detected_gesture = None
    gesture_probability = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. 平滑处理
            hand_landmarks = hand_smoother.smooth(hand_landmarks)

            # 2. 特征提取与预测
            features = extract_and_normalize_features(hand_landmarks)
            probabilities = svm_model.predict_proba(features)[0]

            max_proba_index = np.argmax(probabilities)
            gesture_probability = np.max(probabilities)
            predicted_label = svm_model.classes_[max_proba_index]

            # 3. 阈值过滤
            if gesture_probability >= PROBABILITY_THRESHOLD:
                detected_gesture = predicted_label

                # 显示当前识别结果
                cv2.putText(img, f'{detected_gesture} ({gesture_probability * 100:.1f}%)',
                            (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
            else:
                cv2.putText(img, f'Unsure ({gesture_probability * 100:.1f}%)',
                            (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 3)

            # 4. 执行逻辑分流
            if detected_gesture:
                # 逻辑 A: 实时连续手势 (鼠标移动/音量)
                # 这些操作不需要连续帧计数，也不需要多线程，因为它们必须每一帧都响应
                if detected_gesture in CONTINUOUS_GESTURES:
                    # 直接在主线程执行，确保流畅
                    control.execute_gesture_action(detected_gesture, cap, img, hand_landmarks)
                    # 重置计数器，避免干扰离散手势
                    consecutive_count = 0
                    current_gesture = detected_gesture

                # 逻辑 B: 离散/阻塞手势 (键盘/点击)
                # 这些操作需要防误触 (连续帧检测) 和 多线程 (防止卡顿)
                else:
                    if detected_gesture == current_gesture:
                        consecutive_count += 1
                        if consecutive_count >= CONSECUTIVE_THRESHOLD:
                            # 达到稳定性阈值，提交给后台线程
                            # 注意：execute_gesture_action 内部包含 time.sleep

                            # 使用 AsyncExecutor 运行
                            # 参数说明：task_func, args=(gesture, cap, img, landmarks)
                            # 注意：这里传递 landmarks 可能存在线程安全微小风险，但对于只读操作通常没事
                            async_executor.run(
                                control.execute_gesture_action,
                                args=(detected_gesture, None, None, hand_landmarks)
                            )

                            # 触发后，重置计数，防止立刻再次触发
                            # 也可以不重置，依靠 async_executor 的 is_running 锁来防抖
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
        # 未检测到手
        current_gesture = None
        consecutive_count = 0
        hand_smoother.prev_landmarks = None

    # 计算并显示 FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # 显示画面
    cv2.imshow("Gesture Recognition (Threaded)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()