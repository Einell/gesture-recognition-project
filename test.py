import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import gesture_control as control
import threading
import mouse_controller as mc
from tensorflow.keras.models import load_model

# ================= 配置区域 =================
SVM_PROB_THRESHOLD = 0.8
LSTM_PROB_THRESHOLD = 0.9
SVM_MODEL_PATH = 'gesture_svm_model.pkl'
LSTM_MODEL_PATH = 'gesture_lstm_model.h5'
LSTM_CLASSES_PATH = 'lstm_classes.npy'

# 动态判定参数
MOVEMENT_THRESHOLD = 0.015  # 移动阈值 (标准差)
LSTM_SEQ_LEN = 30

CONTINUOUS_GESTURES = {'right_mouse', 'right_mouse_roll', 'volume_control'}
WINDOW_SCALE = 0.7


# ================= 辅助类 =================

class HandSmoother:
    """平滑滤波类"""

    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = None

    def smooth(self, current_landmarks_proto):
        current_data = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks_proto.landmark])
        if self.prev_landmarks is None:
            self.prev_landmarks = current_data
            return current_landmarks_proto
        smoothed_data = self.alpha * current_data + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed_data
        for i, lm in enumerate(current_landmarks_proto.landmark):
            lm.x = smoothed_data[i, 0]
            lm.y = smoothed_data[i, 1]
            lm.z = smoothed_data[i, 2]
        return current_landmarks_proto


class AsyncExecutor:
    """异步执行器"""

    def __init__(self):
        self.is_running = False
        self.lock = threading.Lock()

    def run(self, task_func, args=()):
        if not self.is_running:
            t = threading.Thread(target=self._worker, args=(task_func, args))
            t.daemon = True
            t.start()

    def _worker(self, task_func, args):
        with self.lock:
            self.is_running = True
        try:
            task_func(*args)
        except Exception as e:
            print(f"Task Error: {e}")
        finally:
            with self.lock:
                self.is_running = False


class MotionDetector:
    """检测手部是否在移动"""

    def __init__(self, history_len=5):
        self.history_len = history_len
        self.positions = []  # 存储最近几帧的手腕坐标 [(x,y), ...]

    def update(self, hand_landmarks):
        # 使用手腕 (Landmark 0)
        wrist = hand_landmarks.landmark[0]
        self.positions.append((wrist.x, wrist.y))
        if len(self.positions) > self.history_len:
            self.positions.pop(0)

    def is_moving(self, threshold=MOVEMENT_THRESHOLD):
        if len(self.positions) < self.history_len:
            return False  # 数据不足，默认不动

        # 计算标准差
        data = np.array(self.positions)
        std_dev = np.std(data, axis=0)  # [std_x, std_y]
        movement = np.mean(std_dev)

        return movement > threshold


# ================= 特征提取函数 =================

def extract_svm_features(hand_landmarks):
    """SVM: 单手，归一化"""
    landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    landmarks = np.array(landmarks)
    wrist = landmarks[0]
    normalized = landmarks - wrist
    dist = np.max(np.linalg.norm(normalized, axis=1))
    if dist > 0: normalized /= dist
    return normalized.flatten().reshape(1, -1)


def extract_lstm_features(results):
    """LSTM: 双手，定长126，归一化"""
    feature_vector = np.zeros(126)
    if not results.multi_hand_landmarks: return feature_vector

    for idx, lms in enumerate(results.multi_hand_landmarks):
        lbl = results.multi_handedness[idx].classification[0].label
        # 提取并归一化
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark])
        arr = arr - arr[0]  # 减去手腕
        mx = np.max(np.linalg.norm(arr, axis=1))
        if mx > 0: arr /= mx

        flat = arr.flatten()
        if lbl == 'Left':
            feature_vector[0:63] = flat
        else:
            feature_vector[63:126] = flat
    return feature_vector


# ================= 主程序 =================

# 1. 加载模型
print("正在加载模型...")
try:
    svm_model = joblib.load(SVM_MODEL_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)
    lstm_classes = np.load(LSTM_CLASSES_PATH)
    print("SVM & LSTM 模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 2. 初始化
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
hand_smoother = HandSmoother()
async_executor = AsyncExecutor()
motion_detector = MotionDetector()

# 状态变量
lstm_buffer = []  # 存储 LSTM 特征序列
current_static_gesture = None
static_consecutive_count = 0
STATIC_THRESHOLD = 5

# 摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 800)

while True:
    ret, img = cap.read()
    if not ret: break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    status_text = "No Hand"
    color = (100, 100, 100)

    if results.multi_hand_landmarks:
        # 默认取第一只手做 SVM 和 运动检测
        primary_hand = results.multi_hand_landmarks[0]
        primary_hand = hand_smoother.smooth(primary_hand)

        # 更新运动检测器
        motion_detector.update(primary_hand)
        moving = motion_detector.is_moving()

        # === 决策流水线 ===

        if moving:
            # --- 动态分支 (LSTM) ---
            status_text = "Dynamic Mode (Moving)"
            color = (0, 165, 255)  # Orange

            # 1. 收集特征
            feat = extract_lstm_features(results)
            lstm_buffer.append(feat)

            # 2. 保持 buffer 长度为 30
            if len(lstm_buffer) > LSTM_SEQ_LEN:
                lstm_buffer.pop(0)

            # 3. 如果填满了，进行预测
            if len(lstm_buffer) == LSTM_SEQ_LEN:
                input_seq = np.expand_dims(np.array(lstm_buffer), axis=0)  # (1, 30, 126)
                prediction = lstm_model.predict(input_seq, verbose=0)[0]
                idx = np.argmax(prediction)
                conf = prediction[idx]

                if conf > LSTM_PROB_THRESHOLD:
                    gesture_name = lstm_classes[idx]
                    print(f"LSTM 检测到: {gesture_name} ({conf:.2f})")
                    cv2.putText(img, f"ACTION: {gesture_name}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                                3)

                    # 异步执行命令
                    async_executor.run(control.execute_gesture_action, args=(gesture_name, None, None, None))

                    # 触发后清空 buffer 防止重复触发
                    lstm_buffer = []

                    # 移动时重置静态计数
            static_consecutive_count = 0
            current_static_gesture = None

        else:
            # --- 静态分支 (SVM) ---
            status_text = "Static Mode (Stable)"
            color = (0, 255, 0)  # Green

            # 清空 LSTM buffer，避免把静态姿势混入动态序列
            lstm_buffer = []

            # 1. SVM 预测
            svm_feat = extract_svm_features(primary_hand)
            probs = svm_model.predict_proba(svm_feat)[0]
            max_idx = np.argmax(probs)
            svm_conf = probs[max_idx]
            svm_label = svm_model.classes_[max_idx]

            if svm_conf > SVM_PROB_THRESHOLD:
                # 逻辑 A: 实时手势 (鼠标/音量) - 直接执行
                if svm_label in CONTINUOUS_GESTURES:
                    control.execute_gesture_action(svm_label, cap, img, primary_hand)
                    static_consecutive_count = 0
                    cv2.putText(img, f"{svm_label}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # 逻辑 B: 触发式手势 - 需稳定 N 帧
                else:
                    if svm_label == current_static_gesture:
                        static_consecutive_count += 1
                        if static_consecutive_count >= STATIC_THRESHOLD:
                            cv2.putText(img, f"ACTION: {svm_label}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            async_executor.run(control.execute_gesture_action,
                                               args=(svm_label, None, None, primary_hand))
                            static_consecutive_count = 0  # 重置
                    else:
                        current_static_gesture = svm_label
                        static_consecutive_count = 1
            else:
                status_text += " (Low Conf)"

        # 绘制
        for h in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, h, mpHands.HAND_CONNECTIONS)

    cv2.putText(img, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Dual-Mode Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()