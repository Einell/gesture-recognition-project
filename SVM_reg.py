# 这是一个基于SVM的静态手势识别程序。
# 利用mediapipe识别手部骨架数据传给SVM模型进行手势识别，并执行相应的手势控制。
# 注意开启摄像头权限。
import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import gesture_control as control # 手势控制
import threading
import mouse_controller as mc # 鼠标控制

PROBABILITY_THRESHOLD = 0.8  # 置信概率阈值
MODEL_PATH = 'gesture_svm_model.pkl' # 模型文件路径
CONTINUOUS_GESTURES = {'right_mouse', 'right_mouse_roll', 'volume_control'} # 实时手势
CONSECUTIVE_THRESHOLD = 5  # 连续帧阈值
current_gesture = None # 当前手势
consecutive_count = 0 # 连续帧计数器

class HandSmoother:
    # 平滑滤波类，减少关键点抖动

    def __init__(self, alpha=0.6):
        self.alpha = alpha # 平滑系数
        self.prev_landmarks = None # 上一帧数据

    def smooth(self, current_landmarks_proto):
        current_data = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks_proto.landmark])
        if self.prev_landmarks is None:
            self.prev_landmarks = current_data
            return current_landmarks_proto

        # 指数移动平均公式
        smoothed_data = self.alpha * current_data + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed_data

        for i, lm in enumerate(current_landmarks_proto.landmark):
            lm.x = smoothed_data[i, 0]
            lm.y = smoothed_data[i, 1]
            lm.z = smoothed_data[i, 2]
        return current_landmarks_proto

class AsyncExecutor:
    # 多线程执行器

    def __init__(self):
        self.is_running = False  # 是否有后台任务正在运行
        self.lock = threading.Lock() # 线程锁

    def run(self, task_func, args=()):
        #尝试执行任务，如没有任务运行，则启动新线程。如有任务在运行，则忽略本次请求。
        if not self.is_running:
            # 启动线程
            t = threading.Thread(target=self._worker, args=(task_func, args))
            t.daemon = True  # 设置为守护线程，主程序退出时自动结束
            t.start()

    def _worker(self, task_func, args):
        with self.lock:
            self.is_running = True

        try:
            # 执行具体的控制逻辑
            task_func(*args)
        except Exception as e:
            print(f"后台任务执行出错: {e}")
        finally:
            # 任务结束，释放状态
            with self.lock:
                self.is_running = False

# 加载模型
try:
    svm_model = joblib.load(MODEL_PATH)
    print(f"成功加载模型: {MODEL_PATH}")
    print(f"支持的手势类别: {svm_model.classes_}")
except FileNotFoundError:
    print(f"错误: 未找到模型文件 '{MODEL_PATH}'")
    exit(1)

# 初始化MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False, # 非静态图片
    max_num_hands=1, # 最多检测1只手
    min_detection_confidence=0.7, # 检测模型置信度
    min_tracking_confidence=0.7 # 跟踪模型置信度
)
mpDRaw = mp.solutions.drawing_utils
handLmsStyle = mpDRaw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3) # 手指关键点样式
handConnStyle = mpDRaw.DrawingSpec(color=(0, 255, 0), thickness=3) # 手指连线样式

hand_smoother = HandSmoother(alpha=0.6) # 创建平滑滤波器
async_executor = AsyncExecutor() # 创建多线程执行器


# 特征提取
def extract_and_normalize_features(hand_landmarks):
    # 提取21个特征点并转换成数组
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    landmarks = np.array(landmarks)
    # 以手腕为中心进行坐标平移
    wrist_coord = landmarks[0]
    translated = landmarks - wrist_coord
    #根据最大距离进行归一化缩放
    distances = np.linalg.norm(translated, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        normalized = translated / max_distance
    else:
        normalized = translated
    return normalized.flatten().reshape(1, -1) # 将特征展平为1维向量

# 摄像头
cap = cv2.VideoCapture(0)
# 摄像头分辨率
wCam, hCam = 1280, 800
cap.set(3, wCam)
cap.set(4, hCam)
frameR = mc.MOUSE_CONTROLLER.frameR # 鼠标控制区间，frameR为边距
pTime = 0 # 用于求fps

print("系统启动完成，按'q'退出")

# 主循环
while True:
    ret, img = cap.read() # 读取摄像头数据
    if not ret:
        print("无法读取摄像头数据")
        continue

    img = cv2.flip(img, 1) # 镜像翻转，更符合观感
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 颜色转换
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)# 鼠标控制映射区间
    results = hands.process(imgRGB) # 手部检测

    detected_gesture = None # 当前手势
    gesture_probability = 0.0 # 当前手势的概率
    # 手部检测结果
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks = hand_smoother.smooth(hand_landmarks) # 平滑处理

            features = extract_and_normalize_features(hand_landmarks) # 特征提取
            probabilities = svm_model.predict_proba(features)[0] # SVM预测

            max_proba_index = np.argmax(probabilities) # 获取概率最大手势
            gesture_probability = np.max(probabilities) # 获取概率最大手势的概率
            predicted_label = svm_model.classes_[max_proba_index] # 获取概率最大手势的标签
            # 阈值过滤，如果大于阈值则更新当前手势
            if gesture_probability >= PROBABILITY_THRESHOLD:
                detected_gesture = predicted_label

                # 显示当前识别结果，颜色区分实时还是离散手势
                text_color = (0, 255, 0) if detected_gesture in CONTINUOUS_GESTURES else (255, 0, 255)
                cv2.putText(img, f'{detected_gesture} ({gesture_probability * 100:.1f}%)',
                            (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 3)
            else:
                cv2.putText(img, f'Unsure ({gesture_probability * 100:.1f}%)',
                            (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 3)
            # 连续/离散手势分流
            if detected_gesture:
                # 连续手势，每一帧响应
                if detected_gesture in CONTINUOUS_GESTURES:
                    control.execute_gesture_action(detected_gesture, cap, img, hand_landmarks)
                    consecutive_count = 0 # 重置计数器
                    current_gesture = detected_gesture

                # 离散手势，
                else:
                    if detected_gesture == current_gesture:
                        consecutive_count += 1
                        if consecutive_count >= CONSECUTIVE_THRESHOLD:
                            # 达到连续帧阈值，提交线程
                            async_executor.run(
                                control.execute_gesture_action,
                                args=(detected_gesture, None, None, hand_landmarks)
                            )
                            consecutive_count = 0 # 重置计数器
                    else:
                        current_gesture = detected_gesture
                        consecutive_count = 1
            else:
                current_gesture = None
                consecutive_count = 0

            # 绘制手部骨架
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
    cv2.imshow("静态SVM识别", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
hands.close()