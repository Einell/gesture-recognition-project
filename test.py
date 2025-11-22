import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import gesture_control as control

model_path = 'gesture_svm_model.pkl'#模型文件
try:
    svm_model = joblib.load(model_path)
    print(f"成功加载模型: {model_path}")
except FileNotFoundError:
    print(f"错误: 未找到模型文件 '{model_path}'，请先运行训练脚本生成模型")
    exit(1)

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,  # 视频模式
    max_num_hands=1,  # 最多检测1只手
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDRaw = mp.solutions.drawing_utils
handLmsStyle = mpDRaw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
handConnStyle = mpDRaw.DrawingSpec(color=(0, 0, 255), thickness=2)


# 特征提取 + 归一化
def extract_and_normalize_features(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    landmarks = np.array(landmarks)
    #归一化
    wrist_coord = landmarks[0]
    translated = landmarks - wrist_coord
    distances = np.linalg.norm(translated, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        normalized = translated / max_distance
    else:
        normalized = translated

    return normalized.flatten().reshape(1, -1) # 将特征展平为1维向量


# 连续识别次数阈值
CONSECUTIVE_THRESHOLD = 5
# 当前连续识别到的手势和次数
current_gesture = None
consecutive_count = 0

# 实时识别
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

pTime = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        continue

    # 镜像翻转画面
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # 存储识别结果
    recognition_results = []
    detected_gesture = None

    # 检测到手时处理
    if results.multi_hand_landmarks:
        # 遍历检测到的所有手部（因为 max_num_hands=1，这里只会循环一次）
        for hand_landmarks in results.multi_hand_landmarks:

            # 1. 提取特征
            features_to_predict = extract_and_normalize_features(hand_landmarks)

            # 2. 预测手势
            detected_gesture = svm_model.predict(features_to_predict)[0]

            # 显示当前预测的手势 (可选)
            cv2.putText(img, detected_gesture, (10, 110),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            # 连续识别计数逻辑
            if detected_gesture == current_gesture:
                consecutive_count += 1

                # 达到阈值，执行操作
                if consecutive_count >= CONSECUTIVE_THRESHOLD:
                    # 【关键修改位置】确保调用在 'for' 循环内部，这样 hand_landmarks 变量是可用的
                    # 将当前手势、视频帧和手部关键点数据传递给控制函数
                    control.execute_gesture_action(detected_gesture, cap, img, hand_landmarks=hand_landmarks)

                    # 重置计数
                    consecutive_count = 0
                    # 添加短暂延迟，防止连续触发
                    time.sleep(1.0)
            else:
                # 不同手势，重置计数
                current_gesture = detected_gesture
                consecutive_count = 1

            # 绘制手部关键点
            mpDRaw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS, handLmsStyle, handConnStyle)

    else:
        # 未检测到手势，重置
        current_gesture = None
        consecutive_count = 0

    # 显示信息
    # 计算FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    # 显示FPS
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (234, 255, 56), 3)

    # 显示识别结果（右上角）
    result_y = 70
    for result in recognition_results:
        cv2.putText(img, result, (img.shape[1] - 250, result_y),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        result_y += 50  # 换行偏移

    # 显示窗口
    cv2.imshow("Gesture Recognition", img)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
hands.close()