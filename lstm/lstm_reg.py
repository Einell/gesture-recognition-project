# 这是一个基于lstm的动态手势识别程序。
# 利用mediapipe识别手部骨架数据传给lstm模型进行手势识别，并执行相应的手势控制。
# 注意开启摄像头权限。

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import threading
import gesture_control

# 参数
MODEL_PATH = 'gesture_lstm_model.keras'
CLASSES_PATH = 'lstm_classes.npy'
SEQUENCE_LENGTH = 20 # 序列长度
THRESHOLD = 0.8 # 置信阈值
ACTION_COOLDOWN = 1.0 # 动作冷却时间
SKIP_FRAMES = 1  # 跳帧数
# MediaPipe初始化
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
handConnStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)

# 特征提取
def extract_keypoints(results):
    feature_vector = np.zeros(126)
    if not results.multi_hand_landmarks:
        return feature_vector
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        handedness = results.multi_handedness[idx].classification[0].label
        # 提取坐标
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # # 以手腕为中心进行坐标平移
        wrist = lm_array[0]
        lm_array = lm_array - wrist

        max_dist = np.max(np.linalg.norm(lm_array, axis=1))
        if max_dist > 0:
            lm_array /= max_dist

        flat_features = lm_array.flatten()
        # 左右手写入
        if handedness == 'Left':
            feature_vector[0:63] = flat_features
        else:
            feature_vector[63:126] = flat_features

    return feature_vector

# 动作执行
def run_action_in_thread(gesture, cap_ref, img_ref, landmarks_ref):
    try:
        gesture_control.execute_gesture_action(gesture, cap_ref, img_ref, landmarks_ref)
    except Exception as e:
        print(f"Action Error: {e}")


# 主程序
def main():
    # 加载模型
    try:
        model = load_model(MODEL_PATH)
        classes = np.load(CLASSES_PATH, allow_pickle=True)
        print(f"模型加载成功: {classes}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    cap = cv2.VideoCapture(0)

    sequence = []
    last_action_time = 0
    current_action = "Waiting..."
    confidence_score = 0.0
    frame_count = 0  # 用于跳帧计数

    with mpHands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
    ) as hands:

        print("启动成功！按 'q' 退出程序。")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 镜像翻转
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # 图像预处理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #跳帧检测
            if frame_count % (SKIP_FRAMES + 1) == 0:
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # 特征提取与预测
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]

                if len(sequence) == SEQUENCE_LENGTH:
                    if results.multi_hand_landmarks:
                        input_data = np.expand_dims(sequence, axis=0)
                        res = model.predict(input_data, verbose=0)[0]
                        best_idx = np.argmax(res)
                        confidence_score = res[best_idx]
                        predicted_gesture = classes[best_idx]

                        # 执行逻辑
                        if confidence_score > THRESHOLD:
                            # 过滤静止与背景
                            if predicted_gesture == 'background' or predicted_gesture == 'static':
                                current_action = "Static/Background"
                                pass

                            # 识别出手势
                            elif (time.time() - last_action_time) > ACTION_COOLDOWN:
                                current_action = predicted_gesture

                                first_hand = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

                                action_thread = threading.Thread(
                                    target=run_action_in_thread,
                                    args=(predicted_gesture, cap, frame, first_hand)
                                )
                                action_thread.start()

                                last_action_time = time.time()
                    else:
                        current_action = "No Hand"
                        confidence_score = 0.0

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if 'results' in locals() and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)

            cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
            color = (0, 255, 0) if (time.time() - last_action_time) > ACTION_COOLDOWN else (0, 0, 255)
            cv2.putText(image, f"{current_action} ({confidence_score:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow('Gesture Control', image)

            # 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("正在退出...")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()