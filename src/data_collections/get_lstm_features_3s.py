# 获取LSTM特征
# 输入手势标签与存放路径，运行程序，按下'r'开始录制，按'q'退出
import re

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# 参数
# 特征总长度 = 20帧 * (2只手 * 21点 * 3坐标) = 2520 ， 单帧特征长度 = 126
SEQUENCE_LENGTH = 20
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3

# MediaPipe初始化
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
handConnStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)

# 特征提取
def extract_dual_hand_features(results):
    feature_vector = np.zeros(126)  # 初始化全零向量

    if not results.multi_hand_landmarks:
        return feature_vector, False
    hands_found = 0

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        hands_found += 1
        handedness = results.multi_handedness[idx].classification[0].label

        # 提取坐标
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        landmarks = np.array(landmarks)

        # 以手腕为中心进行坐标平移
        wrist_coord = landmarks[0]
        translated = landmarks - wrist_coord

        distances = np.linalg.norm(translated, axis=1)
        max_distance = np.max(distances)
        if max_distance > 0:
            normalized = translated / max_distance
        else:
            normalized = translated
        normalized = normalized.flatten().reshape(1, -1)  # 将特征展平为1维向量

        # 左右手写入
        if handedness == 'Left':
            feature_vector[0:63] = normalized
        else:
            feature_vector[63:126] = normalized

    return feature_vector, True

# 主程序
def main():
    # 获取用户输入
    GESTURE_LABEL = input("请输入手势标签: ").strip()
    if not GESTURE_LABEL:
        print("手势标签不能为空！")
        return
    # 验证标签名称（只允许字母、数字、下划线和连字符）
    if not re.match(r'^[a-zA-Z0-9_-]+$', GESTURE_LABEL):
        print("手势标签只能包含字母、数字、下划线和连字符！")
        return

    # 确保目录存在
    output_dir = '../../data/LSTM_data'
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_CSV_PATH = f'../../data/LSTM_data/{GESTURE_LABEL}.csv'


    cap = cv2.VideoCapture(0)
    sequence_buffer = []

    is_recording = False
    is_counting_down = False
    countdown_start_time = 0
    COUNTDOWN_DURATION = 1  # 倒计时秒数

    save_count = 0
    print(f"准备录制动态手势: {GESTURE_LABEL}")
    print(f"按 'r' 开始录制 (倒计时秒{COUNTDOWN_DURATION}后开始)")
    print("按 'q' 退出")

    while True:
        ret, img = cap.read()
        if not ret: break
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        # 绘制
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS,handLmsStyle, handConnStyle)

        # 倒计时
        if is_counting_down:
            elapsed_time = time.time() - countdown_start_time
            # 计算剩余的倒计时
            countdown_val = max(0, int(COUNTDOWN_DURATION - elapsed_time) + 1)

            if countdown_val > 0:
                # 倒计时进行中
                cv2.putText(img, f"START IN: {countdown_val}",
                            (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 165, 255), 5)  # 橙色
            else:
                # 倒计时结束，开始录制
                is_counting_down = False
                is_recording = True
                sequence_buffer = []
                print("倒计时结束，开始录制...")

        # 录制
        elif is_recording:
            # 获取特征和是否检测到手的标志
            feat, hand_detected = extract_dual_hand_features(results)
            if not hand_detected:
                print("丢失手部追踪，录制中止！")
                sequence_buffer = []
                is_recording = False
                continue
            sequence_buffer.append(feat)

            cv2.putText(img, f"REC: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                is_recording = False
                save_count += 1

                # 保存数据
                flat_sequence = np.array(sequence_buffer).flatten().tolist()
                flat_sequence.append(GESTURE_LABEL)

                df = pd.DataFrame([flat_sequence])

                # 添加表头
                header = not os.path.exists(OUTPUT_CSV_PATH)
                df.to_csv(OUTPUT_CSV_PATH, mode='a', header=header, index=False)

                print(f"已保存!")
                sequence_buffer = []

                # 自动重启倒计时
                # 注释则关闭连拍模式
                is_counting_down = True
                countdown_start_time = time.time()

        else:
            # 待机状态
            cv2.putText(img, f"Press 'r' to record '{GESTURE_LABEL}'",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 绿色

        cv2.imshow("Get LSTM Features", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not is_recording and not is_counting_down:
                is_counting_down = True
                countdown_start_time = time.time()
                print(f"按下 'r'，开始 {COUNTDOWN_DURATION} 秒倒计时...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()