# 获取一个手势的特征与标签，并写入CSV文件
import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import os

# 手势标签
GESTURE_LABEL = 'left_snap'

# 输出的CSV文件路径
OUTPUT_CSV_PATH = f'q/{GESTURE_LABEL}.csv'

# 存储采集到的数据 (特征 + 标签)
collected_data = []


# 初始化MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4)
handConnStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)


# 识别特征点并归一化
def extract_and_normalize_features(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    landmarks = np.array(landmarks)

    # 以手腕为原点进行平移
    wrist_coord = landmarks[0]
    translated = landmarks - wrist_coord

    # 缩放到单位球内
    distances = np.linalg.norm(translated, axis=1)# 计算每个关键点距离原点的距离
    max_distance = np.max(distances)# 最大距离
    if max_distance > 0:
        # 归一化（除以最大距离）
        normalized = translated / max_distance
    else:
        # 异常情况，防止除0
        normalized = translated
    return normalized.flatten() # 将特征展平为1维向量



def main():
    global collected_data
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        exit(1)


    print("\n")
    print(f"{GESTURE_LABEL}")
    print("按下's'保存当前特征点与标签")
    print("按下'q'退出并保存所有数据")

    while True:
        ret, img = cap.read()
        if not ret:
            print("错误: 无法读取摄像头画面。")
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        current_features = None

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConnStyle)
                current_features = extract_and_normalize_features(handLms)


        cv2.putText(img, f'lable: {GESTURE_LABEL}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, f'number: {len(collected_data)}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        cv2.putText(img, "Press 's' to save, 'q' to quit", (img.shape[1] - 400, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow("Get Training Data", img)

        # 键盘事件处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n退出")
            break
        elif key == ord('s'):
            if current_features is not None:
                row_data = current_features.tolist() + [GESTURE_LABEL] # 将特征向量转换为列表
                collected_data.append(row_data) # 将当前特征向量添加到数据列表中
                print(f"成功保存样本 {len(collected_data)}: {GESTURE_LABEL}")
            else:
                print("警告: 未检测到有效手部关键点，无法保存样本。")

    cap.release()# 释放资源
    cv2.destroyAllWindows()
    hands.close()

    # 将数据写入CSV文件
    if collected_data:
        print(f"\n共采集到 {len(collected_data)}个样本，正在写入文件 '{OUTPUT_CSV_PATH}'")
        column_names = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + [
            'label']
        df = pd.DataFrame(collected_data, columns=column_names)

        # 如果文件已存在，则追加数据；否则创建新文件
        if os.path.exists(OUTPUT_CSV_PATH):
            df.to_csv(OUTPUT_CSV_PATH, mode='a', header=False, index=False)
            print(f"数据已成功追加到 {OUTPUT_CSV_PATH}")
        else:
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"数据已成功写入新文件 {OUTPUT_CSV_PATH}")
    else:
        print("\n没有采集到任何样本数据。")

    print("程序已退出。")

if __name__ == '__main__':
    main()