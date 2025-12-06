# 获取一个手势的特征与标签，并写入CSV文件
# 设置手势标签与保存位置，运行程序，按下's'键保存，'q'键退出
import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import os

# 配置
# 存储采集到的数据
collected_data = []

# 初始化MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
handConnStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)


# 识别特征点并归一化
def extract_and_normalize_features(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    landmarks = np.array(landmarks)
    # 以手腕为中心进行坐标平移
    wrist_coord = landmarks[0]
    translated = landmarks - wrist_coord
    # 根据最大距离进行归一化缩放
    distances = np.linalg.norm(translated, axis=1)
    max_distance = np.max(distances)
    if max_distance > 0:
        normalized = translated / max_distance
    else:
        normalized = translated
    return normalized.flatten() # 将特征展平为1维向量
# 主函数
def main():
    # 获取用户输入
    GESTURE_LABEL = input("请输入手势标签: ").strip()
    if not GESTURE_LABEL:
        print("手势标签不能为空！")
        return

    OUTPUT_CSV_PATH = f'../../data/SVM_data/{GESTURE_LABEL}.csv'

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

        cv2.imshow("get img data", img)

        # 键盘事件处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n退出")
            break
        elif key == ord('s'):
            if current_features is not None:
                row_data = current_features.tolist() + [GESTURE_LABEL] # 将特征向量+标签转换为列表
                collected_data.append(row_data) # 添加到数据列表中
                print(f"成功保存样本 {len(collected_data)}: {GESTURE_LABEL}")
            else:
                print("未检测到有效手部关键点，无法保存样本!")
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # 数据写入
    if collected_data:
        print(f"\n共采集到 {len(collected_data)}个样本，正在写入文件 '{OUTPUT_CSV_PATH}'")
        column_names = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + [
            'label']
        df = pd.DataFrame(collected_data, columns=column_names)

        # 如果文件已存在，则追加数据；否则创建新文件
        if os.path.exists(OUTPUT_CSV_PATH):
            df.to_csv(OUTPUT_CSV_PATH, mode='a', header=False, index=False)
            print(f"已追加 {OUTPUT_CSV_PATH}")
        else:
            df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"已创建并写入 {OUTPUT_CSV_PATH}")
    else:
        print("\n未采集到数据")

    print("程序已退出")

if __name__ == '__main__':
    main()