import os
import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# --- 1. 配置路径和参数 ---
# 源数据集
source_root = 'Gesture'
# CSV文件名
output_csv = 'gesture_feature.csv'

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5)


# --- 2. 核心功能函数 ---

def extract_hand_landmarks(image_path):
    """
    使用MediaPipe从图片中提取手部关键点。
    返回一个 (21, 3) 的numpy数组，包含每个关键点的 (x, y, z) 坐标。
    如果未检测到手，则返回None。
    """
    # 读取图片，OpenCV默认读取为BGR格式
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告: 无法读取图片 {image_path}")
        return None

    # MediaPipe需要RGB格式的图片
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 处理图片并获取结果
    results = hands.process(image_rgb)

    # 检查是否检测到了手
    if not results.multi_hand_landmarks:
        # print(f"警告: 在图片 {image_path} 中未检测到手")
        return None

    # 我们只取第一只检测到的手
    hand_landmarks = results.multi_hand_landmarks[0]

    # 将关键点转换为numpy数组
    landmarks = []
    for landmark in hand_landmarks.landmark:
        # landmark.x, landmark.y 是相对于图片宽高的归一化坐标
        # landmark.z 是相对于手腕的深度，手腕处z=0
        landmarks.append([landmark.x, landmark.y, landmark.z])

    return np.array(landmarks)


def normalize_landmarks(landmarks):
    """
    对关键点进行归一化处理：
    1. 以手腕（第一个点）为原点进行平移。
    2. 计算所有点到原点的最大距离，并用其对所有坐标进行缩放，使其落在单位球内。
    """
    if landmarks is None:
        return None

    # 1. 平移：所有点减去手腕点的坐标
    wrist_coord = landmarks[0]
    normalized = landmarks - wrist_coord

    # 2. 缩放：计算所有点的欧氏距离
    distances = np.linalg.norm(normalized, axis=1)
    max_distance = np.max(distances)

    # 避免除以零
    if max_distance > 0:
        normalized = normalized / max_distance

    return normalized


# --- 3. 主处理流程 ---

def main():
    print(f"开始从 '{source_root}' 提取特征...")
    print("特征将经过平移和单位球缩放归一化处理。")

    all_data = []

    # --- 修改开始 ---
    # 直接遍历 Gesture 文件夹下的所有手势文件夹
    for gesture_name in os.listdir(source_root):
        gesture_path = os.path.join(source_root, gesture_name)

        # 确保是目录
        if not os.path.isdir(gesture_path):
            continue

        print(f"\n正在处理手势: {gesture_name}...")

        # 遍历该手势文件夹下的所有图片文件
        for image_filename in os.listdir(gesture_path):
            image_path = os.path.join(gesture_path, image_filename)

            # 提取关键点
            landmarks = extract_hand_landmarks(image_path)

            if landmarks is not None:
                # 进行平移和缩放归一化
                normalized_landmarks = normalize_landmarks(landmarks)

                if normalized_landmarks is not None:
                    feature_vector = normalized_landmarks.flatten()
                    # 标签就是手势文件夹的名称
                    row_data = feature_vector.tolist() + [gesture_name]
                    all_data.append(row_data)
    # --- 修改结束 ---

    hands.close()

    if not all_data:
        print("\n错误: 没有成功提取到任何特征数据，无法生成CSV文件。")
        print("可能的原因包括：文件夹中没有图片，或所有图片都未检测到手部关键点。")
        return

    print(f"\n总共处理了 {len(all_data)} 张有效图片。")
    print(f"正在将数据写入 '{output_csv}'...")

    column_names = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + [
        'label']

    df = pd.DataFrame(all_data, columns=column_names)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print("特征提取、归一化和CSV文件生成完成！")
if __name__ == '__main__':
    main()