import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# ================= 配置 =================
GESTURE_LABEL = 'zoom_in'  # 修改这里来录制不同的手势
OUTPUT_CSV_PATH = f'lstm/{GESTURE_LABEL}.csv'
SEQUENCE_LENGTH = 20  # 序列长度 (帧数)
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 2
# 特征总长度 = 25帧 * (2只手 * 21点 * 3坐标) = 3780 (写入CSV的一行)
# 但我们在提取时，单帧特征长度 = 126

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 关键：允许两只手
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils


def extract_dual_hand_features(results):
    """
    提取双手的特征，始终保持 左手在前(0-62), 右手在后(63-125) 的顺序。
    如果某只手不存在，填充0。
    """
    # 初始化全0向量 (126,)
    feature_vector = np.zeros(2 * LANDMARKS_PER_HAND * COORDS_PER_LANDMARK)

    if not results.multi_hand_landmarks:
        return feature_vector

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # 获取左右手标签
        # 注意: MediaPipe的 multi_handedness 可能会与左右手实际位置混淆，
        # 在实际应用中，通常会根据手的 x 坐标来更可靠地判断左右。
        # 这里为了与原代码逻辑一致，仍使用 label。
        handedness = results.multi_handedness[idx].classification[0].label

        # 提取坐标并归一化 (相对于手腕)
        lm_array = []
        for lm in hand_landmarks.landmark:
            lm_array.append([lm.x, lm.y])
        lm_array = np.array(lm_array)

        # 以手腕为中心
        wrist = lm_array[0]
        lm_array = lm_array - wrist

        # 归一化
        max_dist = np.max(np.linalg.norm(lm_array, axis=1))
        if max_dist > 0:
            lm_array /= max_dist

        flat_features = lm_array.flatten()  # (63,)

        # 根据左右手填入对应位置
        if handedness == 'Left':
            feature_vector[0:42] = flat_features
        else:
            feature_vector[42:84] = flat_features

    return feature_vector


def main():
    cap = cv2.VideoCapture(0)
    sequence_buffer = []

    # --- 新增/修改的状态变量 ---
    is_recording = False
    is_counting_down = False
    countdown_start_time = 0
    COUNTDOWN_DURATION = 1  # 倒计时秒数
    # --- -------------------- ---

    print(f"准备录制动态手势: {GESTURE_LABEL}")
    print("按 'r' 开始录制 (倒计时3秒后开始)")
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
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

        # --- 倒计时逻辑 ---
        if is_counting_down:
            elapsed_time = time.time() - countdown_start_time
            # 计算剩余的倒计时整数秒
            countdown_val = max(0, int(COUNTDOWN_DURATION - elapsed_time) + 1)

            if countdown_val > 0:
                # 倒计时进行中，显示数字
                cv2.putText(img, f"START IN: {countdown_val}",
                            (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 165, 255), 5)  # 橙色
            else:
                # 倒计时结束，开始录制
                is_counting_down = False
                is_recording = True
                sequence_buffer = []
                print("倒计时结束，开始录制...")
        # --- -------------------- ---

        # 录制逻辑
        elif is_recording:
            feat = extract_dual_hand_features(results)
            sequence_buffer.append(feat)

            cv2.putText(img, f"RECORDING... {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 红色

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                is_recording = False
                print("录制完成，正在保存...")

                # 保存数据
                # 将 (30, 126) 展平为 (3780,) 加上标签
                flat_sequence = np.array(sequence_buffer).flatten().tolist()
                flat_sequence.append(GESTURE_LABEL)

                df = pd.DataFrame([flat_sequence])

                if os.path.exists(OUTPUT_CSV_PATH):
                    df.to_csv(OUTPUT_CSV_PATH, mode='a', header=False, index=False)
                else:
                    # 创建列名太长了，这里可以不写header，或者用dummy header
                    df.to_csv(OUTPUT_CSV_PATH, mode='w', header=False, index=False)

                print(f"样本已保存到 {OUTPUT_CSV_PATH}")
                sequence_buffer = []  # 清空

        else:
            # 待机状态
            cv2.putText(img, f"Press 'r' to record '{GESTURE_LABEL}'",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # 绿色

        cv2.imshow("Get LSTM Features", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 只有在非录制、非倒计时状态下才启动倒计时
            if not is_recording and not is_counting_down:
                is_counting_down = True
                countdown_start_time = time.time()
                print(f"按下 'r'，开始 {COUNTDOWN_DURATION} 秒倒计时...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()