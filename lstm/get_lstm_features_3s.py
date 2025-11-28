import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# ================= 配置 =================
GESTURE_LABEL = 'zoom_out'  # 修改这里来录制不同的手势
OUTPUT_CSV_PATH = f'lstm-3/{GESTURE_LABEL}.csv'
SEQUENCE_LENGTH = 20  # 序列长度 (帧数)
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
# 特征总长度 = 20帧 * (2只手 * 21点 * 3坐标) = 3780 (写入CSV的一行)
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
    优化版: 使用 '手腕到中指根部' 的距离进行归一化，保证尺度一致性。
    """
    # 初始化全0向量 (126,)
    feature_vector = np.zeros(126)  # 2 * 21 * 3

    if not results.multi_hand_landmarks:
        return feature_vector, False  # 返回 False 表示没检测到手

    hands_found = 0

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        hands_found += 1
        handedness = results.multi_handedness[idx].classification[0].label

        # 提取坐标
        lm_array = []
        for lm in hand_landmarks.landmark:
            lm_array.append([lm.x, lm.y, lm.z])
        lm_array = np.array(lm_array)

        # A. 中心化: 以手腕(0)为原点
        wrist = lm_array[0]
        lm_array = lm_array - wrist

        # B. 尺度归一化: 使用 手腕(0) 到 中指指根(9) 的欧氏距离
        # 这个距离是物理固定的，不会随手指弯曲而改变
        palm_size = np.linalg.norm(lm_array[0] - lm_array[9])

        # 防止除以0
        if palm_size < 1e-6:
            palm_size = 1

        lm_array /= palm_size  # 缩放

        flat_features = lm_array.flatten()

        # C. 左右手对齐 (注意：镜像模式下，MediaPipe的Left/Right可能与视觉相反)
        # 建议实际测试一下，如果发现反了，互换下面的判定逻辑
        if handedness == 'Left':
            feature_vector[0:63] = flat_features
        else:
            feature_vector[63:126] = flat_features

    # 如果需要强制双手数据，可以在这里判断 hands_found 数量
    return feature_vector, True


def main():
    cap = cv2.VideoCapture(0)
    sequence_buffer = []

    # --- 新增/修改的状态变量 ---
    is_recording = False
    is_counting_down = False
    countdown_start_time = 0
    COUNTDOWN_DURATION = 1  # 倒计时秒数
    # --- -------------------- ---
    save_count = 0
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
            # 获取特征 和 是否检测到手的标志
            feat, hand_detected = extract_dual_hand_features(results)
            if not hand_detected:
                print("❌ 丢失追踪，录制中止！")
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

                # 自动添加表头 (如果是新文件)
                header = not os.path.exists(OUTPUT_CSV_PATH)
                df.to_csv(OUTPUT_CSV_PATH, mode='a', header=header, index=False)

                print(f"✅ 样本 {save_count} 已保存! (当前动作: {GESTURE_LABEL})")
                sequence_buffer = []

                # ⚡️ 体验优化: 自动重启倒计时 (连拍模式)
                # 如果你想录完一次暂停，把下面这两行注释掉即可
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
            # 只有在非录制、非倒计时状态下才启动倒计时
            if not is_recording and not is_counting_down:
                is_counting_down = True
                countdown_start_time = time.time()
                print(f"按下 'r'，开始 {COUNTDOWN_DURATION} 秒倒计时...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()