import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import threading  # 💡 新增：用于多线程执行
import gesture_control

# ================= 配置 =================
MODEL_PATH = 'gesture_lstm_model.keras'
CLASSES_PATH = 'lstm_classes.npy'
SEQUENCE_LENGTH = 20
THRESHOLD = 0.85
ACTION_COOLDOWN = 1.0
SKIP_FRAMES = 1  # 💡 新增：每隔2帧检测一次，降低负载

# ================= 初始化 MediaPipe =================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ================= 修复版：特征提取 (必须与训练采集一致) =================
def extract_keypoints(results):
    """
    与 get_lstm_features_3s.py 逻辑保持一致：
    1. 中心化 (减去手腕坐标)
    2. 归一化 (除以最大距离)
    3. 左右手排序
    """
    feature_vector = np.zeros(126)  # 2 * 21 * 3

    if not results.multi_hand_landmarks:
        return feature_vector

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # 获取左右手标签
        handedness = results.multi_handedness[idx].classification[0].label

        # 1. 提取坐标
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # 2. 中心化: 以手腕(0)为原点
        wrist = lm_array[0]
        lm_array = lm_array - wrist

        # 3. 归一化: 使用最大距离进行缩放 (与你的采集脚本匹配)
        max_dist = np.max(np.linalg.norm(lm_array, axis=1))
        if max_dist > 0:
            lm_array /= max_dist

        flat_features = lm_array.flatten()

        # 4. 根据左右手填入对应位置
        if handedness == 'Left':
            feature_vector[0:63] = flat_features
        else:
            feature_vector[63:126] = flat_features

    return feature_vector


# ================= 动作执行线程 =================
def run_action_in_thread(gesture, cap_ref, img_ref, landmarks_ref):
    """在独立线程中运行，防止 gesture_control 里的 time.sleep 卡死视频"""
    try:
        gesture_control.execute_gesture_action(gesture, cap_ref, img_ref, landmarks_ref)
    except Exception as e:
        print(f"Action Error: {e}")


# ================= 主程序 =================
def main():
    # 1. 加载模型
    try:
        model = load_model(MODEL_PATH)
        # 记得加 allow_pickle=True
        classes = np.load(CLASSES_PATH, allow_pickle=True)
        print(f"✅ 模型加载成功: {classes}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    cap = cv2.VideoCapture(0)

    sequence = []
    last_action_time = 0
    current_action = "Waiting..."
    confidence_score = 0.0
    frame_count = 0  # 用于跳帧计数

    with mp_hands.Hands(
            model_complexity=0,  # 0=Lite (最快), 1=Full
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
    ) as hands:

        print("🎥 启动成功！按 'q' 退出程序。")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 镜像翻转
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # 图像预处理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 💡 优化：跳帧检测
            # 只有当帧数能被 (SKIP_FRAMES + 1) 整除时才运行 MediaPipe
            # 其他时候只显示画面，不处理，极大提升流畅度
            if frame_count % (SKIP_FRAMES + 1) == 0:
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # 特征提取与预测
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]

                if len(sequence) == SEQUENCE_LENGTH:
                    # 只有检测到手的时候才进行预测，减少全0数据的干扰
                    if results.multi_hand_landmarks:
                        input_data = np.expand_dims(sequence, axis=0)
                        res = model.predict(input_data, verbose=0)[0]
                        best_idx = np.argmax(res)
                        confidence_score = res[best_idx]
                        predicted_gesture = classes[best_idx]

                        # 执行逻辑
                        if confidence_score > THRESHOLD:
                            # ----------------------------------------------------
                            # 💡 优化 1：过滤“背景”和“冷却中”动作
                            # ----------------------------------------------------
                            # 假设你增加了 'background' 类别
                            if predicted_gesture == 'background' or predicted_gesture == 'static':
                                current_action = "Static/Background"
                                # 即使置信度高，也不执行任何操作
                                pass

                            # 优化 2：如果识别出有效的动作
                            elif (time.time() - last_action_time) > ACTION_COOLDOWN:
                                current_action = predicted_gesture
                                print(f"🚀 执行: {predicted_gesture} ({confidence_score:.2f})")

                                first_hand = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

                                # 关键修改：使用 Thread 启动动作
                                action_thread = threading.Thread(
                                    target=run_action_in_thread,
                                    args=(predicted_gesture, cap, frame, first_hand)
                                )
                                action_thread.start()

                                last_action_time = time.time()
                    else:
                        # 没手的时候
                        current_action = "No Hand"
                        confidence_score = 0.0

            # 绘制 UI (每一帧都画)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 如果有之前的检测结果，可以画一下（可选，这里为了流畅度只画简单的）
            if 'results' in locals() and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 信息条
            cv2.rectangle(image, (0, 0), (640, 40), (0, 0, 0), -1)
            color = (0, 255, 0) if (time.time() - last_action_time) > ACTION_COOLDOWN else (0, 0, 255)
            cv2.putText(image, f"{current_action} ({confidence_score:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow('Gesture Control', image)

            # 💡 退出逻辑：使用 waitKey(1) 提高响应速度
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("正在退出...")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()