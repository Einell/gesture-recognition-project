# 此乃手势控制之主程序
# 运行程序，即可通过手势进行相应控制
import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import threading
from collections import deque
from tensorflow.keras.models import load_model

# 引入你的控制脚本
import gesture_control as control
import mouse_controller as mc

# 配置
# 窗口大小
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 400
# 静态SVM配置
SVM_MODEL_PATH = 'gesture_svm_model.pkl'
SVM_PROB_THRESHOLD = 0.7 # 概率阈值
SVM_STABILITY_FRAMES = 5 # 连续帧数
# 连续静态手势：移动鼠标，鼠标滚轮和音量控制
STATIC_CONTINUOUS_GESTURES = {'right_mouse', 'right_mouse_roll', 'volume_control'}
# 动态LSTM配置
LSTM_MODEL_PATH = 'gesture_lstm_model.h5'
LSTM_CLASSES_PATH = 'lstm_classes.npy'
LSTM_SEQ_LENGTH = 20 # 序列长度
LSTM_PROB_THRESHOLD = 0.8 # 概率阈值
LSTM_COOLDOWN = 1.0 # 冷却时间
# 运动检测配置
MOVEMENT_BUFFER_SIZE = 10 # 检测历史数据长度
MOVEMENT_THRESHOLD = 0.015 # 鼠标控制阈值，阈值调高：降低灵敏度，让鼠标操作更难触发动态模式


def load_compatible_model(model_path):
    """兼容性加载模型"""
    try:
        # 最简单的方法：使用 compile=False
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"简单加载失败: {e}")
        # 如果还是失败，尝试更复杂的修复
        pass

    # 如果简单方法失败，尝试忽略不支持的参数
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.layers import LSTM

    custom_objects = {
        'LSTM': lambda *args, **kwargs: LSTM(
            *args,
            **{k: v for k, v in kwargs.items()
               if k not in ['batch_input_shape', 'time_major', 'unroll']}
        )
    }

    try:
        model = keras_load_model(model_path,
                                 custom_objects=custom_objects,
                                 compile=False)
        return model
    except Exception as e:
        print(f"自定义加载也失败: {e}")
        raise
# 运动检测器
class HandMotionDetector:
    def __init__(self, buffer_size=10, threshold=0.015):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.histories = {'Left': deque(maxlen=buffer_size), 'Right': deque(maxlen=buffer_size)}
    # 更新手腕坐标
    def update(self, landmarks, label):
        wrist = landmarks.landmark[0]
        self.histories[label].append([wrist.x, wrist.y])
    # 检查是否有任意一只手在剧烈运动
    def is_moving_violently(self):
        for label, history in self.histories.items():
            if len(history) < self.buffer_size: continue
            std = np.std(np.array(history), axis=0)
            avg_std = np.mean(std)
            if avg_std > self.threshold:
                return True
        return False
    # 重置手腕坐标
    def reset_history(self, label):
        if label in self.histories: self.histories[label].clear()

# 异步执行器
class AsyncExecutor:
    def __init__(self):
        self.running = False
        self.lock = threading.Lock()

    def run(self, func, args):
        if not self.running:
            t = threading.Thread(target=self._task, args=(func, args))
            t.daemon = True
            t.start()

    def _task(self, func, args):
        with self.lock:
            self.running = True
        try:
            func(*args)
        except Exception as e:
            print(f"Action Error: {e}")
        finally:
            with self.lock:
                self.running = False


# 手势平滑滤波
class HandSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev_landmarks = {'Left': None, 'Right': None}

        try:
            from mediapipe.framework.formats import landmark_pb2
            self.NormalizedLandmarkList = landmark_pb2.NormalizedLandmarkList
        except ImportError:
            try:
                from mediapipe.python.framework.formats import landmark_pb2
                self.NormalizedLandmarkList = landmark_pb2.NormalizedLandmarkList
            except ImportError as e:
                raise ImportError(f"无法找到 NormalizedLandmarkList: {e}")
    # 平滑
    def smooth(self, current_landmarks_proto, label):
        current_data = np.array([[lm.x, lm.y, lm.z] for lm in current_landmarks_proto.landmark])

        if self.prev_landmarks[label] is None:
            self.prev_landmarks[label] = current_data
            return current_landmarks_proto

        # 指数移动平均公式
        smoothed_data = self.alpha * current_data + (1 - self.alpha) * self.prev_landmarks[label]
        self.prev_landmarks[label] = smoothed_data
        # 转为 proto
        smoothed_landmarks_proto = self.NormalizedLandmarkList()
        for i in range(len(current_landmarks_proto.landmark)):
            landmark = smoothed_landmarks_proto.landmark.add()
            landmark.x = smoothed_data[i, 0]
            landmark.y = smoothed_data[i, 1]
            landmark.z = smoothed_data[i, 2]

        return smoothed_landmarks_proto


# 静态手势特征提取
def get_svm_features(hand_landmarks):
    lm = np.array([[l.x, l.y, l.z] for l in hand_landmarks.landmark])
    lm = lm - lm[0]
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0: lm /= max_dist
    return lm.flatten().reshape(1, -1)

# 动态手势特征提取
def get_lstm_features(results):
    feats = np.zeros(126)

    if results.multi_hand_landmarks:
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label

            lm = np.array([[l.x, l.y, l.z] for l in landmarks.landmark])
            lm = lm - lm[0]
            max_dist = np.max(np.linalg.norm(lm, axis=1))
            if max_dist > 0: lm /= max_dist
            flat = lm.flatten()

            if label == 'Left':
                feats[0:63] = flat
            elif label == 'Right':
                feats[63:126] = flat

    return feats

# 主程序
def main():
    # 加载模型
    try:
        svm_model = joblib.load(SVM_MODEL_PATH)

        # 使用兼容性加载
        lstm_model = load_compatible_model(LSTM_MODEL_PATH)

        # 如果需要，可以重新编译（可选）
        try:
            lstm_model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        except:
            pass  # 如果已经有编译信息，跳过

        lstm_classes = np.load(LSTM_CLASSES_PATH, allow_pickle=True)
        print(f"模型加载完毕。\nSVM: {svm_model.classes_}\nLSTM: {lstm_classes}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=4)
    handConnStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=3)
    # 状态变量
    motion_detector = HandMotionDetector(threshold=MOVEMENT_THRESHOLD)
    executor = AsyncExecutor()
    hand_smoother = HandSmoother(alpha=0.6)

    lstm_seq = []
    last_lstm_time = 0
    prev_time = 0

    svm_state = {
        'Left': {'cmd': None, 'count': 0},
        'Right': {'cmd': None, 'count': 0}
    }

    cap = cv2.VideoCapture(0)
    cap.set(3, CAMERA_WIDTH)
    cap.set(4, CAMERA_HEIGHT)
    frame_r = mc.MOUSE_CONTROLLER.frameR

    while True:
        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # 绘制控制矩形框
        cv2.rectangle(frame, (frame_r, frame_r), (CAMERA_WIDTH - frame_r, CAMERA_HEIGHT - frame_r), (255, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (CAMERA_WIDTH, 80), (0, 0, 0), -1)

        mode = "NO HAND"
        display_results = []

        # 模式切换标志
        force_static_lock = False

        if results.multi_hand_landmarks:
            smoothed_landmarks = []
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                lbl = results.multi_handedness[idx].classification[0].label
                # 对当前手部关键点进行平滑
                smoothed_lms = hand_smoother.smooth(hand_lms, lbl)
                smoothed_landmarks.append(smoothed_lms)
                # 运动检测使用平滑后的关键点
                motion_detector.update(smoothed_lms, lbl)
                # 绘制平滑后的骨架
                mpDraw.draw_landmarks(frame, smoothed_lms, mp_hands.HAND_CONNECTIONS,handLmsStyle, handConnStyle)

            results.multi_hand_landmarks = smoothed_landmarks
            # 检测是否有连续静态手势
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                lbl = results.multi_handedness[idx].classification[0].label
                feats = get_svm_features(hand_lms)
                probs = svm_model.predict_proba(feats)[0]
                best_idx = np.argmax(probs)
                conf = probs[best_idx]
                gesture = svm_model.classes_[best_idx]

                # 默认显示颜色为白色
                text_color = (255, 255, 255)
                # 检查是否为高置信度的连续移动型手势
                if gesture in STATIC_CONTINUOUS_GESTURES and conf > SVM_PROB_THRESHOLD:
                    # 发现鼠标控制手势，强制锁定
                    force_static_lock = True
                    # 执行时高亮
                    text_color = (0, 255, 255)  # 高亮
                    control.execute_gesture_action(gesture, cap, frame, hand_lms)
                    # 实时手势不走防抖逻辑，但需要更新状态
                    svm_state[lbl]['cmd'] = gesture
                    svm_state[lbl]['count'] = 0

                # 存储识别结果和颜色
                display_results.append({
                    'text': f"{lbl}: {gesture} ({conf:.0%}) {'[LOCK]' if force_static_lock else ''}",
                    'color': text_color
                })
            # 判定模式
            is_moving_violently = motion_detector.is_moving_violently()

            if force_static_lock:
                mode = 'STATIC (LOCKED)'
            elif is_moving_violently:
                mode = 'DYNAMIC'
            else:
                mode = 'STATIC'

            # 动态模式
            if 'DYNAMIC' in mode:
                feats = get_lstm_features(results)
                lstm_seq.append(feats)
                lstm_seq = lstm_seq[-LSTM_SEQ_LENGTH:]

                # 清除静态计数器
                svm_state['Left']['count'] = 0
                svm_state['Right']['count'] = 0

                if len(lstm_seq) == LSTM_SEQ_LENGTH:
                    input_data = np.expand_dims(lstm_seq, axis=0)
                    pred = lstm_model.predict(input_data, verbose=0)[0]
                    best_idx = np.argmax(pred)
                    conf = pred[best_idx]
                    gesture = lstm_classes[best_idx]

                    text_color = (255, 255, 255)  # 默认白色

                    if conf > LSTM_PROB_THRESHOLD:
                        if time.time() - last_lstm_time > LSTM_COOLDOWN:
                            if gesture not in ['background', 'static']:
                                print(f"执行动态: {gesture}")
                                text_color = (0, 255, 0)
                                main_hand = results.multi_hand_landmarks[0]
                                executor.run(control.execute_gesture_action, (gesture, cap, frame, main_hand))
                                last_lstm_time = time.time()
                                lstm_seq = []  # 触发后清空序列

                    # 添加显示结果
                    display_results.append({
                        'text': f"LSTM: {gesture} ({conf:.0%})",
                        'color': text_color
                    })

            # 静态模式
            else:
                # 续向 LSTM 序列添加静态/背景帧，保证序列平稳
                feats = get_lstm_features(results)
                lstm_seq.append(feats)
                lstm_seq = lstm_seq[-LSTM_SEQ_LENGTH:]

                # 遍历每只手，独立识别
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    lbl = results.multi_handedness[idx].classification[0].label

                    # 如果这只手已经被 "LOCK" 逻辑处理过，则跳过重复的执行逻辑，只处理显示
                    is_locked = (force_static_lock and svm_state[lbl]['cmd'] in STATIC_CONTINUOUS_GESTURES)

                    # 重新提取特征进行预测
                    feats = get_svm_features(hand_lms)
                    probs = svm_model.predict_proba(feats)[0]
                    best_idx = np.argmax(probs)
                    conf = probs[best_idx]
                    gesture = svm_model.classes_[best_idx]

                    # 查找这只手的显示结果，并更新其颜色
                    result_entry = next((item for item in display_results if item['text'].startswith(f"{lbl}:")), None)

                    if not is_locked:
                        if conf > SVM_PROB_THRESHOLD:
                            state = svm_state[lbl]

                            # 离散手势
                            if gesture not in STATIC_CONTINUOUS_GESTURES:
                                if gesture == state['cmd']:
                                    state['count'] += 1
                                    if state['count'] >= SVM_STABILITY_FRAMES:
                                        print(f"执行静态: {gesture}")
                                        if result_entry: result_entry['color'] = (255, 0, 0)  # 红色高亮
                                        executor.run(control.execute_gesture_action, (gesture, None, None, hand_lms))
                                        state['count'] = 0
                                else:
                                    state['cmd'] = gesture
                                    state['count'] = 1

                                if state['count'] > 0 and result_entry:
                                    result_entry['color'] = (255, 165, 0)  # 计数中
                            else:
                                state['count'] = 0
                                state['cmd'] = gesture
                        else:
                            svm_state[lbl]['count'] = 0

                    # 如果没有被 LOCK 逻辑显示，且result_entry未存在，则添加
                    if not result_entry:
                        display_results.append({
                            'text': f"{lbl}: {gesture} ({conf:.0%})",
                            'color': (255, 255, 255)
                        })
                    # 如果是Lock逻辑添加的，但置信度低于阈值，则去掉LOCK标志
                    elif is_locked and conf <= SVM_PROB_THRESHOLD:
                        result_entry['text'] = result_entry['text'].replace(' [LOCK]', '')
                        result_entry['color'] = (255, 255, 255)  # 恢复白色


        else:
            # 无手时清空状态
            lstm_seq = []
            motion_detector.histories['Left'].clear()
            motion_detector.histories['Right'].clear()
            hand_smoother.prev_landmarks['Left'] = None
            hand_smoother.prev_landmarks['Right'] = None

        # 显示模式状态
        color = (0, 255, 0) if 'STATIC' in mode else (0, 165, 255)
        if mode == 'NO HAND': color = (0, 0, 255)
        cv2.putText(frame, f"MODE: {mode}", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        # 显示FPS
        fps_text = f"FPS: {int(fps)}"
        fps_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
        cv2.putText(frame, fps_text, (CAMERA_WIDTH - fps_size[0] - 10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0),
                    2)

        # 显示识别结果和概率
        y = 50
        for item in display_results:
            cv2.putText(frame, item['text'], (400, y), cv2.FONT_HERSHEY_PLAIN, 2, item['color'], 2)
            y += 30

        cv2.imshow('Merged Gesture System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()