import cv2
import mediapipe as mp
import time
import numpy as np
import joblib
import gesture_control as control
import threading
import mouse_controller as mc

# ... (其他导入保持不变)

# ================= 配置区域 =================
PROBABILITY_THRESHOLD = 0.8  # 只有当概率大于此值时才执行操作
MODEL_PATH = 'gesture_svm_model.pkl'  # 模型文件路径
CONTINUOUS_GESTURES = {'right_mouse', 'right_mouse_roll', 'volume_control'}

# 【新增配置】窗口缩放比例
# 1.0 表示不缩放 (全尺寸 1280x720)
# 0.7 表示缩放至 70% (约 896x504)
WINDOW_SCALE = 0.7
# ... (其他配置和类定义保持不变) ...


# ================= 主程序初始化 =================
# ... (代码同原文件) ...

cap = cv2.VideoCapture(0)
# 摄像头原始分辨率 (不对这个分辨率进行缩放，MediaPipe 基于此计算坐标)
wCam, hCam = 1280, 720
cap.set(3, wCam)
cap.set(4, hCam)

# Frame Reduction 值 (在原始分辨率下的像素距离)
frameR = 200

print("系统启动完成。按 'q' 退出。")

# ================= 主循环 =================
while True:
    ret, img = cap.read()
    if not ret:
        print("无法读取摄像头数据")
        continue

    # 镜像翻转 + 颜色转换
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ================== 绘制操作区域矩形框 (基于原始 wCam, hCam) ==================
    # 矩形框坐标仍基于原始分辨率 wCam, hCam 计算
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)
    # ==============================================================================

    # 手部检测
    results = hands.process(imgRGB)

    # ... (手势检测和执行逻辑不变) ...

    # 计算并显示 FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # 【核心修改】：缩放窗口
    if WINDOW_SCALE != 1.0:
        # 使用 cv2.resize 缩放图像
        display_w = int(wCam * WINDOW_SCALE)
        display_h = int(hCam * WINDOW_SCALE)
        img_display = cv2.resize(img, (display_w, display_h))
    else:
        img_display = img  # 不缩放

    # 显示画面
    # 【修改】：显示缩放后的图像
    cv2.imshow("Gesture Recognition (Scaled)", img_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()