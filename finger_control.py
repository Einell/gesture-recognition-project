import pyautogui
import cv2


# 全局变量用于存储轨迹点和时间
trail_points = []  # 存储轨迹点 (x, y, timestamp)
click_timer = None  # 记录指尖停留开始时间
CLICK_DELAY = 2  # 停留2秒后点击

def finger_control(frame, hand_landmarks):
    global trail_points, click_timer

    # 获取屏幕分辨率
    screen_width, screen_height = pyautogui.size()

    # 获取食指指尖坐标 ( landmark[8] 是食指指尖)
    index_finger_tip = hand_landmarks.landmark[8]

    # 将手势坐标转换为屏幕坐标
    # 注意：图像是镜像的，需要反转x轴
    frame_height, frame_width, _ = frame.shape
    screen_x = screen_width - (index_finger_tip.x * frame_width)
    screen_y = index_finger_tip.y * frame_height

    # 移动鼠标
    pyautogui.moveTo(screen_x, screen_y)

    # 记录轨迹点
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    trail_points.append((screen_x, screen_y, current_time))

    # 绘制轨迹
    draw_trail(frame, trail_points)

    # 检测指尖停留
    if len(trail_points) > 10:
        # 计算最近10帧的移动距离
        last_10_points = trail_points[-10:]
        distances = []
        for i in range(1, len(last_10_points)):
            dx = last_10_points[i][0] - last_10_points[i-1][0]
            dy = last_10_points[i][1] - last_10_points[i-1][1]
            distances.append(np.sqrt(dx**2 + dy**2))

        avg_distance = np.mean(distances)

        # 如果移动距离小于阈值，开始计时
        if avg_distance < 5:  # 阈值可以调整
            if click_timer is None:
                click_timer = current_time
            elif current_time - click_timer >= CLICK_DELAY:
                # 停留超过2秒，执行点击
                pyautogui.click()
                print("执行鼠标左键点击")
                click_timer = None  # 重置计时器
        else:
            click_timer = None  # 移动时重置计时器

    # 限制轨迹点数量，避免内存占用过多
    if len(trail_points) > 100:
        trail_points.pop(0)

def draw_trail(frame, trail_points):
    """绘制随时间淡化的轨迹"""
    frame_height, frame_width, _ = frame.shape
    current_time = cv2.getTickCount() / cv2.getTickFrequency()

    # 绘制轨迹
    for i in range(1, len(trail_points)):
        x1, y1, t1 = trail_points[i-1]
        x2, y2, t2 = trail_points[i]

        # 根据时间计算透明度
        alpha = 1.0 - (current_time - t2) / 2.0  # 2秒内逐渐淡化
        alpha = max(0, min(1, alpha))

        # 将屏幕坐标转换为帧坐标
        frame_x1 = int((frame_width - x1) / frame_width * frame_width)
        frame_y1 = int(y1 / frame_height * frame_height)
        frame_x2 = int((frame_width - x2) / frame_width * frame_width)
        frame_y2 = int(y2 / frame_height * frame_height)

        # 绘制线段
        cv2.line(frame, (frame_x1, frame_y1), (frame_x2, frame_y2),
                 (0, 255, 255), 2, cv2.LINE_AA)

    # 绘制当前指尖位置
    if trail_points:
        x, y, _ = trail_points[-1]
        frame_x = int((frame_width - x) / frame_width * frame_width)
        frame_y = int(y / frame_height * frame_height)
        cv2.circle(frame, (frame_x, frame_y), 5, (0, 0, 255), -1)