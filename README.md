# 基于计算机视觉的实时手势识别与控制系统

## 项目概述

本项目是一个基于计算机视觉的手势识别系统，能够通过摄像头捕捉用户手势，识别14种静态手势和4种动态手势，并转换为对电脑的相应控制操作。系统结合了传统机器学习（SVM）和深度学习（LSTM）方法，实现了高效、准确的手势识别与控制功能。
目前仅支持macOS系统，在windows系统中存在问题，请自行解决。
## 核心功能

### 手势识别能力
- **静态手势（14种）**：包括握拳、张开手掌、点赞、OK手势等。
  - 识别准确率：≥95%
  - 使用SVM分类器
- **动态手势（4种）**：包括握拳打开、关闭、放大、缩小。
  - 识别准确率：≥80%
  - 使用LSTM时序模型

### 电脑控制功能
- **媒体控制**：播放/暂停、音量调节、全屏切换、快进、快退
- **演示控制**：下一页、上一页、开始演示
- **系统控制**：音乐播放、保存、退出、鼠标移动、左右键点击
- **自定义扩展**：可根据需要添加更多控制功能

## 系统架构

```
数据采集 → 特征提取  →  手势识别  →  控制执行
    │       │           │          │
    │    MediaPipe  SVM/LSTM    PyAutoGUI
    │    (手部关键点) (分类模型)   (系统控制)
 OpenCV
(视频流处理)
```

## 依赖环境

### Python版本
- Python == 3.10.11

### 核心依赖库
```
pandas~=2.3.3
numpy~=1.26.4
pynput~=1.8.1
PyAutoGUI~=0.9.54
screeninfo~=0.8.1
opencv-python~=4.8.1.78
mediapipe~=0.10.21
joblib~=1.5.2
tensorflow~=2.19.1
matplotlib~=3.10.7
seaborn~=0.13.2
scikit-learn~=1.7.2
```

## 项目结构

```
Gesture_recognition/
├── data/
│   ├── LSTM_data/          # 原始记录的动态手势数据
│   ├── SVM_data/           # 原始记录的静态手势数据
│   ├── gestures.csv        # SVM训练测试数据集
│   └── gestures_lstm-3.csv  # LSTM训练测试数据集
├── models/
│   ├── LSTM/        # 动态手势训练及生成模型
│   │    ├── gesture_lstm_model.h5     #动态手势LSTM模型
│   │    ├── gesture_lstm_model.keras     #动态手势LSTM模型
│   │    ├── lstm_classes.npy     # 动态手势类别
│   │    └── LSTM_train.py     # 动态手势LSTM训练脚本
│   └── SVM/        # 动态手势LSTM模型
│        ├── gesture_svm_model.pkl     #静态手势SVM模型
│        └── SVM_train.py     # 静态手势SVM训练脚本
├── src/
│   ├── controllers/        #控制文件，包括手势、鼠标、音乐控制
│   ├── data_collections/        # 数据采集，包括动态和静态手势数据采集
│   └── utils/        # 数据合并工具，用于将静态手势数据与动态手势数据合并
├── tests/                   # 测试脚本
├── resources/                   # 音乐文件
├── demo/                   # 演示
├── requirements.txt         # 依赖列表
├── README.md         # 项目说明
└── main.py           # 系统启动主程序
```

## 快速开始

### 1. 环境安装
```bash
# 克隆项目
git clone <repository-url>
cd Gesture_recognition

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
数据存放在data目录下，包含静态手势数据集gestures.csv和动态手势数据集gestures_lstm-3.csv。

### 3. 模型训练
模型存放在models目录下，包含静态手势模型gesture_svm_model.pkl和动态手势模型gesture_lstm_model.h5。

### 4. 运行系统
```bash
# 启动手势控制系统
python main.py
```

## 模型详细信息

### 静态手势识别模型（SVM）
- **输入特征**：单只手21个关键点 × 3个坐标（x, y, z）
- **特征维度**：63维特征向量
- **特征处理**：标准化,降维
- **分类器**：SVM带RBF核
- **准确率**：准确率≥95%

### 动态手势识别模型（LSTM）
- **输入特征**：20帧序列 × 双手 × 21关键点 × 3坐标
- **特征维度**：20×2×21×3 = 2520个时序特征
- **网络结构**：
  - LSTM层：128个单元（第一层）+ 256个单元（第二层）
  - Dropout层：0.3
  - 全连接层：128个神经元
  - 输出层：4个动态手势类别
- **准确率**：准确率≥80%

## 手势控制映射

### 静态手势示例
| 手势  | 动作           | 电脑控制   |
|-----|--------------|--------|
| ✊   | 握拳           | 等待动作   |
| ✋   | 手掌           | 播放/暂停  |
| 👍  | 向上拇指         | 方向键向上  |
| 👎  | 向下拇指         | 方向键向下  |
|     | 向左拇指         | 方向键向左  |
|     | 向右拇指         | 方向键向右  |
| 👌  | OK           | 保存/确定  |
| 🫷  | 举手           | 退出     |
| 🫰  | 打响指          | 播放音乐   |
| ✌️  | 比耶️          | 鼠标移动   |
| ✌️  | 比耶️但食指合并指向镜头 | 鼠标滚轮   |
| 🖕️ | 中指           | 鼠标右键点击 |
| 👆️ | 食指️          | 鼠标左键点击 |
| L️  | L手势          | 截屏     |

### 动态手势示例
| 手势      | 动作      | 电脑控制 |
|---------|---------|------|
| 👊->🖐️ | 握拳顺时针打开 | 全屏   |
| 🖐️->👊 | 手掌逆时针握拳 | 退出全屏 |
| ⬅️✋🤚➡️ | 手掌分开    | 放大   |
| ✋➡️⬅️🤚 | 手掌合并    | 缩小   |

## 性能指标

### 静态手势识别
- 准确率: 0.9968
- 精确率: 0.9976
- 召回率: 0.9976
- F1分数: 0.9976

### 动态手势识别
- 测试集准确率: 1.0000
- 精确率: 1.0000
- 召回率: 1.0000
- F1分数: 1.0000

### 系统性能
- 处理延迟：< 100ms
- 摄像头要求：720p及以上
- 内存占用：~500MB
- CPU使用率：~15-25%

## 自定义配置

### 添加新手势
1. 数据采集 
  ```bash
  # 采集手势数据（需要摄像头）运行并输入手势标签，'s'保存,'r'录制，'q'退出
  python src/data_collection/get_SVM_features.py
  python src/data_collection/get_lstm_features.py
  # 合并数据
  python src/utils/merge.py
  python src/utils/merge_lstm.py
  ```
2. 重新训练模型
```bash
# 训练静态手势模型（SVM）
python models/SVM/SVM_train.py

# 训练动态手势模型（LSTM）
python models/LSTM/LSTM_train.py
```
3. 在'src/controllers/gestures_control.py中配置控制映射
4. 重新运行主函数

### 调整控制灵敏度
在src/controllers/mouse_controller.py
self.frameR = 100  #帧缩减像素，修改控制框的大小，frameR越大框越小，鼠标控制越灵敏
self.smoothening = 5  # 平滑系数，值越大越平滑，但延迟越高
self.SCROLL_THRESHOLD = 0.015 # 滚动阈值，值越小越灵敏

在main.py
SVM_PROB_THRESHOLD = 0.7 # 概率阈值
SVM_STABILITY_FRAMES = 5 # 连续帧数
LSTM_SEQ_LENGTH = 20 # 序列长度
LSTM_PROB_THRESHOLD = 0.8 # 概率阈值
LSTM_COOLDOWN = 1.0 # 冷却时间
MOVEMENT_THRESHOLD = 0.015 # 鼠标控制阈值，阈值调高：降低灵敏度，让鼠标操作更难触发动态模式

## 项目报告要点

### 技术亮点
1. **双模型架构**：结合SVM和LSTM，兼顾静态与动态手势识别
2. **MediaPipe集成**：高效、准确的手部关键点检测
3. **实时性能**：满足实时控制需求，延迟低
4. **模块化设计**：易于扩展和维护
5. **用户友好**：提供配置文件和自定义选项

### 创新点
- 使用个人采集的数据集训练，提高个性化识别精度
- 结合传统机器学习与深度学习的优势
- 实现细粒度的手势控制，支持多种应用场景

### 应用场景
- 演讲演示控制
- 媒体播放控制
- 无障碍辅助技术
- 游戏交互
- 智能家居控制

## 测试与验证

### 单元测试
```bash
# 运行SVM静态手势测试
python tests/SVM_reg.py
# 运行LSTM动态手势测试
python tests/lstm_reg.py
```


## 致谢

- **MediaPipe**：Google提供的优秀手部关键点检测方案
- **OpenCV**：计算机视觉基础库
- **Scikit-learn**：机器学习工具库
- **TensorFlow**：深度学习框架支持

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目作者：陈妙
- 课程：人工智能导论
- 邮箱：2927261741@qq.com

---

**提示**：使用前请确保摄像头正常工作，并在充足光照环境下使用本系统以获得最佳识别效果。