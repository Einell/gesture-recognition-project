import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping  # 引入 EarlyStopping
import os

# ================= 配置 =================
CSV_FILE = 'gestures_lstm.csv'
MODEL_PATH = 'gesture_lstm_model.keras'  # 推荐使用新格式
SEQUENCE_LENGTH = 20
FEATURES_PER_FRAME = 84


def main():
    if not os.path.exists(CSV_FILE):
        print(f"未找到数据文件: {CSV_FILE}")
        return

    print("正在加载数据...")

    # 核心修复 1: 解决表头错误和 dtype 错误
    TOTAL_FEATURES = SEQUENCE_LENGTH * FEATURES_PER_FRAME
    dtype_dict = {i: np.float32 for i in range(TOTAL_FEATURES)}
    dtype_dict[TOTAL_FEATURES] = str
    # 使用 skiprows=1 跳过表头
    df = pd.read_csv(CSV_FILE, header=None, dtype=dtype_dict, skiprows=1)

    # 核心修复 2: 清理缺失值
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"警告: 已移除 {rows_dropped} 行包含缺失值的数据。")

    # 分离特征和标签
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    # Reshape X 到 (samples, timesteps, features)
    n_samples = X_raw.shape[0]
    X = X_raw.reshape(n_samples, SEQUENCE_LENGTH, FEATURES_PER_FRAME)

    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    y_categorical = to_categorical(y_encoded)
    classes = le.classes_
    print(f"识别到的类别: {classes}")
    np.save('lstm_classes.npy', classes)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # 构建模型 (关键优化区域)
    model = Sequential([
        # 优化点 1: LSTM 激活函数改为 'tanh'
        LSTM(64, return_sequences=True, activation='tanh', input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
        # 优化点 1: LSTM 激活函数改为 'tanh'
        LSTM(128, return_sequences=False, activation='tanh'),
        Dropout(0.3),  # 优化点 2: 增加 Dropout 率至 0.3
        # 优化点 3: Dense 层可保持 'relu'，因为它计算效率高且在非序列部分表现良好
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])

    # 优化点 4: 降低学习率
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 优化点 5: 添加 Early Stopping 回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # 容忍 15 轮没有改善，然后停止
        restore_best_weights=True  # 停止后恢复到最佳权重
    )

    print("开始训练...")
    # 增加 epochs 到 100 轮（由 Early Stopping 负责提前停止）
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]  # 启用 Early Stopping
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy:.4f}")

    model.save(MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")


if __name__ == '__main__':
    main()