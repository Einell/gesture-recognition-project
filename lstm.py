import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# ================= 配置 =================
CSV_FILE = 'gestures_lstm.csv'
MODEL_PATH = 'gesture_lstm_model.h5'
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 126  # 2 hands * 21 points * 3 coords


def main():
    if not os.path.exists(CSV_FILE):
        print(f"未找到数据文件: {CSV_FILE}")
        return

    print("正在加载数据...")
    # header=None 因为我们在收集时没写header
    df = pd.read_csv(CSV_FILE, header=None)

    # 分离特征和标签
    X_raw = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    # Reshape X 到 (samples, timesteps, features)
    # 原始行长度是 30 * 126 = 3780
    n_samples = X_raw.shape[0]
    X = X_raw.reshape(n_samples, SEQUENCE_LENGTH, FEATURES_PER_FRAME)

    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    y_categorical = to_categorical(y_encoded)

    classes = le.classes_
    print(f"识别到的类别: {classes}")

    # 保存标签映射，以便 inference 使用
    np.save('lstm_classes.npy', classes)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # 构建模型
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
        LSTM(128, return_sequences=False, activation='relu'),
        Dropout(0.2),  # 防止过拟合
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("开始训练...")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy:.4f}")

    model.save(MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")


if __name__ == '__main__':
    main()