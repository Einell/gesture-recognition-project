# 基于LSTM的手势分类，训练+评估
import pandas as pd
import numpy as np
import os

# 设置环境变量使用 Keras 2
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# 参数
CSV_FILE = 'gestures_lstm-3.csv'  # 数据文件
MODEL_PATH = 'gesture_lstm_model.h5'  # 修改为 .h5 格式，更兼容
SEQUENCE_LENGTH = 20  # 序列长度
FEATURES_PER_FRAME = 126  # 单帧特征长度


# 绘制训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # 绘制准确率曲线
    ax1.plot(history.history['accuracy'], label='训练准确率')
    ax1.plot(history.history['val_accuracy'], label='验证准确率')
    ax1.set_title('模型准确率')
    ax1.set_xlabel('训练轮次')
    ax1.set_ylabel('准确率')
    ax1.legend()
    ax1.grid(True)
    # 绘制损失曲线
    ax2.plot(history.history['loss'], label='训练损失')
    ax2.plot(history.history['val_loss'], label='验证损失')
    ax2.set_title('模型损失')
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


# 评估模型
def evaluate_model(model, X_test, y_test, classes):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"测试集损失: {loss:.4f}")
    print(f"测试集准确率: {accuracy:.4f}")
    # 预测
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=classes))
    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, classes)
    return loss, accuracy, y_pred, y_true


def main():
    if not os.path.exists(CSV_FILE):
        print(f"未找到数据文件: {CSV_FILE}")
        return

    TOTAL_FEATURES = SEQUENCE_LENGTH * FEATURES_PER_FRAME
    dtype_dict = {i: np.float32 for i in range(TOTAL_FEATURES)}
    dtype_dict[TOTAL_FEATURES] = str
    df = pd.read_csv(CSV_FILE, header=None, dtype=dtype_dict, skiprows=1)

    # 清理缺失值
    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"已移除 {rows_dropped} 行包含缺失值的数据！")

    # 分离特征和标签
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    # 重塑 X 到 (samples, timesteps, features)
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )

    # ============ 使用简单的 Sequential 模型 ============
    print("构建LSTM模型...")

    # 使用 Sequential 模型（最简单，最兼容）
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh',
             input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
        LSTM(256, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])

    # 使用标准 Adam 优化器（不要用 legacy）
    model.compile(
        optimizer='adam',  # 直接使用字符串
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 打印模型摘要
    model.summary()

    # 添加早停
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    print("开始训练...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # 绘制训练历史
    plot_training_history(history)

    # 评估模型
    print("\n模型评估结果:")
    loss, accuracy, y_pred, y_true = evaluate_model(model, X_test, y_test, classes)

    # 保存模型
    model.save(MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")

    # 训练总结
    print(f"\n训练总结:")
    print(f"- 最终训练准确率: {history.history['accuracy'][-1]:.4f}")
    print(f"- 最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
    print(f"- 测试集准确率: {accuracy:.4f}")
    print(f"- 总训练轮次: {len(history.history['accuracy'])}")

    # 保存评估结果到文件
    with open('model_evaluation.txt', 'w') as f:
        f.write("LSTM手势分类模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试集准确率: {accuracy:.4f}\n")
        f.write(f"测试集损失: {loss:.4f}\n")
        f.write(f"类别数量: {len(classes)}\n")
        f.write(f"类别列表: {list(classes)}\n")
        f.write(f"总训练轮次: {len(history.history['accuracy'])}\n")

    print("\n✅ 训练完成！")


if __name__ == '__main__':
    main()