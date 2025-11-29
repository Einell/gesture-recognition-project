# 基于LSTM的手势分类，训练+评估
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping  # 引入 EarlyStopping
from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 参数
CSV_FILE = 'gestures_lstm-3.csv' # 数据文件
MODEL_PATH = 'gesture_lstm_model.keras' # 模型保存路径
SEQUENCE_LENGTH = 20 # 序列长度
FEATURES_PER_FRAME = 126 # 单帧特征长度

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

    TOTAL_FEATURES = SEQUENCE_LENGTH * FEATURES_PER_FRAME # 特征总数
    dtype_dict = {i: np.float32 for i in range(TOTAL_FEATURES)} # 设置数据类型
    dtype_dict[TOTAL_FEATURES] = str # 设置最后一列label为字符串
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

    # 将一维的手势特征数据重塑为LSTM模型所需的三维时序数据格式
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
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # 构建模型
    model = Sequential([
        # 第一层LSTM
        LSTM(128, return_sequences=True, activation='tanh', input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME)),
        # 第二层LSTM
        LSTM(256, return_sequences=False, activation='tanh'),
        # 随机30%神经元失活，防止过拟合
        Dropout(0.3),
        # 全连接层
        Dense(128, activation='relu'),
        # 输出层
        Dense(len(classes), activation='softmax')
    ])

    custom_optimizer = LegacyAdam(learning_rate=0.001) # 学习率0.001

    # 编译配置好的LSTM模型
    # 参数说明：
    # optimizer=custom_optimizer 使用定义的LegacyAdam优化器来更新模型权重
    # loss='categorical_crossentropy' 使用多分类交叉熵作为损失函数
    # metrics=['accuracy'] 使用准确率作为评估指标
    model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 添加早停
    early_stopping = EarlyStopping(
        monitor='val_loss', # 监控的指标
        patience=15,  # 容忍 15 轮没有改善，然后停止
        restore_best_weights=True  # 停止后恢复到最佳权重
    )

    print("开始训练...")
    history = model.fit(
        X_train, y_train,
        epochs=50, # 训练轮数
        batch_size=16, # 批量大小
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],  # 启用早停
        verbose = 1 # 显示训练进度
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


if __name__ == '__main__':
    main()