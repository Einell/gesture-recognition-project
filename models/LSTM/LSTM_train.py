# 基于LSTM的手势分类，训练+评估
import pandas as pd
import numpy as np
import os

import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# 参数
CSV_FILE = '../../data/gestures_lstm.csv'  # 数据文件
MODEL_PATH = 'gesture_lstm_model.h5'  # 模型路径
SEQUENCE_LENGTH = 20  # 序列长度
FEATURES_PER_FRAME = 126  # 单帧特征长度
N_SPLITS = 10  # K折交叉验证的折数
TEST_SIZE = 0.2 # 测试集比例


# 绘制训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # 绘制准确率曲线
    ax1.plot(history.history['accuracy'], label='train accuracy')
    ax1.plot(history.history['val_accuracy'], label='Verification accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    # 绘制损失曲线
    ax2.plot(history.history['loss'], label='train loss')
    ax2.plot(history.history['val_loss'], label='Verification')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
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
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
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

    accuracy_sk = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"\n评估指标:")
    print(f"准确率 (Accuracy): {accuracy_sk:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数 (F1-Score): {f1:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, classes)
    return loss, accuracy, y_pred, y_true

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, activation='tanh',
             input_shape=input_shape),
        LSTM(256, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    if not os.path.exists(CSV_FILE):
        print(f"未找到数据文件: {CSV_FILE}")
        return

    # 数据加载与预处理
    TOTAL_FEATURES = SEQUENCE_LENGTH * FEATURES_PER_FRAME
    dtype_dict = {i: np.float32 for i in range(TOTAL_FEATURES)}
    dtype_dict[TOTAL_FEATURES] = str
    df = pd.read_csv(CSV_FILE, header=None, dtype=dtype_dict, skiprows=1)

    initial_rows = len(df)
    df = df.dropna()
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"已移除 {rows_dropped} 行包含缺失值的数据！")

    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    n_samples = X_raw.shape[0]
    X_full = X_raw.reshape(n_samples, SEQUENCE_LENGTH, FEATURES_PER_FRAME)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    classes = le.classes_
    num_classes = len(classes)
    print(f"识别到的类别: {classes}")
    np.save('lstm_classes.npy', classes)

    # 划分测试集
    X_cv, X_test_final, y_cv_encoded, y_test_encoded_final = train_test_split(
        X_full, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
    )

    y_cv_categorical = to_categorical(y_cv_encoded)
    y_test_categorical_final = to_categorical(y_test_encoded_final)

    print(f"\n数据已划分为:")
    print(f"训练/验证集大小 (80%): {X_cv.shape[0]}")
    print(f"独立测试集大小 (20%): {X_test_final.shape[0]}")

    # 交叉验证设置与循环
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_accuracies = []
    best_val_accuracy = 0.0
    best_model = None

    print(f"\n--- 开始 {N_SPLITS} 折分层交叉验证---")

    for fold, (train_index, val_index) in enumerate(skf.split(X_cv, y_cv_encoded), 1):
        print(f"\n======== 折叠 {fold}/{N_SPLITS} ========")

        # 划分训练集和验证集
        X_train, X_val = X_cv[train_index], X_cv[val_index]
        y_train, y_val = y_cv_categorical[train_index], y_cv_categorical[val_index]

        # 每次折叠都重新初始化一个模型
        model = build_lstm_model(
            input_shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME),
            num_classes=num_classes
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )

        print(f"开始训练 折叠 {fold}...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0  # 设为0以减少输出
        )

        # 评估当前折叠
        print(f"评估 折叠 {fold}...")
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"折叠 {fold} 验证集准确率: {accuracy:.4f}")
        fold_accuracies.append(accuracy)

        # 保存最佳模型
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model = model
            best_model.save(MODEL_PATH)
            print(f"已保存当前最佳模型到 {MODEL_PATH} (准确率: {best_val_accuracy:.4f})")

    # 评估
    print("\n--- 交叉验证调优总结 ---")
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"各折叠验证准确率: {fold_accuracies}")
    print(f"平均验证准确率: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    print(f"最佳模型已保存到: {MODEL_PATH}, 准确率: {best_val_accuracy:.4f}")

    # 评估
    if best_model:
        print("\n--- 使用最佳模型测试集评估 (Final Hold-out Test) ---")
        final_model = keras.models.load_model(MODEL_PATH)
        evaluate_model(final_model, X_test_final, y_test_categorical_final, classes)


if __name__ == '__main__':
    main()