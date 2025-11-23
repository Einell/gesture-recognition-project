# 基于支持向量机的手势分类
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 特征文件路径
feature_csv_path = 'gestures.csv'
# 模型保存路径
model_save_path = 'gesture_svm_model.pkl'
# 训练集比例
test_size = 0.3
# 随机种子
random_state = 42

# 数据加载
# 检查CSV文件是否存在
if not os.path.exists(feature_csv_path):
    print(f"错误: 特征文件 '{feature_csv_path}' 不存在！")
    exit(1)

# 读取CSV文件
df = pd.read_csv(feature_csv_path)

# 分离特征和标签
X = df.iloc[:, :-1].values # 特征（前63列）
y = df.iloc[:, -1].values # 标签 （最后一列）

print(f"数据加载完成。特征维度: {X.shape}, 标签数量: {y.shape[0]}")
print(f"手势类别: {np.unique(y)}") # 显示所有手势类别

# 划分训练集和测试集
print("\n正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y # 确保训练集和测试集中各类别的比例与原始数据一致
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 初始化并训练SVM模型
print("\n正在初始化并训练SVM模型...")
# 参数说明:
# C: 惩罚系数，控制模型复杂度；kernel: 核函数；gamma: 'rbf'核的带宽参数；class_weight: 'balanced' 会给样本量较少的类别赋予更大的权重，有助于处理不平衡数据集
svm_classifier = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=random_state,probability=True)

# 训练模型
svm_classifier.fit(X_train, y_train)

print("模型训练完成。")

# 模型评估

print("\n" + "="*50)
print("模型评估结果")

# 预测
y_pred = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n准确率 (Accuracy): {accuracy:.4f}")

# 生成并打印分类报告
print("\n分类报告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=np.unique(y)))

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 打印混淆矩阵
print("\n混淆矩阵 (Confusion Matrix):")
print(cm)


print("\n正在生成混淆矩阵热力图...")
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建热力图
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.unique(y),
    yticklabels=np.unique(y)
)
plt.title('SVM confusion matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 保存模型

if model_save_path:
    joblib.dump(svm_classifier, model_save_path)
    print("模型保存成功。")
