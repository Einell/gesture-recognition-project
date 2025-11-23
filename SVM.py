import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# ================= 配置路径 =================
DATA_PATH = 'gestures.csv'
MODEL_PATH = 'gesture_svm_model.pkl'
SCALER_PATH = 'svm_scaler.pkl'  # 保存标准化器 (StandardScaler)

# ================= 1. 加载数据 =================
try:
    data = pd.read_csv(DATA_PATH)
    print(f"成功加载数据: {data.shape}")
except FileNotFoundError:
    print(f"错误: 找不到文件 {DATA_PATH}")
    exit()

# 2. 准备特征和标签
# 假设最后一列是标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ================= 4. 特征标准化 (StandardScaler) =================
# 关键：在这里定义并训练 scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("特征数据已完成 Standard Scaling (Z-score 标准化)。")

# ================= 5. 训练 SVM 模型 =================
print("开始训练 SVM 模型...")
# 使用 C=0.1, kernel='linear' 匹配您之前的配置
svm_model = SVC(probability=True, kernel='linear', C=0.1, random_state=42)
svm_model.fit(X_train, y_train)
print("SVM 模型训练完成。")

# ================= 6. 评估模型 =================
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率 (测试集): {accuracy:.4f}")

# ================= 7. 保存模型和标准化器 =================
try:
    # 保存 SVM 模型
    joblib.dump(svm_model, MODEL_PATH)
    print(f"SVM 模型已保存至: {MODEL_PATH}")

    # 保存标准化器 (scaler) - 修复 NameError
    joblib.dump(scaler, SCALER_PATH)
    print(f"StandardScaler 已保存至: {SCALER_PATH}")

except Exception as e:
    print(f"保存文件时发生错误: {e}")