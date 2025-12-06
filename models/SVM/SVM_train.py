# 基于SVM的手势分类，训练+评估
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 参数
# 手势特征文件路径
feature_csv_path = '../../data/gestures.csv'
# 模型保存路径
model_save_path = 'gesture_svm_model.pkl'
# 训练集比例
test_size = 0.3
# 随机种子
random_state = 30

# 数据加载
# 检查CSV文件是否存在
if not os.path.exists(feature_csv_path):
    print(f"错误: 特征文件 '{feature_csv_path}' 不存在！")
    exit(1)

# 读取CSV文件，不读取表头
df = pd.read_csv(feature_csv_path, header=None)

# 分离特征和标签
X = df.iloc[:, :-1].values  # 特征（前63列）
y = df.iloc[:, -1].values  # 标签（最后一列）

print(f"数据加载完成。特征维度: {X.shape}, 标签数量: {y.shape[0]}")
print(f"手势类别: {np.unique(y)}")  # 显示所有手势类别

# 划分训练集和测试集
print("\n正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y  # 确保训练集和测试集中各类别的比例与原始数据一致
)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 初始化并训练SVM模型
print("\n正在初始化并训练SVM模型...")
# 参数说明:
# C: 惩罚系数，控制模型复杂度；kernel: 核函数；gamma: 'rbf'核的带宽参数；class_weight: 'balanced' 会给样本量较少的类别赋予更大的权重，有助于处理不平衡数据集
svm_classifier = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=random_state,
                     probability=True)

# 训练模型
svm_classifier.fit(X_train, y_train)

print("模型训练完成。")
# 保存模型
if model_save_path:
    joblib.dump(svm_classifier, model_save_path)
    print(f"\n模型已保存到: {model_save_path}")

# 模型评估
print("\n" + "=" * 50)
print("模型评估结果")

# 预测
y_pred = svm_classifier.predict(X_test)

# 计算准确率、平均精确率、召回率和F1分数
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"\n准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision_macro:.4f}")
print(f"召回率 (Recall): {recall_macro:.4f}")
print(f"F1分数 (F1-Score): {f1_macro:.4f}")

# 生成并打印分类报告
# print("\n分类报告 (Classification Report):")
# print(classification_report(y_test, y_pred, target_names=np.unique(y)))
# 生成打印混淆矩阵
# cm = confusion_matrix(y_test, y_pred)
# print("\n混淆矩阵 (Confusion Matrix):")
# print(cm)


# 绘制分类报告
def plot_classification_report(y_true, y_pred, class_names):
    from sklearn.metrics import classification_report
    import pandas as pd

    # 生成分类报告字典
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # 转换为DataFrame，排除支持度列
    report_df = pd.DataFrame(report_dict).transpose()
    metrics_df = report_df.iloc[:-3, :-1]  # 排除accuracy、macro avg、weighted avg的最后一行（support）

    # 创建分类报告热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', center=0.5, cbar_kws={'label': 'Score'})

    plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return metrics_df

# 绘制混淆矩阵
def plot_normalized_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 绘制每个类别的准确率
def plot_class_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracy, color='skyblue', alpha=0.7, edgecolor='black')
    plt.title('Accuracy per Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

# 绘制ROC曲线
def plot_roc_curves(X_test, y_test, model, class_names):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_test_bin = label_binarize(y_test, classes=class_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 8))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(class_names)))
    for i, color in zip(range(len(class_names)), colors):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# 绘制学习曲线
def plot_learning_curve(X_train, y_train, model, class_names):
    from sklearn.model_selection import learning_curve

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=random_state
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Learning Curve", fontsize=16, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"学习曲线绘制失败: {e}")

# 绘制精确率-召回率曲线
def plot_precision_recall_curve(X_test, y_test, model, class_names):
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import label_binarize

    try:
        y_test_bin = label_binarize(y_test, classes=class_names)

        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))

        for i, color in zip(range(len(class_names)), colors):
            precision, recall, _ = precision_recall_curve(
                y_test_bin[:, i], model.predict_proba(X_test)[:, i]
            )
            plt.plot(recall, precision, color=color, lw=2,
                     label=f'{class_names[i]}')

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves for All Classes', fontsize=16, fontweight='bold')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"精确率-召回率曲线绘制失败: {e}")

# 绘制预测置信度分布
def plot_prediction_confidence(X_test, model):
    try:
        proba = model.predict_proba(X_test)
        max_proba = np.max(proba, axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(max_proba, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Maximum Prediction Probability', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Confidence', fontsize=16, fontweight='bold')
        plt.axvline(x=0.8, color='red', linestyle='--', label='80% confidence threshold', linewidth=2)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"预测置信度分布绘制失败: {e}")

# 绘制特征重要性
def plot_feature_importance(model, class_names):
    plt.figure(figsize=(10, 6))

    if hasattr(model, 'coef_'):
        feature_importance = np.mean(np.abs(model.coef_), axis=0)
        top_features = np.argsort(feature_importance)[-15:]  # 取最重要的15个特征

        plt.barh(range(len(top_features)), feature_importance[top_features],
                 color='lightgreen', edgecolor='black', alpha=0.7)
        plt.title('Top 15 Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature Index', fontsize=12)
        plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_features])

        # 添加数值标签
        for i, v in enumerate(feature_importance[top_features]):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available for RBF kernel',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Feature Importance', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.show()


print("\n正在生成基础混淆矩阵热力图...")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


# 分类报告热力图
print("\n正在生成分类报告热力图...")
plot_classification_report(y_test, y_pred, np.unique(y))

# 归一化混淆矩阵
print("\n正在生成归一化混淆矩阵...")
plot_normalized_confusion_matrix(y_test, y_pred, np.unique(y))

# 每个类别的准确率
# print("\n正在生成类别准确率图...")
# plot_class_accuracy(y_test, y_pred, np.unique(y))

# ROC曲线
print("\n正在生成ROC曲线...")
plot_roc_curves(X_test, y_test, svm_classifier, np.unique(y))

# 学习曲线
# print("\n正在生成学习曲线...")
# plot_learning_curve(X_train, y_train, svm_classifier, np.unique(y))

# 精确率-召回率曲线
# print("\n正在生成精确率-召回率曲线...")
# plot_precision_recall_curve(X_test, y_test, svm_classifier, np.unique(y))

# 预测置信度分布
# print("\n正在生成预测置信度分布...")
# plot_prediction_confidence(X_test, svm_classifier)

# 特征重要性
# print("\n正在生成特征重要性图...")
# plot_feature_importance(svm_classifier, np.unique(y))

