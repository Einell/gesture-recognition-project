# 这是用于合并指定目录下所有csv文件的程序（LSTM动态手势）
# 把文件放在指定目录下，运行本程序即可
# 把所有手势文件合并成一个csv文件进行训练
import pandas as pd
import os
import glob
import numpy as np

# 配置
# 指定目录
CSV_DIR = 'lstm-3'
# 合并后的输出文件名称
OUTPUT_FILE = 'gestures_lstm-3.csv'
# 动态手势序列长度
SEQUENCE_LENGTH = 20
# 每帧的特征数量
FEATURES_PER_FRAME = 126

TOTAL_FEATURES = SEQUENCE_LENGTH * FEATURES_PER_FRAME  # 总长度

# 生成列名
def generate_column_names(total_features):
    column_names = [f'feature_{i}' for i in range(total_features)]
    column_names.append('label')
    return column_names

def main():
    column_names = generate_column_names(TOTAL_FEATURES)
    EXPECTED_COLUMNS = len(column_names)
    # 确保目录存在
    if not os.path.exists(CSV_DIR):
        print(f"错误：未找到特征目录 '{CSV_DIR}'。请创建此目录并将CSV文件放入其中。")
        return

    # 查找所有CSV文件
    search_pattern = os.path.join(CSV_DIR, '*.csv')
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"错误：在目录 '{CSV_DIR}' 下未找到任何 CSV 文件！")
        return

    print(f"\n找到 {len(csv_files)} 个 CSV 文件待合并...")

    # 读取并合并所有CSV
    all_data = []
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            # 读取单个 CSV
            df = pd.read_csv(file_path, header=None)
            # 检查列数是否符合预期
            if df.shape[1] != EXPECTED_COLUMNS:
                print(f"警告：文件 '{file_name}' 列数不匹配 ({df.shape[1]} != {EXPECTED_COLUMNS})，已跳过。")
                continue
            # 重新设置列名
            df.columns = column_names
            all_data.append(df)
            print(f" - 成功读取：{file_name} ({len(df)} 行数据)")

        except Exception as e:
            print(f"警告：读取 {file_name} 失败 - {str(e)}，已跳过。")

    if not all_data:
        print("\n没有有效数据可以合并。")
        return

    # 合并所有 DataFrame
    merged_df = pd.concat(all_data, ignore_index=True)

    # 写入最终的CSV文件
    output_path = os.path.join(OUTPUT_FILE)
    merged_df.to_csv(output_path, index=False)  # 写入时包含表头 (header=True)

    print(f"合并完成！")
    print(f"总样本数: {len(merged_df)} 行")
    print(f"文件保存至: {output_path}")

if __name__ == '__main__':
    main()