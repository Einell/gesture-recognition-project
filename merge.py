#合并目录下所有csv文件
import pandas as pd
import os
import glob

# 目录
csv_dir = 'gestures_csv'
# 输出文件名称
output_file = 'gestures.csv'
# 共用的表头
column_names = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']

def main():
    print("开始合并")

    # 查找同目录下所有CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    if not csv_files:
        print("错误：当前目录下未找到任何 CSV 文件！")
        return

    print(f"找到 {len(csv_files)}个CSV 文件待合并：")
    for file in csv_files:
        print(f" - {os.path.basename(file)}")

    #读取并合并所有CSV
    all_data = []
    for file_path in csv_files:
        try:
            # 读取单个 CSV，指定表头
            df = pd.read_csv(file_path, header=None if os.path.basename(file_path) == output_file else 0)
            # 新文件，手动添加表头
            if df.shape[1] == len(column_names) and not all(col in df.columns for col in column_names):
                df.columns = column_names
            df = df[column_names]
            all_data.append(df)
            print(f"成功读取：{os.path.basename(file_path)}（{len(df)} 行数据）")
        except Exception as e:
            print(f"警告：读取 {os.path.basename(file_path)} 失败 - {str(e)}，已跳过该文件")

    if not all_data:
        print("错误：没有读取到任何有效 CSV 数据！")
        return

    # 合并，去重
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df = merged_df.drop_duplicates()  # 去重
    print(f"\n合并完成：共 {len(merged_df)} 行有效数据（已去重）")

    # 保存文件
    merged_df.to_csv(output_file, index=False, header=True)
    print(f"合并后的文件已保存为：{output_file}")
    print(f"文件路径：{os.path.abspath(output_file)}")


if __name__ == '__main__':
    main()