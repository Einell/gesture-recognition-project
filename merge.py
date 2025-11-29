# 这是用于合并指定目录下所有csv文件的程序
# 把文件放在q目录下，运行本程序即可
# 把所有手势文件合并成一个csv文件进行训练
import pandas as pd
import os
import glob

# 指定目录
csv_dir = "q"
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

    # 读取并合并所有CSV
    all_data = []
    for file_path in csv_files:
        try:
            # 读取单个CSV，跳过表头
            df = pd.read_csv(file_path, header=None)
            # 检查数据列数是否匹配
            if df.shape[1] != len(column_names):
                print(
                    f"警告：{os.path.basename(file_path)} 的列数不匹配（期望{len(column_names)}列，实际{df.shape[1]}列），已跳过该文件")
                continue

            # 设置表头
            df.columns = column_names
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