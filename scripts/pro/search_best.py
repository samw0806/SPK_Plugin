import os
import pandas as pd

# 定义要遍历的文件夹路径
study = 'stad'
# base_dir = f'/home/ubuntu/disk1/wys/code/SurvPath/results_ori/results_{study}'
# base_dir = f'/home/ubuntu/disk1/wys/test/SurvPath/results/results_surv_plugin_{study}'
base_dir = '/home/ubuntu/disk1/wys/SurvPath/results/results_cat_mlp_with_p/results_add_global'

# 初始化变量以保存最大平均值及其对应的文件夹
max_avg_cindex = -float('inf')
max_avg_folder = None

# 遍历 base_dir 文件夹中的所有子文件夹
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        summary_file = os.path.join(folder_path, 'summary.csv')
        
        # 检查summary.csv文件是否存在
        if os.path.exists(summary_file):
            # 读取CSV文件
            df = pd.read_csv(summary_file)
            
            # 检查是否有val_cindex列
            if 'val_cindex' in df.columns:
                # 计算val_cindex列的平均值
                avg_cindex = df['val_cindex'].mean()
                
                # 如果当前文件夹的平均值大于记录的最大平均值，更新变量
                if avg_cindex > max_avg_cindex:
                    max_avg_cindex = avg_cindex
                    max_avg_folder = folder_name

# 打印结果
if max_avg_folder:
    print(f"文件夹: {max_avg_folder}, 平均val_cindex: {max_avg_cindex}")
else:
    print("未找到包含val_cindex列的summary.csv文件")
