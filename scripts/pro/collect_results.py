# import os
# import pandas as pd

# # 定义基础路径
# base_path = '/home/ubuntu/disk1/wys/data/results_b2/'

# # 创建一个空的列表用于存储结果
# results = []

# # 遍历所有模型名称的文件夹
# for model_name in os.listdir(base_path):
#     model_path = os.path.join(base_path, model_name)
    
#     # 检查是否是目录
#     if os.path.isdir(model_path):
#         # 遍历每个模型目录下的研究类型文件夹
#         for research_type in os.listdir(model_path):
#             research_path = os.path.join(model_path, research_type)

#             # 检查是否是目录
#             if os.path.isdir(research_path):
#                 # 遍历研究类型文件夹中的所有文件夹
#                 for subfolder in os.listdir(research_path):
#                     subfolder_path = os.path.join(research_path, subfolder)
                    
#                     # 检查是否是目录，并且包含summary.csv文件
#                     if os.path.isdir(subfolder_path):
#                         summary_file = os.path.join(subfolder_path, 'summary.csv')
#                         if os.path.exists(summary_file):
#                             try:
#                                 # 读取summary.csv
#                                 df = pd.read_csv(summary_file)

#                                 # 检查val_cindex列是否存在
#                                 if 'val_cindex' in df.columns:
#                                     # 计算均值和方差
#                                     mean_val_cindex = df['val_cindex'].mean()
#                                     var_val_cindex = df['val_cindex'].var()

#                                     # 将结果添加到结果列表中
#                                     results.append({
#                                         'Model Name': model_name,
#                                         'Research Type': research_type,
#                                         'Mean val_cindex': mean_val_cindex,
#                                         'Variance val_cindex': var_val_cindex
#                                     })
#                             except Exception as e:
#                                 print(f"Error processing {summary_file}: {e}")

# # 将结果转换为DataFrame
# results_df = pd.DataFrame(results)

# # 将结果保存到CSV文件
# output_file = os.path.join(base_path, 'summary_results.csv')
# results_df.to_csv(output_file, index=False)

# print(f'Results saved to {output_file}')

import os
import pandas as pd

def calculate_val_cindex(folder_path):
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 只处理文件名为 summary.csv 的文件
            if file == 'summary.csv':
                file_path = os.path.join(root, file)
                
                # 读取 csv 文件
                df = pd.read_csv(file_path)
                
                # 检查是否有 val_cindex 列
                if 'val_cindex' in df.columns:
                    # 计算均值和方差并四舍五入到小数点后三位
                    mean_val_cindex = round(df['val_cindex'].mean(), 3)
                    var_val_cindex = round(df['val_cindex'].var(), 3)
                    
                    # 输出文件夹名称及计算结果
                    folder_name = os.path.basename(root)
                    print(f"文件夹: {folder_name}, 均值: {mean_val_cindex}, 方差: {var_val_cindex}")

# 使用时替换为您文件夹的路径
calculate_val_cindex('/home/ubuntu/disk1/wys/data/results_b2/results_transmil_wsi_pathways_with_p/results_add_cross_best')
