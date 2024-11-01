import pandas as pd

study = "stad"
# 读取 CSV 文件
tcga_stad_df = pd.read_csv(f'/home/ubuntu/disk1/wys/SurvPath/datasets_csv/metadata/tcga_{study}.csv')
knowledge_df = pd.read_csv(f'/home/ubuntu/disk1/wys/SurvPath/datasets_csv/prior_knowledge/{study}/knowledge_p4.csv')
knowledge_df['Label'] = knowledge_df['Label'].str.replace('.jpg', '.svs', regex=False)
# 定义需要处理的列
cols_to_clean = ['Ans_1', 'Ans_2', 'Ans_3', 'Ans_4', 'Ans_5']

# 去掉指定字符
for col in cols_to_clean:
    knowledge_df[col] = knowledge_df[col].str.replace('"', '', regex=False)
    knowledge_df[col] = knowledge_df[col].str.replace('[', '', regex=False)
    knowledge_df[col] = knowledge_df[col].str.replace("'", '', regex=False)
    knowledge_df[col] = knowledge_df[col].str.replace(']', '', regex=False)

# 检查 Label 和 slide_id 的唯一值
print("知识库中的标签:", knowledge_df['Label'].unique())
print("TCGA 数据集中的 slide_id:", tcga_stad_df['slide_id'].unique())

# 合并数据
merged_df = pd.merge(tcga_stad_df, knowledge_df, how='left', left_on='slide_id', right_on='Label')

# 打印合并后的数据框以检查是否匹配
print("合并后的数据框：")
print(merged_df)

# 选择需要的列
result_df = merged_df[tcga_stad_df.columns.tolist() + ['Ans_1', 'Ans_2', 'Ans_3', 'Ans_4', 'Ans_5']]


print(f"合并结果已保存至 'merged_tcga_{study}.csv'")


# 保存结果到新的 CSV 文件
result_df.to_csv(f'/home/ubuntu/disk1/wys/SurvPath/datasets_csv/metadata/merged_knowledge_{study}.csv', index=False)
