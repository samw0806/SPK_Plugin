import pandas as pd
import os

def check_filename_slide_id(txt_file, csv_file):
    # 读取 .txt 文件并提取 filename 列
    txt_data = pd.read_csv(txt_file, delimiter='\t')  # 使用制表符分隔
    txt_filenames = set(txt_data['filename'])

    # 读取 .csv 文件并提取 slide_id 列
    csv_data = pd.read_csv(csv_file)
    csv_slide_ids = set(csv_data['slide_id'].astype(str))  # 确保为字符串格式以便比较

    # 找出 txt 文件中有但 csv 中没有的条目
    txt_only = txt_filenames - csv_slide_ids
    # 找出 csv 文件中有但 txt 文件中没有的条目
    csv_only = csv_slide_ids - txt_filenames

    # 打印结果
    if not txt_only and not csv_only:
        print("所有的 filename 和 slide_id 都匹配！")
    else:
        if txt_only:
            print("在 txt 文件中有但在 csv 文件中没有的条目:")
            print(txt_only)
        if csv_only:
            print("在 csv 文件中有但在 txt 文件中没有的条目:")
            print(csv_only)



def check_files_in_folder(folder_path, file_names):
    """
    检查指定文件夹中是否包含特定的文件。

    参数:
    - folder_path (str): 文件夹路径
    - file_names (list): 要检查的文件名列表

    返回:
    - dict: 每个文件名是否存在的结果，True表示存在，False表示不存在
    """
    # 创建一个字典来存储每个文件的检查结果
    file_existence = {}
    
    # 遍历文件名列表，检查每个文件是否在文件夹中
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        file_existence[file_name] = os.path.isfile(file_path)
    
    for file_name, exists in file_existence.items():
        print(f"{file_name}: {'存在' if exists else '不存在'}")


def find_missing_files(txt_file, folder_path):
    # 读取 .txt 文件内容
    df = pd.read_csv(txt_file, delimiter='\t')
    
    # 获取所有 filename 列中的文件名
    filenames = df['filename'].tolist()
    
    # 记录缺失的文件
    missing_files = []
    
    # 检查每个文件是否在指定文件夹中
    for filename in filenames:
        filename = filename.replace('svs',"jpg")
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
    
    # 输出缺失的文件
    if missing_files:
        print("文件夹中缺失以下文件：")
        for missing_file in missing_files:
            print(missing_file)
        print(len(missing_files))    
    else:
        print("所有文件都存在于文件夹中。")

def find_missing_slide_ids(csv_file, folder_path):
    """
    查找指定文件夹中缺失的slide_id文件。

    参数:
    csv_file (str): CSV文件的路径。
    folder_path (str): 要检查的文件夹路径。

    返回:
    list: 缺失的slide_id文件名列表。
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 获取slide_id列
    slide_ids = df['slide_id'].tolist()
    print(1)
    # 查找缺失的slide_id文件
    missing_files = []
    for slide_id in slide_ids:
        slide_id = slide_id.replace("svs","jpg")
        file_path = os.path.join(folder_path, slide_id)
        if not os.path.isfile(file_path):
            missing_files.append(slide_id)
            print(slide_id)

def filter_txt_by_csv(csv_file, txt_file, output_file,folder_path):
    """
    根据CSV文件中的slide_id过滤TXT文件中的行，保留匹配的行。

    参数:
    csv_file (str): CSV文件的路径。
    txt_file (str): TXT文件的路径。
    output_file (str): 输出文件的路径，将保存过滤后的数据。
    """
    # 读取CSV文件
    csv_df = pd.read_csv(csv_file)

        # 查找缺失的slide_id文件

    
    # 获取slide_id列表
    slide_ids = csv_df['slide_id'].tolist()
    missing_files = []
    for slide_id in slide_ids:
        slide_id = slide_id.replace("svs","jpg")
        file_path = os.path.join(folder_path, slide_id)
        if not os.path.isfile(file_path):
            missing_files.append(slide_id.replace("jpg","svs"))

    # 读取TXT文件
    txt_df = pd.read_csv(txt_file, sep='\t')  # 以制表符分隔读取

    # 筛选匹配的行
    filtered_txt_df = txt_df[txt_df['filename'].isin(missing_files)]

    # 将结果保存到输出文件
    filtered_txt_df.to_csv(output_file, sep='\t', index=False)
    print(f"已将匹配的行保存到 {output_file}。")


# 示例调用
dataset_name = "blca"
txt_file = f'/home/ubuntu/disk1/wys/data/tcga_svs/{dataset_name}/{dataset_name}.txt'
csv_file = f'/home/ubuntu/disk1/wys/SurvPath/datasets_csv/metadata/tcga_{dataset_name}.csv'
svs_fold = f'/home/quanyj/samw/datasets/TCGA/{dataset_name}_svs/{dataset_name}'
jpg_fold = f'/home/ubuntu/disk1/wys/data/survival/data/imgs/{dataset_name}_jpgs'
# check_filename_slide_id(txt_file, csv_file)
# check_files_in_folder("/home/quanyj/samw/datasets/TCGA/hnsc_svs/hnsc_jpgs",['TCGA-D6-6825-01Z-00-DX1.eb4d4223-48fe-40e0-901f-6a4146db9a1f.jpg','TCGA-P3-A6T6-01Z-00-DX1.4873798C-A2AD-40D5-B5B3-FCF75735C4F3.jpg'])

# filter_txt_by_csv(csv_file, '/home/ubuntu/disk1/wys/data/tcga_svs/blca/gdc_manifest.2024-10-26.txt', '/home/ubuntu/disk1/wys/data/tcga_svs/blca/blca_extend.txt',jpg_fold)
find_missing_slide_ids(csv_file,jpg_fold)