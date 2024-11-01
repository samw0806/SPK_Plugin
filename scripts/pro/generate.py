#----> pytorch imports
import torch
from tqdm import tqdm 
#----> general imports
import pandas as pd
import numpy as np
import argparse
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment
from models.ourmodel import Qllava_pro

from utils.process_args import _process_args

def main(args):
    img_path=f"/home/ubuntu/disk1/wys/data/survival/data/imgs/{args.study}_jpgs"
    save_fold = f"/home/quanyj/samw/SurvPath/datasets_csv/prior_knowledge/tcga_{args.study}/"
    prompt_base="Considering the clinical information provided, could you give a concise description of the histopathology image shown?"
    csv_file = f'/home/ubuntu/disk1/wys/SurvPath/datasets_csv/metadata/tcga_{args.study}.csv'
    cli_file = f'/home/ubuntu/disk1/wys/SurvPath/datasets_csv/clinical_mine/{args.study}/clinical.tsv'
    
    csv_data = pd.read_csv(csv_file)
    csv_slide_ids = csv_data['slide_id'].tolist()
    model = Qllava_pro(
                 model_path='wisdomik/Quilt-Llava-v1.5-7b',
                 img_path=img_path,
                 patch_count=args.patch_count,
                 prompt_base=prompt_base
                 )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = pd.read_csv(cli_file, sep='\t')
    data_all = []
    for file_name in tqdm(os.listdir(img_path), desc="Processing files"):
        prefix = file_name[:12]
        # 检查文件是否以.jpg结尾
        if file_name.lower().endswith(".jpg"):
            matched_row = df[df['case_submitter_id'].str.startswith(prefix)]
            age_at_index = matched_row['age_at_index'].values[0]
            gender = matched_row['gender'].values[0]
            race = matched_row['race'].values[0]
            # clinical_prompt = f"The patient is at stage {clinical_info[0]}, grade {clinical_info[1]}, and has a cancer type of {clinical_info[2]}."
            clinical_prompt = f"The patient is {gender},{age_at_index} years old,race is {race}"
            input = (file_name,clinical_prompt)
            ans_list = model(input)
            data = {"Label": file_name, **{f"Ans_{i+1}": [val] for i, val in enumerate(ans_list)}}
            data_all.append(data)

    df = pd.DataFrame(data_all)
    df.to_csv(os.path.join(save_fold, f"knowledge_p{args.patch_count}.csv"), index=False)



if __name__ == "__main__":
    start = timer()

    #----> read the args
    parser = argparse.ArgumentParser(description='Configurations for SurvPath Survival Prediction Training')

    parser.add_argument('--study', type=str, help='study name')
    parser.add_argument('--patch_count', type=int)

    args = parser.parse_args()

    
    #----> create dataset factory


    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))