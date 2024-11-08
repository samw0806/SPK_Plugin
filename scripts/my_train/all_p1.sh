
#!/bin/bash

DATA_ROOT_DIR="/home/ubuntu/disk1/wys/data/pathways" # where are the TCGA features stored?
BASE_DIR="/home/ubuntu/disk1/wys/SurvPath" # where is the repo cloned?
TYPE_OF_PATH="combine" # what type of pathways? 

MODELS=("omics") # what type of model do you want to train?
LRS=(0.0005)
STUDIES=("hnsc" "blca" "brca" "stad")
BATCH_SIZES=(1)
#务必记得确认checkpoint路径，学习率不一致所以文件夹名不一致
for MODEL in ${MODELS[@]};
do 
    for STUDY in ${STUDIES[@]};
    do
        BASE_PATH=/home/ubuntu/disk1/wys/data/survival/data/results_ori/${MODEL}/results_${STUDY}_${MODEL}/

        # 获取唯一的子文件夹名称
        subfolder_name=$(ls -d "$BASE_PATH"*/ | head -n 1 | xargs basename)

        # 将完整路径赋值给变量
        full_path="${BASE_PATH}${subfolder_name}" 
        for STUDY in ${STUDIES[@]};
        do
            CUDA_VISIBLE_DEVICES=1 python main.py \
                --load_model \
                --checkpoint_path $full_path \
                --study tcga_${STUDY} \
                --task survival \
                --split_dir splits \
                --which_splits 5foldcv \
                --type_of_path $TYPE_OF_PATH \
                --modality $MODEL \
                --data_root_dir $DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/ \
                --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
                --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
                --results_dir /home/ubuntu/disk1/wys/data/results/results_${MODEL}_with_p/results_add_cross_best \
                --batch_size 4 \
                --lr 0.005 \
                --global_act \
                --fusion_p "concat_linear" \
                --pk_path /home/ubuntu/disk1/wys/SPK_Plugin/datasets_csv/prior_knowledge/tcga_${STUDY}/knowledge_p4.csv \
                --opt adam \
                --reg 0.001 \
                --alpha_surv 0.5 \
                --weighted_sample \
                --max_epochs 30 \
                --label_col survival_months_dss \
                --k 5 \
                --bag_loss nll_surv \
                --n_classes 4 \
                --num_patches 4096 \
                --wsi_projection_dim 256

            wait
        done 
    done
done
