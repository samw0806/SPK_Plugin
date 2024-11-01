#!/bin/bash

DATA_ROOT_DIR="/home/ubuntu/disk1/wys/data/pathways"  # TCGA 特征存储路径
BASE_DIR="/home/ubuntu/disk1/wys/SurvPath"  # repo 克隆位置
TYPE_OF_PATH="combine"  # 路径类型

MODEL="omics"  # 模型类型
STUDY="stad"  # 研究类型

# 使用 GPU 0 训练模型，使用 "add" 融合方式
CUDA_VISIBLE_DEVICES=0 python main.py \
    --load_model \
    --checkpoint_path /home/ubuntu/disk1/wys/data/survival/data/results_ori/${MODEL}/results_${STUDY}_${MODEL}/tcga_${STUDY}__nll_surv_a0.5_lr5e-04_l2Weight_0.0001_5foldcv_b1_survival_months_dss_dim1_768_patches_4096_wsiDim_256_epochs_5_fusion_None_modality_omics_pathT_combine \
    --study tcga_${STUDY} \
    --task survival \
    --split_dir splits \
    --which_splits 5foldcv \
    --type_of_path $TYPE_OF_PATH \
    --modality $MODEL \
    --data_root_dir $DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/ \
    --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
    --fusion "add" \
    --results_dir "results_cat_mlp_with_p/results_add" \
    --batch_size 1 \
    --lr 0.0005 \
    --global_act \
    --opt radam \
    --reg 0.0001 \
    --alpha_surv 0.5 \
    --weighted_sample \
    --max_epochs 20 \
    --label_col survival_months_dss \
    --k 5 \
    --bag_loss nll_surv \
    --n_classes 4 \
    --num_patches 4096 \
    --wsi_projection_dim 256 
#     &



# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --load_model \
#     --checkpoint_path /home/ubuntu/disk1/wys/data/survival/data/results_ori/${MODEL}/results_${STUDY}_${MODEL}/tcga_${STUDY}__nll_surv_a0.5_lr5e-04_l2Weight_0.0001_5foldcv_b1_survival_months_dss_dim1_768_patches_4096_wsiDim_256_epochs_5_fusion_None_modality_omics_pathT_combine \
#     --study tcga_${STUDY} \
#     --task survival \
#     --split_dir splits \
#     --which_splits 5foldcv \
#     --type_of_path $TYPE_OF_PATH \
#     --modality $MODEL \
#     --data_root_dir $DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/ \
#     --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
#     --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} \
#     --fusion "add" \
#     --results_dir "results_cat_mlp_with_p/results_add_global" \
#     --batch_size 1 \
#     --global_act \
#     --lr 0.0005 \
#     --opt radam \
#     --reg 0.0001 \
#     --alpha_surv 0.5 \
#     --weighted_sample \
#     --max_epochs 20 \
#     --label_col survival_months_dss \
#     --k 5 \
#     --bag_loss nll_surv \
#     --n_classes 4 \
#     --num_patches 4096 \
#     --wsi_projection_dim 256  &

# wait
