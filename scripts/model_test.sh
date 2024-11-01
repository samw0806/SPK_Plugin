#!/bin/bash

DATA_ROOT_DIR="/home/ubuntu/disk1/wys/data/pathways" # where are the TCGA features stored?
BASE_DIR="/home/ubuntu/disk1/wys/test/SurvPath" # where is the repo cloned?
TYPE_OF_PATH="combine" # what type of pathways? 
MODEL="survpath_mine" # what type of model do you want to train?
DIM1=8
DIM2=16
STUDIES=("stad")
LRS=(0.00005 0.0001 0.0005 0.001)
DECAYS=(0.00001 0.0001 0.001 0.01)

for decay in ${DECAYS[@]};
do
    for lr in ${LRS[@]};
    do 
        for STUDY in ${STUDIES[@]};
        do
            CUDA_VISIBLE_DEVICES=1 python main.py \
                --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
                --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/ --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
                --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir results_test \
                --batch_size 1 --lr $lr --opt radam --reg $decay \
                --alpha_surv 0.5 --weighted_sample --max_epochs 2 --encoding_dim 1024 \
                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 \
                --encoding_layer_1_dim ${DIM1} --encoding_layer_2_dim ${DIM2} --encoder_dropout 0.25 --load_model --checkpoint_path /home/ubuntu/disk1/wys/code/SurvPath/results_ori/results_hnsc/tcga_hnsc__nll_surv_a0.5_lr5e-05_l2Weight_1e-05_5foldcv_b1_survival_months_dss_dim1_1024_patches_4096_wsiDim_256_epochs_2_fusion_None_modality_survpath_pathT_combine
        done 
    done
done 