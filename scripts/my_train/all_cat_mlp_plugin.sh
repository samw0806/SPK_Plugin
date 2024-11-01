
#!/bin/bash

DATA_ROOT_DIR="/home/ubuntu/disk1/wys/data/pathways" # where are the TCGA features stored?
BASE_DIR="/home/ubuntu/disk1/wys/SurvPath" # where is the repo cloned?
TYPE_OF_PATH="combine" # what type of pathways? 

MODEL="omics" # what type of model do you want to train?
LRS=(0.0005)
STUDIES=("stad" "hnsc")
BATCH_SIZES=(32 1)
#务必记得确认checkpoint路径，学习率不一致所以文件夹名不一致
for bs in ${BATCH_SIZES[@]};
do 
    for lr in ${LRS[@]};
    do 
        for STUDY in ${STUDIES[@]};
        do
            CUDA_VISIBLE_DEVICES=0 python main.py \
                --load_model --checkpoint_path /home/ubuntu/disk1/wys/data/survival/data/results_ori/${MODEL}/results_${STUDY}_${MODEL}/tcga_${STUDY}__nll_surv_a0.5_lr5e-04_l2Weight_0.0001_5foldcv_b1_survival_months_dss_dim1_768_patches_4096_wsiDim_256_epochs_5_fusion_None_modality_omics_pathT_combine --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
                --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/ --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
                --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --fusion "add" --results_dir "results_cat/results_add_b$bs" \
                --batch_size 1 --batch_sizes $bs --lr $lr --opt radam --reg 0.0001 \
                --alpha_surv 0.5 --weighted_sample --max_epochs 20 \
                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 &

            CUDA_VISIBLE_DEVICES=0 python main.py \
                --load_model --checkpoint_path /home/ubuntu/disk1/wys/data/survival/data/results_ori/${MODEL}/results_${STUDY}_${MODEL}/tcga_${STUDY}__nll_surv_a0.5_lr5e-04_l2Weight_0.0001_5foldcv_b1_survival_months_dss_dim1_768_patches_4096_wsiDim_256_epochs_5_fusion_None_modality_omics_pathT_combine --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
                --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/ --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
                --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --fusion "multiply" --results_dir "results_cat/results_multiply_b$bs" \
                --batch_size 1 --batch_sizes $bs --lr $lr --opt radam --reg 0.0001 \
                --alpha_surv 0.5 --weighted_sample --max_epochs 20 \
                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 &

            CUDA_VISIBLE_DEVICES=1 python main.py \
                --load_model --checkpoint_path /home/ubuntu/disk1/wys/data/survival/data/results_ori/${MODEL}/results_${STUDY}_${MODEL}/tcga_${STUDY}__nll_surv_a0.5_lr5e-04_l2Weight_0.0001_5foldcv_b1_survival_months_dss_dim1_768_patches_4096_wsiDim_256_epochs_5_fusion_None_modality_omics_pathT_combine --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
                --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/$STUDY/tiles-l1-s224/feats-l1-s224-CTransPath-sampler/pt_files/ --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
                --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --fusion "concat_linear" --results_dir "results_cat/results_concat_linear_b$bs" \
                --batch_size 1 --batch_sizes $bs --lr $lr --opt radam --reg 0.0001 \
                --alpha_surv 0.5 --weighted_sample --max_epochs 20 \
                --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 &

            wait
        done 
    done
done
