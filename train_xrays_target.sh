
SRC_DOMAIN=$1
TGT_DOMAIN=$2
GPUID=$3
DATA_ROOT=${4:-'data'}

python train_target.py --data_dir ${DATA_ROOT} --dataset domainnet_style_datasets_Cardiomegaly/${TGT_DOMAIN} \
--source domainnet_style_datasets_Cardiomegaly_${SRC_DOMAIN}_source \
--run_name Cardiomegaly_${SRC_DOMAIN}2${TGT_DOMAIN} --batch_size 64 \
--num_class 2 --lr 0.001 --train_resampling natural --test_resampling natural \
--gpuid ${GPUID} --flag '' --num_epochs 1000 --num_neighbors 10 --ctr_weight 0.0 --div_weight 0.0  --wandb
