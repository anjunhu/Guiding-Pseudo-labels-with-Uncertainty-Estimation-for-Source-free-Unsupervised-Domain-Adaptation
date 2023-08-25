
SRC_DOMAIN=$1
DATA_ROOT=${2:-'data'}

python train_source.py --data_dir ${DATA_ROOT} --num_class 2 --batch_size 128 \
--dataset domainnet_style_datasets_Cardiomegaly/${SRC_DOMAIN} --run_name domainnet_style_datasets_Cardiomegaly_${SRC_DOMAIN}_source \
--gpuid 2 --lr 0.001 --train_resampling natural --test_resampling natural --wandb
