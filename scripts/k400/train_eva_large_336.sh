# Set the path to save checkpoints
OUTPUT_DIR='output/k400/eva_large_336_f8'

DATA_PATH='/bpfs/v2_mnt/VIS/test/k400/train_320_frames'
# path to pretrain model
MODEL_PATH=' '

MASTER_ADDR='127.0.0.1'
NNODES=1

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 18892 --nnodes=${NNODES} --node_rank=0 --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model EVA02-CLIP-L-14-336 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --input_size 336 \
    --short_side_size 336 \
    --save_ckpt_freq 5 \
    --num_frames 8 \
    --embed_dim 768 \
    --opt adamw \
    --lr 2e-4 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 20 \
    --drop_path 0. \
    --drop 0. \
    --dist_eval \
    --enable_deepspeed