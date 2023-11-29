# Set the path to save checkpoints
OUTPUT_DIR='output/ssv1/base_f8'

DATA_PATH='/bpfs/v2_mnt/VIS/wuwenhao/20bn-something-something-v1'
# path to pretrain model
MODEL_PATH=' '

MASTER_ADDR='127.0.0.1'
NNODES=1

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 18892 --nnodes=${NNODES} --node_rank=0 --master_addr=${MASTER_ADDR} \
    run_class_finetuning.py \
    --model ViT-B/16 \
    --data_set SSV1 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 5 \
    --num_frames 8 \
    --embed_dim 512 \
    --opt adamw \
    --lr 7e-4 \
    --layer_decay 0.70 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 20 \
    --drop_path 0. \
    --drop 0. \
    --dist_eval \
    --enable_deepspeed \
    --aa True