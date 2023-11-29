# ViT-B/16 F8: 

# Set the path to save checkpoints
OUTPUT_DIR='./output/test'
DATA_PATH='/bpfs/v2_mnt/VIS/wuwenhao/20bn-something-something-v2-frames'

MODEL_PATH='output/ssv2/base_f8_bs16_atm7/checkpoint-best/mp_rank_00_model_states.pt'

MASTER_ADDR='127.0.0.1'

NNODES=1
                                       
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 18899 --nnodes=${NNODES} --node_rank=0 --master_addr=${MASTER_ADDR} \
    test_for_frame.py \
    --model ViT-B/16 \
    --data_set SSV2 \
    --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 10 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 5 \
    --num_frames 8 \
    --embed_dim 512 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --resume ${MODEL_PATH}


