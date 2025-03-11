unset CUDA_VISIBLE_DEVICES

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,2" 
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"


torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    --node_rank=0 \
    train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --data_path train.json \
    --eval_data_path eval.json \
    --bf16 True \
    --output_dir test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --medusa_num_heads 3 \
    --medusa_num_layers 1