export CUDA_VISIBLE_DEVICES=2,3
export NPROC_PER_NODE=2
export MASTER_PORT=12345

BASE_MODEL="/home/yifengx4/DeepAnalyze/models/DeepSeek-R1-0528-Qwen3-8B"
MODEL_SINGLE_ABILITY_PATH="/home/yifengx4/DeepAnalyze/models/DeepSeek-R1-0528-Qwen3-8B-sft"
DATA_DIR="/home/yifengx4/DeepAnalyze/data/DataScience-Instruct-500K"

# Make sure you are in directory ./deepanalyze/ms-swift/
swift sft \
    --model "${BASE_MODEL}" \
    --train_type "full" \
    --dataset \
        "${DATA_DIR}/reasoning/SKGInstruct_199989.json" \
        "${DATA_DIR}/reasoning/TableQA_distillation_39301.json" \
        "${DATA_DIR}/reasoning/TableQA_refinement_39301.json" \
        "${DATA_DIR}/reasoning/TableGPT_29448.json" \
        "${DATA_DIR}/reasoning/file_database_3833.json" \
        "${DATA_DIR}/reasoning/file_csv_3007.json" \
        "${DATA_DIR}/reasoning/file_xlsx_3663.json" \
        "${DATA_DIR}/reasoning/file_any_2520.json" \
        "${DATA_DIR}/reasoning/math_20000.json" \
        "${DATA_DIR}/reasoning/code_20000.json" \
        "${DATA_DIR}/reasoning/science_20000.json" \
        "${DATA_DIR}/reasoning/instruction_following_20000.json" \
        "${DATA_DIR}/reasoning/other_19998.json" \
    --torch_dtype "bfloat16" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 32 \
    --packing true \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 1 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --lazy_tokenize true \
    --save_total_limit 1 \
    --response_prefix "" \
    --save_only_model false \
    --gradient_checkpointing true \
    --output_dir "${MODEL_SINGLE_ABILITY_PATH}" \
    --deepspeed "zero3_offload" \
    --use_liger_kernel true \
    --attn_impl "flash_attn" \
    --model_type "deepseek_r1_distill"