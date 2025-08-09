export REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Get the directory containing the script, then go to VLM-R1
echo "REPO_HOME: $REPO_HOME"
export PROJECT_ROOT="$(dirname "$REPO_HOME")" # Get TTRL4GUI directory
echo "PROJECT_ROOT: $PROJECT_ROOT"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TRITON_CACHE_DIR=/tmp/triton_cache

# on remote
data_paths="${PROJECT_ROOT}/data/screenspot_v2/screenspot_v2.jsonl" # change to your own data path
image_folders="${PROJECT_ROOT}/data/screenspot_v2/images" # change to your own image path
model_path="Qwen/Qwen2.5-VL-3B-Instruct" # change to your own model path
is_reward_customized_from_vlm_module=False
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="TTRL-GUI-Grounding-screenspot_v2-Qwen2.5-VL-3B-bs64" # TODO: change this to your own experiment name
TASK_TYPE="gui"
cd ${REPO_HOME}/src/open-r1-multimodal/src

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

# tensorboard
export TENSORBOARD_LOG_DIR="${REPO_HOME}/runs/${EXP_NAME}/tensorboard_logs/debug_log.$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p $TENSORBOARD_LOG_DIR
# export WANDB_DISABLED=true
# export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export WANDB_RESUME="allow"
# export NCCL_DEBUG=INFO
# export CUDA_VISIBLE_DEVICES=2,3,4,5 # TODO: change this to your own GPUs

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
  -m open_r1.grpo_jsonl \
    --use_vllm True \
    --output_dir ${PROJECT_ROOT}/checkpoints/ttrl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --max_steps 40 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --save_steps 10 \
    --num_generations 16 \
    --max_completion_length 128 \
    --temperature 0.7 \
    --reward_funcs voting \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --logging_dir $TENSORBOARD_LOG_DIR \
    --point_expand_size 50 \

echo "Training completed for ${EXP_NAME}"
