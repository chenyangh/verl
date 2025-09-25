set -x



nproc_per_node=4
save_path=$HOME/verl_ckpt/sft/phi_mini

# Shift the arguments so $@ refers to the rest
WANDB_PROJECT=MoE-HW
MODEL_PATH=${MODEL_PATH:-/root/hf_cache/hub/models/microsoft/Phi-mini-MoE-instruct}
# MODEL_PATH="/root/hf_cache/hub/models/Qwen/Qwen2.5-3B-Instruct"
EXP_NAME="phi-mini-sft"



# torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
#     -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=$HOME/data/gsm8k/train.parquet \
#     data.val_files=$HOME/data/gsm8k/test.parquet \
#     data.prompt_key=extra_info \
#     data.response_key=extra_info \
#     data.prompt_dict_keys=['question'] \
#     +data.response_dict_keys=['answer'] \
#     data.micro_batch_size_per_gpu=4 \
#     model.partial_pretrain=${MODEL_PATH} \
#     trainer.default_local_dir=$save_path \
#     trainer.project_name=gsm8k-sft \
#     trainer.experiment_name=${EXP_NAME}  \
#     trainer.total_epochs=4 \
#     trainer.project_name=$WANDB_PROJECT \
#     trainer.logger='["console","wandb"]' $@


    

export DEBUG_ALL_RANKS=0
export DEBUG_RANK=0
export DEBUG_WAIT=1
export BASE_PORT=5678

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
  my_scripts/torchrun_debug_wrap.py \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys=['question'] \
  +data.response_dict_keys=['answer'] \
  data.micro_batch_size_per_gpu=4 \
  model.partial_pretrain=${MODEL_PATH} \
  trainer.default_local_dir=$save_path \
  trainer.project_name=gsm8k-sft \
  trainer.experiment_name=${EXP_NAME}  \
  trainer.total_epochs=4 \
  trainer.project_name=$WANDB_PROJECT \
  trainer.logger='["console","wandb"]' "$@"
