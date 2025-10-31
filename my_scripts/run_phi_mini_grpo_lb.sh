set -x

EXP_NAME="phi-moe-math-grpo-lb"
# EXP_NAME="Qwen2.5-3B-math-grpo"
WANDB_API_KEY=496fa07a6ccb07d43292fe86646aff9c1a362b35
WANDB_PROJECT=MoE-HW
NCCL_SHM_DISABLE=1 
NCCL_P2P_DISABLE=1

MODEL_PATH="/root/hf_cache/hub/models/microsoft/Phi-mini-MoE-instruct"
# MODEL_PATH="/root/hf_cache/hub/models/Qwen/Qwen2.5-3B-Instruct"

# TRAIN_PARQUET=${TRAIN_PARQUET:-/root/data/instruction_embedded_verl/train.parquet}
# VAL_PARQUET=${VAL_PARQUET:-/root/data/instruction_embedded_verl/val.parquet}

TRAIN_PARQUET="/root/data/orz_math_57k_instruction/train.parquet"
VAL_PARQUET="/root/data/orz_math_57k_instruction/dev.parquet"

SAVE_DIR=${SAVE_DIR:-/root/verl_ckpt/moe/${EXP_NAME}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${SAVE_DIR}/rollout_data}
VAL_ROLLOUT_DIR=${VAL_ROLLOUT_DIR:-${SAVE_DIR}/val_rollout_data}
SAVE_DIR=${SAVE_DIR}/checkpoints
mkdir -p "${SAVE_DIR}" "${ROLLOUT_DIR}" "${VAL_ROLLOUT_DIR}"
# export PHIMOE_FREEZE_ROUTER=1
export MOE_AUX_LOSS=1
export MOE_AUX_LOSS_COEF=0.001


# export RAY_DEBUG_POST_MORTEM=1 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_NAME  \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.default_local_dir=${SAVE_DIR} \
    trainer.rollout_data_dir=${ROLLOUT_DIR} \
    trainer.validation_data_dir=${VAL_ROLLOUT_DIR} \
    custom_reward_function.path=/workspace/verl/verl/utils/reward_score/orz_math_reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.strategy=fsdp2 \
    trainer.total_epochs=15 $@

# actor_rollout_ref.rollout.top_p=0.8 \
# actor_rollout_ref.rollout.temperature=1.0 \
# actor_rollout_ref.rollout.do_sample=True \


# could add
    # actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    # actor_rollout_ref.ref.fsdp_config.forward_prefetch=False \
    # actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
    # actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    # actor_rollout_ref.actor.use_torch_compile=False \
    # actor_rollout_ref.ref.use_torch_compile=False \
    # actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
