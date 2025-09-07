# Fine-Tuning with Ray Train and DeepSpeed
# @see https://docs.ray.io/en/latest/train/examples/deepspeed/gptj_deepspeed_fine_tuning.html

import deepspeed
import functools
import os
import torch
from torch.utils._pytree import tree_map
from peft import LoraConfig, get_peft_model

import ray
from ray import train
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer

from transformers.utils.logging import disable_progress_bar, enable_progress_bar
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed
)


# --- Example ChatML Dataset ---
# {"messages": [{"role": "system", "content": "You're a helpful assistant for answering questions!"}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi, how can I help you?"}]}
# {"messages": [{"role": "system", "content": "You're a helpful assistant for answering questions!"}, {"role": "user", "memontent": "Tell me a joke."}, {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}]}

# --- Configuration Section ---

USE_LORA = False

# Global config (can be loaded from a JSON or environment later)
CONFIG = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    # "model_name": "/home/huming/workspace/ai/finetune/output/sft_result_checkpoints/TorchTrainer_2025-09-07_23-43-24/TorchTrainer_63d7a_00000_0_2025-09-07_23-43-24/checkpoint_000004/checkpoint",
    "num_workers": 1,
    "block_size": 16000,
    "batch_size": 1,
    "train_path": "/home/huming/workspace/ai/finetune/misc/train_dataset.jsonl",
    "validation_path": "/home/huming/workspace/ai/finetune/misc/test_dataset.jsonl",
    "response_template": "<|im_start|>assistant",
    "seed": 42,
    "learning_rate": 1e-5,
    "num_train_epochs": 5,
    "checkpoint_to_keep": None,
    "deepspeed_config": {
        "fp16": {"enabled": "auto"},
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True
        },
        "steps_per_print": 1000,
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    },
    "lora_config": {
            "r": 512,
            "lora_alpha": 1024,
            "lora_dropout": 0.1,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            "task_type": "CAUSAL_LM",
            "modules_to_save": [],
            "bias": "none",
            "fan_in_fan_out": False,
            "init_lora_weights": True
    }
}

# --- Helper Functions ---

def get_storage_path() -> str:
    return "/home/huming/workspace/ai/finetune/output/sft_result_checkpoints"

def get_datasets(train_path, validation_path):
    train_ds = ray.data.read_json(train_path)
    eval_ds = ray.data.read_json(validation_path)
    datasets = {
        "train": train_ds,
        "validation": eval_ds
    }
    config = ray.train.DataConfig(
        datasets_to_split=["train", "validation"])
    return datasets, config

def collate_func(batch, tokenizer, block_size, device):
    batch_list = tokenizer.apply_chat_template(
            [y for x in batch["messages"] for y in x],
            chat_template = tokenizer.chat_template,
            tokenize=False)

    if isinstance(block_size, int) and block_size > tokenizer.model_max_length:
        max_length = tokenizer.model_max_length
    else:
        max_length = block_size

    out_batch = tokenizer(
        batch_list,
        padding=True,
        max_length=max_length,
        truncation='longest_first',
        add_special_tokens=True,
        return_tensors="pt",
    )

    out_batch["labels"] = out_batch["input_ids"].clone()
    out_batch = tree_map(lambda x: x.to(device), out_batch)
    return out_batch

# --- Training Function ---

def train_func(config: dict):
    # Use the actual number of CPUs assigned by Ray
    os.environ["OMP_NUM_THREADS"] = str(
        train.get_context().get_trial_resources().bundles[-1].get("CPU", 1)
    )

    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(config["seed"])

    print("Preparing training arguments")
    training_args = TrainingArguments(
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=5,
        metric_for_best_model="eval_loss",
        save_strategy="steps",
        save_steps=config["steps_per_epoch"],
        max_steps=config["steps_per_epoch"] * config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=1,
        learning_rate=config["learning_rate"],
        weight_decay=0.01,
        warmup_steps=0,
        label_names=["input_ids", "attention_mask"],
        push_to_hub=False,
        report_to="none",
        disable_tqdm=True,  # declutter the output a little
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        # deepspeed=config["deepspeed_config"],
        save_safetensors=True,
    )

    disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model files are cached in ~/.cache/huggingface/hub
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_cache=False,
        # attn_implementation="flash_attention_2",
        # device_map="auto"  # automatically places model on GPU if available
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if USE_LORA:
        # Apply LoRA to model
        model = get_peft_model(model, LoraConfig(**config["lora_config"]))

    print("Model loaded")

    enable_progress_bar()

    train_ds = train.get_dataset_shard("train")
    eval_ds = train.get_dataset_shard("validation")

    custom_collate_func = functools.partial(
        collate_func,
        tokenizer=tokenizer,
        block_size=config["block_size"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_ds_iterable = train_ds.iter_torch_batches(
        batch_size=config["batch_size"],
        local_shuffle_buffer_size=train.get_context().get_world_size() * config["batch_size"],
        collate_fn=custom_collate_func
    )
    eval_ds_iterable = eval_ds.iter_torch_batches(batch_size=config["batch_size"],
                                                  collate_fn=custom_collate_func
                                                  )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Add callback to report checkpoints to Ray Train
    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)
    trainer.train()

# --- Ray Main Function ---

def main():
    # Initialize Ray
    ray.init(log_to_driver=True, object_store_memory=4e9)

    datasets, dataset_config = get_datasets(CONFIG["train_path"], CONFIG["validation_path"])
    CONFIG["steps_per_epoch"] = (datasets["train"].count()) // (CONFIG["batch_size"] * CONFIG["num_workers"])

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=CONFIG,
        scaling_config=ScalingConfig(
            num_workers=CONFIG["num_workers"],
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 1},
        ),
        run_config=RunConfig(
            storage_path=get_storage_path(),
            checkpoint_config=CheckpointConfig(
                num_to_keep=CONFIG["checkpoint_to_keep"],
                checkpoint_score_attribute="perplexity",
                checkpoint_score_order="min",
            ),
        ),
        datasets=datasets,
        dataset_config=dataset_config,
    )

    result = trainer.fit()
    print(f"Results are stored at: {result.path}")


if __name__ == "__main__":
    main()
