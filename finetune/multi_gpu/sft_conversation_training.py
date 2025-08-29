# --- Train ChatML dataset with SFTTrainer, ray and deepspeed ---

import json
import os
import argparse
import torch

import ray
from ray import train
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


# --- Example ChatML Dataset ---
# {"messages": [{"role": "system", "content": "You're a helpful assistant for answering questions!"}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi, how can I help you?"}]}
# {"messages": [{"role": "system", "content": "You're a helpful assistant for answering questions!"}, {"role": "user", "memontent": "Tell me a joke."}, {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"}]}

# --- Configuration Section ---

# Global config (can be loaded from a JSON or environment later)
CONFIG = {
    "model_name": "Qwen2.5-Coder-3B",
    "num_workers": 2,
    "block_size": 512,
    "batch_size": 1,
    "num_checkpoints_to_keep": 6,
    "train_path": "train_dataset.jsonl",
    "test_path": "test_dataset.jsonl",
    "response_template": "<|assistant|>",
    "max_seq_length": 16000,
    "seed": 42,
    "learning_rate": 1e-5,
    "num_train_epochs": 6,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
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
    }
}

# --- Helper Functions ---

def get_storage_path() -> str:
    return f"./output/sft_result_checkpoints"

def load_and_tokenize_data(config: dict):
    dataset = load_dataset("json", data_files=config["train_path"], split="train")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def format_chatml_prompt(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
        }

    dataset = dataset.map(format_chatml_prompt, remove_columns=dataset.column_names)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length, padding=False)

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"Peek dataset: {dataset[0]}")
    return dataset, tokenizer

def create_training_args(config: dict, local_rank: int, world_size: int):
    output_dir = config["output_dir"]
    num_train_epochs = config["num_train_epochs"]
    learning_rate = config["learning_rate"]

    # Calculate batch size
    per_device_train_batch_size = config["per_device_train_batch_size"]
    grad_acc_steps = config["gradient_accumulation_steps"]
    total_batch_size = per_device_train_batch_size * grad_acc_steps * world_size

    print(f"Global batch size: {total_batch_size}, LR: {learning_rate}, Epochs: {num_train_epochs}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="no",
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
        deepspeed=config["deepspeed_config"],
    )

    return training_args

# --- Training Function ---

def train_func(config: dict):
    set_seed(config["seed"])

    ctx = train.get_context()
    local_rank = ctx.get_local_rank()
    world_size = ctx.get_world_size()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset, _ = load_and_tokenize_data(config)

    # Set up data collator to compute loss only on assistant responses
    collator = DataCollatorForCompletionOnlyLM(config["response_template"], tokenizer=tokenizer)

    training_args = create_training_args(config, local_rank, world_size)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",  # Not used since we pre-tokenized, but required
        max_seq_length=config["max_seq_length"],
        packing=False,  # Disable packing since we use completion-only loss
    )

    # Train and save
    trainer.train()

    # Save model only on one process
    if ctx.get_world_rank() == 0:
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])


# --- Ray Main Function ---

def main():
    # Initialize Ray
    ray.init(log_to_driver=True)

    # Run config
    run_config = RunConfig(
        storage_path=get_storage_path(),
        checkpoint_config=CheckpointConfig(
            num_to_keep=None,
            checkpoint_score_attribute="perplexity",
            checkpoint_score_order="min",
        ),
    )

    CONFIG["output_dir"] = get_storage_path()

    trainer = TorchTrainer(
        train_func,
        train_loop_config=CONFIG,
        scaling_config=ScalingConfig(
            num_workers=CONFIG["num_workers"],
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 2},
            trainer_resources={"CPU": 0}
        ),
        run_config=run_config,
    )

    result = trainer.fit()
    print(f"Training completed. Output saved to: {result.checkpoint}")


if __name__ == "__main__":
    main()
