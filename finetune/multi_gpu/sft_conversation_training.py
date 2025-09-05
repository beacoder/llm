# Fine-Tuning with Ray Train and DeepSpeed
# @see https://docs.ray.io/en/latest/train/examples/deepspeed/gptj_deepspeed_fine_tuning.html

import functools
import os
import re
import torch

import ray
from ray import train
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.train.torch import TorchTrainer

import deepspeed
from peft import LoraConfig, get_peft_model
# import evaluate
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
    "model_name": "Qwen2.5-Coder-3B",
    "num_workers": 2,
    "block_size": 512,
    "batch_size": 1,
    "num_checkpoints_to_keep": 6,
    "train_path": "train_dataset.jsonl",
    "validation_path": "test_dataset.jsonl",
    "response_template": "<|im_start|>assistant",
    "max_seq_length": 16000,
    "seed": 42,
    "learning_rate": 1e-5,
    "num_train_epochs": 6,
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
    return f"./output/sft_result_checkpoints"

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
            chat_template = QWEN2_32B_TEMPLTE,
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
    out_batch = tree.map_structure(lambda x: x.to(device), out_batch)
    return out_batch

# --- Training Function ---

def train_func(config: dict):
    # Use the actual number of CPUs assigned by Ray
    # os.environ["OMP_NUM_THREADS"] = str(
    #     train.get_context().get_trial_resources().bundles[-1].get("CPU", 1)
    # )

    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(config["seed"])

    print("Preparing training arguments")
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=config["steps_per_epoch"],
        max_steps=config["steps_per_epoch"] * config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
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
        deepspeed=config["deepspeed_config"],
        save_safetensors=True,
    )

    disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_cache=False,
        attn_implementation="flash_attention_2",
        # device_map="auto"  # automatically places model on GPU if available
    )
    model.resize_token_embeddings(len(tokenizer))

    if USE_LORA:
        # Apply LoRA to model
        model = get_peft_model(model, LoraConfig(**config["lora_config"]))

    print("Model loaded")

    enable_progress_bar()

    # metric = evaluate.load("accuracy")

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
        collate_func=custom_collate_func)
    eval_ds_iterable = eval_ds.iter_torch_batches(batch_size=config["batch_size"],
                                                  collate_func=custom_collate_func)

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        # compute_metrics=compute_metrics,
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
    ray.init(log_to_driver=True)

    datasets, dataset_config = get_datasets(CONFIG["train_path"], CONFIG["validation_path"])
    CONFIG["output_dir"] = get_storage_path()
    CONFIG["steps_per_epoch"] = (datasets["train"].count()) // (CONFIG["batch_size"] * CONFIG["num_workers"])

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=CONFIG,
        scaling_config=ScalingConfig(
            num_workers=CONFIG["num_workers"],
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 6},
            trainer_resources={"CPU": 0}
        ),
        run_config=RunConfig(
            storage_path=get_storage_path(),
            checkpoint_config=CheckpointConfig(
                num_to_keep=None,
                checkpoint_score_attribute="perplexity",
                checkpoint_score_order="min",
            ),
        ),
        datasets=datasets,
        dataset_config=dataset_config,
    )

    result = trainer.fit()
    best_checkpoint, best_checkpoint_metrics = result.best_checkpoints[-1]

    print(f"Results are stored at: {result.path}")
    print(f"Best checkpoint is stored at: {best_checkpoint}, with perplexity: {best_checkpoint_metrics['perplexity']}")


if __name__ == "__main__":
    main()
