from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
import torch


# Configuration settings
max_seq_length = 2048  # Supports RoPE Scaling internally
dtype = None  # Auto-detects: Float16 for Tesla T4/V100, Bfloat16 for Ampere+
load_in_4bit = True  # Reduces memory usage with 4bit quantization

# Model files are cached in ~/.cache/huggingface/hub
# Initialize model and tokenizer with pretrained weights
model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name = "unsloth/Qwen2.5-3B-Instruct",  # Requires ~3GB VRAM
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# Apply PEFT (LoRA) for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # Suggested values: 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Data Preparation
alpaca_prompt = """You are a naughty girlfriend, your task is to answer boyfriend's questions.

### Question:
{}

### Answer:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Required for generation
def formatting_prompts_func(examples):  # Formats prompts for training
    questions = examples["instruction"]
    answers      = examples["output"]
    texts = []
    for question, answer in zip(questions, answers):
        # EOS_TOKEN prevents infinite generation
        text = alpaca_prompt.format(question, answer) + EOS_TOKEN
        texts.append(text)
    return { "text": texts }

# Load and preprocess dataset
# dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = load_dataset("csv", data_files="/home/huming/workspace/ai/finetune/sft/chat_dataset.csv", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Training setup
# Initialize SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs=5,
        # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# GPU memory statistics
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU: {gpu_stats.name}. Max memory: {max_memory} GB")
print(f"Reserved memory: {start_gpu_memory} GB")

# Start model training
trainer_stats = trainer.train()

# Inference
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
    [
        alpaca_prompt.format(
            "明天吃啥?",  # question
            "",  # answer (leave blank for generation)
        )
    ],
    return_tensors = "pt"
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# Save LoRA adapters
model.save_pretrained("qwen2.5-3B-chat")
tokenizer.save_pretrained("qwen2.5-3B-chat")
