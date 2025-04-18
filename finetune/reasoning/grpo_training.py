from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import GRPOConfig, GRPOTrainer
import re, torch


# Configuration settings
MAX_SEQ_LENGTH = 1024  # Can increase for longer reasoning traces
LORA_RANK = 64 # Larger rank = smarter, but slower

# Model files are cached in ~/.cache/huggingface/hub
# Initialize model and tokenizer with pretrained weights
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-3B-Instruct",  # Requires ~3GB VRAM
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
    # fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = LORA_RANK,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

# Apply PEFT (LoRA) for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_RANK,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# Data Preparation
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

class TrainerSetup:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.training_args = self._setup_training_args()

    def _setup_training_args(self):
        return GRPOConfig(
            # use_vllm = True, # use vLLM for fast inference!
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            logging_steps=1,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_generations=8,
            max_prompt_length=256,
            max_completion_length=200,
            max_steps=250,
            save_steps=250,
            max_grad_norm=0.1,
            report_to="none",
            output_dir="outputs",
        )

    def _get_gpu_stats(self):
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}. Max memory: {max_memory} GB")
        print(f"Reserved memory: {start_gpu_memory} GB")

    def train_model(self):
        self._get_gpu_stats()
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                xmlcount_reward_func,
                soft_format_reward_func,
                strict_format_reward_func,
                int_reward_func,
                correctness_reward_func,
            ],
            args=self.training_args,
            train_dataset=self.dataset,
        )
        trainer.train()
        return trainer

    def save_model(self, output_dir="qwen2.5-3B-reasoning"):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

# Initialize and execute training
trainer_setup = TrainerSetup(model, tokenizer, dataset)
trainer = trainer_setup.train_model()
trainer_setup.save_model()
