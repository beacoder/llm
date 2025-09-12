import ray
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import json


# --- Configuration Constants ---
USE_LORA = True

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_PATH = "/home/huming/workspace/ai/finetune/output/sft_result_checkpoints/TorchTrainer_2025-09-10_22-15-13/TorchTrainer_915ed_00000_0_2025-09-10_22-15-13/checkpoint_000003/checkpoint"
VERIFICATION_PATH = "./verify_dataset.jsonl"
OUTPUT_PATH = "/home/huming/workspace/ai/finetune/output/merged_model"

START_TOKEN = "assistant\n"
STOP_TOKEN = "<|im_end|>"
SYSTEM_PROMPT = "You are a naughty girlfriend, your task is to answer boyfriend's questions."
SEPARATOR_LINE = "===================================================================================================================================================================================="

# --- Helper Functions ---
def save_merged_model(tokenizer, model, path: Path):
    path.mkdir(exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    model.config.save_pretrained(path)
    tokenizer.save_pretrained(path)

def load_model(tokenizer, model_path, lora_path=None, lora_enabled=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        use_cache=False,
        device_map= None if USE_LORA else "auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    if lora_enabled and lora_path:
        model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
        # save_merged_model(tokenizer, model, Path(OUTPUT_PATH))

    # Common pattern for inference
    model = model.eval().to(model.device)  # Set mode + ensure correct device
    return model

def build_chat_input(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

def format_output(lines):
    print('\n')
    for line in lines:
        print(line)
    print('\n')

def extract_last_ai_response(output_list):
    output = ''.join(output_list)
    start_index = len(START_TOKEN) + output.rfind(START_TOKEN)
    stop_index = output.rfind(STOP_TOKEN)
    return output[start_index:stop_index]

# --- Model Actor Definition ---
@ray.remote(num_gpus=2 if USE_LORA else 1)
class ModelActor:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.model = load_model(tokenizer, MODEL_PATH, CHECKPOINT_PATH, USE_LORA)
        self.device = self.model.device

    def predict(self, input_data, max_new_tokens=16000):
        inputs = build_chat_input(self.tokenizer,input_data)
        print("Inference input:\n", inputs)
        model_inputs = self.tokenizer([inputs], return_tensors="pt", add_special_tokens=False).to(self.device)

        with torch.no_grad():  # Disable gradient tracking
            outputs = self.model.generate(
                **model_inputs,
                temperature=0.01,
                max_new_tokens=max_new_tokens
            )

        return self.tokenizer.batch_decode(outputs)

# --- Inference Functions ---
def inference(actor, data):
    future = actor.predict.remote(data)
    last_ai_response = extract_last_ai_response(ray.get(future))
    print("Inference output:\n")
    format_output(last_ai_response.split("\n"))
    return last_ai_response

def do_inference(actor):
    with open(VERIFICATION_PATH, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            conversations = data["conversations"]
            chat_history = [conversations[0]]
            for msg in [x["content"] for x in conversations if x["role"] == "user"]:
                chat_history.append({"role": "user", "content": msg})
                chat_history.append({"role": "assistant", "content": inference(actor, chat_history)})
            print(SEPARATOR_LINE)

def do_interactive_inference(actor):
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user_input = input("\nYour input (quit/clear): ").strip()
        if user_input.lower() == "quit":
            print("Goodbye")
            break
        if user_input.lower() == "clear":
            chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Chat cleared")
            continue
        user_input = user_input.encode('utf-8', 'ignore').decode('utf-8')
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": inference(actor, chat_history)})


# --- Main Execution ---
if __name__ == "__main__":
    ray.init()
    actor = ModelActor.remote()
    # do_inference(actor)
    do_interactive_inference(actor)
    ray.shutdown()
