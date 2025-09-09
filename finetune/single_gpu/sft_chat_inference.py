from unsloth import FastLanguageModel
from transformers import TextStreamer
from typing import Tuple
import logging
from peft import PeftModel


# Configuration for Unsloth model
MAX_SEQ_LENGTH = 2048  # Auto supports RoPE Scaling
DTYPE = None  # Auto-detects dtype (Float16 for Tesla T4/V100, Bfloat16 for Ampere+)
LOAD_IN_4BIT = True  # Reduces memory usage with 4bit quantization

# Initialize base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-3B-Instruct",  # Requires ~3GB VRAM
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)
# Merge LoRA adapters
model = PeftModel.from_pretrained(model, "qwen2.5-3B-chat").merge_and_unload()
FastLanguageModel.for_inference(model)  # Enable faster inference

alpaca_prompt = """You are a naughty girlfriend, your task is to answer boyfriend's questions.

### Question:
{}

### Answer:
{}"""

# Inference
while True:
    user_input = input("Enter your input (type 'quit' to exit): ")

    if user_input.lower() == "quit":
        logging.info("Exiting inference loop.")
        print("Goodbye!")
        break

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                user_input,  # question
                "",  # answer (leave blank for generation)
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    logging.info("Generating response...")
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
