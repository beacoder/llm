from unsloth import FastLanguageModel
from transformers import TextStreamer
from typing import Tuple
import logging


# Configuration for Unsloth model
MAX_SEQ_LENGTH = 2048  # Auto supports RoPE Scaling
DTYPE = None  # Auto-detects dtype (Float16 for Tesla T4/V100, Bfloat16 for Ampere+)
LOAD_IN_4BIT = True  # Reduces memory usage with 4bit quantization

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen2.5-3B-chat", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)  # Enable faster inference

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
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
            ALPACA_PROMPT.format(
                user_input,  # instruction
                "",  # input
                "",  # output (leave blank for generation)
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    logging.info("Generating response...")
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
