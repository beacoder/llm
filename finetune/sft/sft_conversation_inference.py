import logging

from unsloth import FastLanguageModel
from transformers import TextStreamer


# Configuration for Unsloth model
MAX_SEQ_LENGTH = 2048  # Auto supports RoPE Scaling
DTYPE = None  # Auto-detects dtype (Float16 for Tesla T4/V100, Bfloat16 for Ampere+)
LOAD_IN_4BIT = True  # Reduces memory usage with 4bit quantization

# Initialize model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen2.5-3B-conversation",
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# conversation
messages = [
    {"role": "user", "content": "where is france ?"},
    {"role": "assistant", "content": "france is located in france."},
    {"role": "user", "content": "does it have any sightseeing?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
logging.info("Generating response...")
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
