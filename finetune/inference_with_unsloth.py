from unsloth import FastLanguageModel
from transformers import TextStreamer
from typing import Tuple
import logging


# Configuration for Unsloth model
MAX_SEQ_LENGTH = 2048 # Choose any! We auto support RoPE Scaling internally!
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True # Use 4bit quantization to reduce memory usage. Can be False. # Choose any! We auto support RoPE Scaling internally!

def load_model_and_tokenizer(model_name: str) -> Tuple[FastLanguageModel, any]:
    """
    Load the model and tokenizer.
    :param model_name: Name of the pre-trained model.
    :return: Tuple of model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)  # Enable faster inference
    return model, tokenizer

# Initialize model and tokenizer
model, tokenizer = load_model_and_tokenizer("qwen2.5-3B-chat")

# Inference setup
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def run_inference() -> None:
    """Run inference loop for user input."""
    while True:
        user_input = input("Enter your input (type 'quit' to exit): ")

        if user_input.lower() == "quit":
            logging.info("Exiting inference loop.")
            print("Goodbye!")
            break

        inputs = tokenizer(
            [
                ALPACA_PROMPT.format(
                    f"{user_input}",  # instruction
                    "",  # input
                    "",  # output - leave this blank for generation!
                )
            ], return_tensors="pt").to("cuda")

        text_streamer = TextStreamer(tokenizer)
        logging.info("Generating response...")
        _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)


if __name__ == "__main__":
    run_inference()
