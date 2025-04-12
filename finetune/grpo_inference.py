import logging
from typing import Tuple

from unsloth import FastLanguageModel
from transformers import TextStreamer


# Configuration for Unsloth model
MAX_SEQ_LENGTH = 1024  # Can increase for longer reasoning traces
LORA_RANK = 64 # Larger rank = smarter, but slower

def load_model_and_tokenizer(model_name: str) -> Tuple[FastLanguageModel, any]:
    """
    Load the model and tokenizer.

    Args:
        model_name: Name of the pre-trained model.

    Returns:
        Tuple of model and tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
        # fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = LORA_RANK,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )
    FastLanguageModel.for_inference(model)  # Enable faster inference
    return model, tokenizer

# Initialize model and tokenizer
model, tokenizer = load_model_and_tokenizer("qwen2.5-3B-reasoning")

# Inference prompt template
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def run_inference() -> None:
    """Run inference loop for user input."""
    while True:
        user_input = input("Enter your input (type 'quit' to exit): ")

        if user_input.lower() == "quit":
            logging.info("Exiting inference loop.")
            print("Goodbye!")
            break

        messages = [
            {"role" : "system", "content" : SYSTEM_PROMPT},
            {"role" : "user", "content" : user_input},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = False
        )

        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        logging.info("Generating response...")
        _ = model.generate(
            **tokenizer(inputs, return_tensors = "pt").to("cuda"),
            max_new_tokens = 128, # Increase for longer outputs!
            temperature = 0.8, top_p = 0.95, top_k = 64, # Recommended Qwen-2.5 settings!
            streamer = text_streamer,
        )


if __name__ == "__main__":
    run_inference()
