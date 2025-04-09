from unsloth import FastLanguageModel
from transformers import TextStreamer


#1 Unsloth configuration
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False. # Choose any! We auto support RoPE Scaling internally!

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen2.5-3B-chat", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

#2 Inference
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def run_inference():
    while True:
        user_input = input("Enter something: ")

        if user_input == "quit":
            print("bye byte !!!")
            break;

        inputs = tokenizer(
        [
            alpaca_prompt.format(
                f"{user_input}", # instruction
                "", # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to("cuda")

        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


if __name__ == "__main__":
    run_inference()
