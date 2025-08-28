import pandas as pd


def read_chat_history(file_path):
    """Read chat history from a plain text file."""
    with open(file_path, "r") as f:
        return f.readlines()


def process_chat_history(lines):
    """Process chat history to extract instructions, inputs, and outputs."""
    instructions = []
    inputs = []
    outputs = []
    i = 0

    while i < len(lines):
        if lines[i].startswith("User:"):
            instruction = lines[i].replace("User:", "").strip()

            if i + 1 < len(lines) and lines[i + 1].startswith("Assistant:"):
                output = lines[i + 1].replace("Assistant:", "").strip()
                i += 2
            else:
                output = ""

            instructions.append(instruction)
            inputs.append("")  # No additional context
            outputs.append(output)
        else:
            i += 1

    return instructions, inputs, outputs


def save_to_csv(instructions, inputs, outputs, output_file):
    """Save processed data to a CSV file."""
    df = pd.DataFrame({
        "instruction": instructions,
        "input": inputs,
        "output": outputs
    })
    df.to_csv(output_file, index=False)
    print(f"CSV file saved successfully to {output_file}!")


# Main execution
if __name__ == "__main__":
    chat_history_file = "chat_history.txt"
    output_csv_file = "chat_dataset.csv"

    lines = read_chat_history(chat_history_file)
    instructions, inputs, outputs = process_chat_history(lines)
    save_to_csv(instructions, inputs, outputs, output_csv_file)
