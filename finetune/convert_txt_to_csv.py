import pandas as pd


# Load the chat history from a plain text file
with open("chat_history.txt", "r") as f:
    lines = f.readlines()

# Prepare lists to store the data
instructions = []
inputs = []
outputs = []

# Process the chat history
i = 0
while i < len(lines):
    if lines[i].startswith("User:"):
        # Extract user message (instruction)
        instruction = lines[i].replace("User:", "").strip()

        # Extract assistant message (output)
        if i + 1 < len(lines) and lines[i + 1].startswith("Assistant:"):
            output = lines[i + 1].replace("Assistant:", "").strip()
            i += 2  # Move to the next user message
        else:
            output = ""  # No assistant response found

        # Append to lists
        instructions.append(instruction)
        inputs.append("")  # Leave blank if no additional context
        outputs.append(output)
    else:
        i += 1  # Skip invalid lines

# Create a DataFrame
df = pd.DataFrame({
    "instruction": instructions,
    "input": inputs,
    "output": outputs
})

# Save to CSV
df.to_csv("chat_dataset.csv", index=False)
print("CSV file saved successfully!")
