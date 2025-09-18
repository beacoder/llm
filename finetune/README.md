# Prepare dataset
- 85% as finetune dataset.
- 10% as test dataset.
- 5% as verification (inference) dataset.
- dataset should be at least 100+, 1000+ would be better.

# Train/Eval lost
- during training, both train_loss and eval_loss should go down
- the final train_loss should be around 0.5 ~ 0.3
- train_loss will be a little smaller than eval_loss
- train_loss will go down to a pointer where it can't go down any further. (train_loss line becomes flat)
- if train_loss line is not approaching flat, means further training needed, could be due to too small datasets.

# Use SFTTrainer when
- You're doing instruction tuning or chat model fine-tuning.
- Your data is in conversational format (e.g., ChatML: <|im_start|>system\n...\n<|im_end|>).
- You want to fine-tune a model to follow instructions or respond in chats.
- You want automatic masking of loss so only the assistant's reply contributes to the loss.
- You're using datasets in these formats: ChatML, Alpaca, UltraChat.

# LoRA vs Full-Parameter Fine-Tuning
## ✅ **LoRA is usually better**:
  - **Efficiency**:
    - Trains **<1% of parameters** (vs. 100% in full fine-tuning).
    - **Fits on small GPUs** (e.g., 24GB VRAM), avoids OOM errors.
  - **Performance**:
    - Matches/maintains full fine-tuning accuracy for most tasks (including sentiment analysis).
    - **Less catastrophic forgetting** (preserves base knowledge better).
  - **Practicality**:
    - Instantly switch tasks by swapping LoRA weights.
    - Merge weights post-training for deployment (no inference overhead).

## ⚠️ **Full fine-tuning only if**:
  - You have **massive task-specific data** (e.g., >1M labeled examples).
  - You have **dedicated infrastructure** (multi-GPU, high VRAM).
  - Marginal accuracy gains justify **3-4× higher cost** (training + storage).

## Recommendation:
  - **Start with LoRA** (`r=8`, `alpha=16`). It’s the **industry standard** for task adaptation (used in 90%+ of real-world deployments).
  - **Only consider full fine-tuning** if LoRA underperforms *and* you have resources to burn.

# Non-Reasoning vs Reasoning
- ✅ LLMs are excellent at statistical pattern matching — but not “understanding” in the human sense.
- ✅ Reasoning improves accuracy by forcing structured, step-by-step generation — reducing errors and hallucinations.
- ✅ Reasoning helps LLMs tap into deeper, more robust “hidden patterns” in the data — especially multi-step, causal, or logical structures that lead to correct answers.
## Reasoning prompts
- Let's think step by step.
- Always reason step by step before answering.
- Reason through this step by step to ensure your answer is correct.
- Etc.
