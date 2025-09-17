# Prepare dataset
1. 85% as finetune dataset.
2. 10% as test dataset.
3. 5% as verification (inference) dataset.
4. dataset should be at least 100+, 1000+ would be better.

# Train/Eval lost
1. during training, both train_loss and eval_loss should go down
2. the final train_loss should be around 0.5 ~ 0.3
3. train_loss will be a little smaller than eval_loss
4. train_loss will go down to a pointer where it can't go down any further. (train_loss line becomes flat)
5. if train_loss line is not approaching flat, means further training needed, could be due to too small datasets.

# Use SFTTrainer when
1. You're doing instruction tuning or chat model fine-tuning.
2. Your data is in conversational format (e.g., ChatML: <|im_start|>system\n...\n<|im_end|>).
3. You want to fine-tune a model to follow instructions or respond in chats.
4. You want automatic masking of loss so only the assistant's reply contributes to the loss.
5. You're using datasets in these formats: ChatML, Alpaca, UltraChat.

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
  **Start with LoRA** (`r=8`, `alpha=16`). It’s the **industry standard** for task adaptation (used in 90%+ of real-world deployments). Only consider full fine-tuning if LoRA underperforms *and* you have resources to burn.
