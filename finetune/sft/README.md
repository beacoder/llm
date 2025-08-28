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
