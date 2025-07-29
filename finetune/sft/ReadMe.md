# Prepare dataset

1. 85% as finetuen dataset.
2. 10% as test dataset.
3. 5% as verification (inference) dataset.
4. dataset should be at least 100+, 1000+ would be better.


Train/Eval lost

during training, both train_loss and eval_loss should go down
the final train_loss should be around 0.5 ~ 0.3
train_loss will be a little smaller than eval_loss
train_loss will go down to a pointer where it can't go down any further. (train_loss line becomes flat)
if train_loss line is not approaching flat, means further training needed, could be due to too small datasets.
