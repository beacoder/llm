### Dataset Preparation
- Split: **85% fine-tune**, **10% test**, **5% inference validation**.
- Minimum size: **100+ samples** (1,000+ strongly recommended).

### Training/Evaluation Loss
- Both `train_loss` and `eval_loss` must **decrease** during training.
- Target final `train_loss`: **0.3–0.5**.
- `train_loss` should be **slightly lower** than `eval_loss`.
- `train_loss` must **converge** (flatten); non-convergence indicates **insufficient data** or **inadequate training**.

### When to Use SFTTrainer
- For **instruction/chat fine-tuning** (e.g., instruction-following, conversational models).
- Input data in **ChatML**, **Alpaca**, or **UltraChat** format.
- Requires **automatic loss masking** (only assistant responses contribute to loss).

### LoRA vs. Full Fine-Tuning
#### ✅ **Prefer LoRA** (`r=8`, `alpha=16`)
  - **Efficiency**: Trains **<1% parameters** (vs. 100% full FT); runs on **24GB VRAM GPUs**.
  - **Performance**: Matches full FT accuracy; **reduces catastrophic forgetting**.
  - **Practicality**: Task-switching via weight swaps; **zero inference overhead** post-merge.
  - *Industry standard for 90%+ deployments*.

#### ⚠️ **Full FT Only If**
  - **>1M task-specific samples** + **dedicated multi-GPU infrastructure**.
  - Marginal accuracy gains justify **3–4× higher costs**.

### Reasoning Capabilities
- LLMs excel at **statistical pattern matching** (not human-like "understanding").
- **Reasoning prompts** (e.g., *"Let's think step by step"*) improve accuracy by:
  - Enforcing **structured, step-by-step generation**.
  - Reducing **errors/hallucinations**.
  - Unlocking **deeper causal/logical patterns** in data.
  *Common prompts: "Reason step by step", "Ensure correctness via reasoning".*
