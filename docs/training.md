# Training and layer pruning

Training code is in `training/train/`; progressive layer pruning is in `training/pruning/`.
The native environment is described in [environment.md](environment.md).

Terms used below: the **target** is the verifier model; the **draft** is the
model being trained; an **anchor** is a token position that conditions a draft
block.

## Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The launcher uses FlashAttention-2 by default:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Pass `--attn-implementation sdpa` to use PyTorch SDPA instead. SDPA still
requires the GPU launcher; it is not a CPU training mode.

## Training data

Each line is one conversation:

```json
{"id":"example-0","status":"success","conversations":[{"role":"user","content":"What is 2 + 2?"},{"role":"assistant","content":"4"}]}
```

- `conversations` or `messages` is accepted.
- ShareGPT-style `from` / `value` fields with `human` / `gpt` roles are normalized.
- Assistant `reasoning_content` is passed to the tokenizer chat template when present.
- The last assistant turn is supervised; earlier turns form its prompt.
- Rows with `status` other than absent or `success` are skipped.
- Rows exceeding the configured prompt or sequence limits, or lacking enough
  response tokens, are skipped.

Use responses from the target model and its tokenizer. Keep validation and test
prompts disjoint from training prompts. Formal experiments should use at least
32,768 distinct training prompts; repeated epochs or extra anchors from one
prompt do not count as distinct examples.

## Offline training

Provide local paths for the draft checkpoint, target tokenizer, and training
jsonl:

```bash
export DRAFT_MODEL=checkpoints/Qwen3-1.7B
export TARGET_MODEL=checkpoints/Qwen3-8B
export DATA_PATH=data/train.jsonl
export OUTPUT_DIR=outputs/d3d_b16_a128

CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/train.sh \
    --attn-implementation sdpa --disable-wandb
```

`TARGET_MODEL` selects the tokenizer. The default CE objective uses saved target
responses and does not load a live target model. KL objectives require an
explicit `--target-model`.

Defaults: block size 16, 128 anchors per example, position-decayed
cross-entropy (CE) plus an autoregressive (AR) loss, BF16, FSDP2, gradient
accumulation 8, and four GPUs. Override them with
CLI flags:

```bash
bash scripts/train.sh --help
bash scripts/train.sh --block-size 32 --parallel-anchors-per-example 64 \
    --loss-decay-gamma 15.5 --gradient-checkpointing --disable-wandb
```

`PY` selects the launcher interpreter; `MASTER_PORT` changes the distributed
port. For W&B, omit `--disable-wandb` and authenticate with `wandb login` or
`WANDB_MODE=offline`.

Saved checkpoints include the model configuration and training-compatible
remote code. Evaluate a local checkpoint with the normal `--drafter` / `--draft`
flags.

## Progressive layer pruning

Each cycle removes one decoder layer:

```text
select → prune → train → choose
```

Start from a trained D3D checkpoint and repeat from the selected recovery
checkpoint until the desired depth is reached.

### Selection and validation data

Use fixed tokenized datasets in `.parquet`, `.jsonl`, or `.json` format with:

| Column | Meaning |
|:---|:---|
| `prefix_ids` | Causal context token IDs |
| `target_ids` | Anchor token followed by at least one full draft block |
| `analysis_id` | Optional unique row identifier |
| `split` | Optional split label |

Token IDs must use the target tokenizer. With a `split` column, the defaults
select `layer_selection` and `layer_validation`; without it, the whole file is
used. Both commands expect 512 rows by default; set `--expected-samples` for a
different size.

### Commands

```bash
export PARENT=outputs/d3d_b16_a128/step-5000
export CYCLE=outputs/pruning/cycle-01

bash scripts/prune.sh select \
    --parent "$PARENT" --examples data/layer_selection.parquet \
    --output-dir "$CYCLE/selection" --devices cuda:0 cuda:1

bash scripts/prune.sh prune \
    --selection "$CYCLE/selection/selection.json" \
    --output "$CYCLE/pruned"

CUDA_VISIBLE_DEVICES=0,1 bash scripts/prune.sh train --nproc-per-node 2 \
    --student "$CYCLE/pruned" --dataset "$DATA_PATH" \
    --output-dir "$CYCLE/recovery" --max-steps 1000

bash scripts/prune.sh choose \
    --run-dir "$CYCLE/recovery" --examples data/layer_validation.parquet \
    --output-dir "$CYCLE/chosen" --devices cuda:0 cuda:1
```

Recovery uses FSDP2 and the full-depth teacher recorded in the pruning
provenance. The default optimizer is TorchAO 8-bit AdamW; use `--optimizer
adamw` for standard AdamW. Preserve the teacher checkpoint and metadata between
stages. Selection and pruning commands refuse to overwrite existing outputs.

`choose` writes `selected_checkpoint.json` with the selected checkpoint path
instead of copying weights. Use that path as the next cycle's parent or as the
inference draft.

## Code map

| Path | Role |
|:---|:---|
| `training/train/train.py`, `training/train/trainer.py` | Distributed training entry point and loop |
| `training/train/dataset.py`, `training/train/blocks.py` | Conversation loading and packed anchor-conditioned blocks |
| `training/train/losses.py`, `training/train/objective_kernels.py` | Cross-entropy / Kullback-Leibler losses, position weighting, and auxiliary objectives |
| `training/train/runtime.py`, `training/train/model_utils.py` | FSDP, checkpoint/code staging, and model helpers |
| `training/train/trajectory.py` | Optional trajectory-distillation utilities; disabled by default |
| `training/pruning/` | Layer selection, weight remapping, recovery, and validation |
| `training/model/` | Qwen3-derived training model and checkpoint auto-map template |
| `common/token_ids.py` | Checkpoint mask-token convention and validation |

Qwen and Hugging Face attribution is preserved in the imported model files.
