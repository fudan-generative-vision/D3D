# Environment

`requirements.txt` defines the native environment for training, pruning, and
inference. Training uses `models/d3d_training/`; inference uses `models/d3d/`.
Python 3.10+ is supported; validation used Python 3.12.

| Package | Version |
|:---|:---|
| PyTorch | 2.8.0 (CUDA 12.8 build) |
| TorchVision | 0.23.0 |
| Transformers | 4.57.3 |
| Accelerate | 1.12.0 |
| TorchAO | 0.13.0 |
| FlashAttention | 2.8.3 |
| Datasets | 4.4.2 |
| Safetensors | 0.7.0 |

`SDPA` is PyTorch scaled dot-product attention. `FSDP2` is PyTorch fully sharded
data parallel v2. `CE` is cross-entropy; `AR` is autoregressive.

## Validation

Validated on two NVIDIA H800 80 GB GPUs with Python 3.12.

- Target: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`
- Draft: `fudan-generative-ai/D3D-1.7B-8B` at `23d559e19b110774c5930b077428b955ff020d8b`

| Check | Configuration | Result |
|:---|:---|:---|
| Single-request packed decoding | 3 MATH-500 samples, 128 tokens each, block 16, 3 anchors, BF16, SDPA | 384 tokens |
| FlashAttention-2 decoding | 1 MATH-500 sample, 128 tokens | 128 tokens |
| Concurrent decoding | 4 MATH-500 samples, 64 tokens each, concurrency 1 and 4, block 16, 3 anchors, 128 KV pages | 256 tokens per cell |
| Target reference | 4 prompts, greedy, 32 tokens each | Exact match with Transformers |
| Draft reference | Packed candidate comparison | 46/46 keys and 736/736 tokens match |
| CPU test suite | `python -m unittest discover -s tests -v` | 52 tests pass; GPU-only Liger test skipped |
| FSDP2 training | 8 synthetic arithmetic conversations, 2 optimizer updates, block 16, 4 anchors, CE + AR, BF16, SDPA and FlashAttention-2 | Step-2 checkpoint saved and generated 32 tokens on one MATH-500 request |

These are compatibility checks, not performance or accuracy measurements.

## Reproduce

Run the CPU suite:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

Run the two-GPU reference check with the revisions above:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m serving.bench \
    --target Qwen/Qwen3-8B --draft fudan-generative-ai/D3D-1.7B-8B \
    --datasets math500 --max-samples 4 --max-new-tokens 64 \
    --concurrency 1 4 --anchor-num 3 --pages 128 --validate \
    --output outputs/unified_environment/concurrent.json
```

The runner records runtime metadata and reference comparisons.

## SGLang

The optional SGLang backend uses a separate environment with PyTorch 2.13.0 and
Transformers 5.12.1, pinned in `scripts/sglang/runtime.lock`. Do not install it
into the native environment. See the [SGLang guide](sglang-backend.md).
