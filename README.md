<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
</p>

<h1 align="center">D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding</h1>

<p align="center">
  <a href="https://huggingface.co/collections/fudan-generative-ai/d3d"><img src="https://img.shields.io/badge/Collection-D3D-fbbf24?style=flat-square" alt="D3D collection"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-Get_started-22c55e?style=flat-square" alt="Quick start"></a>
  <a href="https://arxiv.org/abs/2611.06425"><img src="https://img.shields.io/badge/Paper-arXiv%3A2611.06425-b31b1b?style=flat-square" alt="D3D paper"></a>
  <a href="docs/architecture.md"><img src="https://img.shields.io/badge/Architecture-Docs-0ea5e9?style=flat-square" alt="Architecture guide"></a>
  <a href="#benchmarking"><img src="https://img.shields.io/badge/Benchmarks-Protocol-f97316?style=flat-square" alt="Benchmark protocol"></a>
  <a href="#training-data"><img src="https://img.shields.io/badge/Training_Data-Released-8b5cf6?style=flat-square" alt="Training data"></a>
</p>

<a id="overview"></a>

## ✨ Overview

**D3D** decouples a diffusion **draft model** from the **target verifier** on
separate GPUs. While the target verifies a draft block, the draft prepares
continuations for several possible verification outcomes. The target's result
then selects the matching continuation for the next decoding step.

<p align="center">
  <img src="docs/architecture.png" alt="D3D method overview: drafter training, progressive layer pruning, and asynchronous inference" width="960">
</p>

**How it works:** a target-selected anchor starts a parallel diffusion draft;
the target verifies the pending block while the draft GPU prepares possible
continuations; verification feedback selects the continuation to reuse.
See the [architecture guide](docs/architecture.md) for the execution flow.

<a id="model-zoo"></a>

## 🤗 Model Zoo

**[Browse the D3D collection →](https://huggingface.co/collections/fudan-generative-ai/d3d)**

The [D3D collection](https://huggingface.co/collections/fudan-generative-ai/d3d)
contains six Qwen3-derived draft checkpoints. These checkpoints are **draft
models**, not replacements for their target models. `p` denotes pruning; `24D`
and `31D` denote retained transformer-layer counts.

| Target verifier | Draft family | Layers | D3D checkpoint |
|:---|:---|:---:|:---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 1.7B | 28 | [D3D-1.7B-8B](https://huggingface.co/fudan-generative-ai/D3D-1.7B-8B) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8B | 36 | [D3D-8B-8B](https://huggingface.co/fudan-generative-ai/D3D-8B-8B) |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Pruned 8B | 31 | [D3D-31D-p8B-8B](https://huggingface.co/fudan-generative-ai/D3D-31D-p8B-8B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 1.7B | 28 | [D3D-1.7B-32B](https://huggingface.co/fudan-generative-ai/D3D-1.7B-32B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 8B | 36 | [D3D-8B-32B](https://huggingface.co/fudan-generative-ai/D3D-8B-32B) |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Pruned 1.7B | 24 | [D3D-24D-p1.7B-32B](https://huggingface.co/fudan-generative-ai/D3D-24D-p1.7B-32B) |

<a id="installation"></a>

## 📦 Installation

```bash
git clone https://github.com/fudan-generative-vision/D3D.git
cd D3D
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e . --no-deps
```

<a id="quick-start"></a>

## 🚀 Quick Start

### 1. Target–draft configuration

```bash
export CUDA_VISIBLE_DEVICES=0,1

# optional: override the defaults used below
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=fudan-generative-ai/D3D-1.7B-8B
```

The target runs on `cuda:0`, and the draft runs on `cuda:1`.

### 2. Run asynchronous decoding

**Single-request** — one MATH-500 sample, up to 128 generated tokens:

```bash
python inference/native/single_request/eval_async_packed.py \
    --drafter "$DRAFT_MODEL" \
    --verifier "$TARGET_MODEL" \
    --dataset math500 \
    --max-samples 1 \
    --max-new-tokens 128 \
    --verifier-device cuda:0 \
    --draft-device cuda:1 \
    --dtype bf16 \
    --temperature 0 \
    --block-size 16 \
    --denoising-steps 1 \
    --attention-mask-mode causal \
    --anchor-num 3 \
    --packed-accepted-counts all \
    --output-json outputs/single_request/result.json \
    --summary-json outputs/single_request/summary.json
```

**Multi-request** — four samples at concurrency 4:

```bash
python inference/native/multi_request/bench.py \
    --target "$TARGET_MODEL" \
    --draft "$DRAFT_MODEL" \
    --datasets math500 \
    --max-samples 4 \
    --max-new-tokens 128 \
    --concurrency 4 \
    --anchor-num 3 \
    --pages 512 \
    --output outputs/multi_request/result.json
```

Override `TARGET_MODEL` / `DRAFT_MODEL` for another model pair. Increase
`--max-new-tokens` for longer generations; for multi-request runs, increase
`--concurrency` and adjust `--pages` to fit the KV budget.

<a id="benchmarking"></a>

## 📊 Benchmarking

Run single-request evaluation over the standard datasets:

```bash
bash scripts/benchmark_single_request.sh
```

Run multi-request evaluation at request concurrency 1, 4, 8, and 16:

```bash
bash scripts/benchmark_multi_request.sh
```

Datasets are downloaded at runtime. See the
[benchmark protocol](benchmarks/README.md) for sources and splits.

## ⚙️ Inference Engine Support

The repository supports native D3D inference and an optional two-GPU SGLang
Hybrid backend. See the [SGLang backend guide](docs/sglang-backend.md).
Install SGLang separately:

```bash
bash scripts/sglang/install.sh
```

Start the two-GPU backend (GPU 0: Target + local Draft; GPU 1: optional Draft):

```bash
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=fudan-generative-ai/D3D-1.7B-8B
export TARGET_GPU=0 DRAFT_GPU=1
export MAX_RUNNING_REQUESTS=1
bash scripts/sglang/launch.sh hybrid
```

Then run a MATH-500 evaluation from another terminal:

```bash
export MAX_RUNNING_REQUESTS=1
export TARGET_MODEL=Qwen/Qwen3-8B
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset math500 \
  --tokenizer "$TARGET_MODEL" \
  --samples 128 --concurrency "$MAX_RUNNING_REQUESTS" \
  --repetitions 1 --max-new-tokens 512 \
  --output outputs/sglang/example.json
```

The dataset is downloaded automatically and cached by Hugging Face Datasets.

Set `MAX_RUNNING_REQUESTS` to `1`, `4`, `8`, or `16` in both terminals;
restart the backend after changing it.

**SGLang Hybrid throughput on the complete seven-benchmark suite.**

D3D-1.7B → Qwen3-8B, concurrency 1, `anchor_num=1`, block size 16,
`max_new_tokens=512`, greedy decoding, thinking disabled, and natural EOS.

| Dataset | tokens/s (speedup) |
|---|---:|
| GSM8K | 658.10 (4.876×) |
| MATH-500 | 572.66 (4.243×) |
| AIME25 | 394.56 (2.924×) |
| HumanEval | 479.15 (3.550×) |
| MBPP | 428.39 (3.174×) |
| LCB-v6 | 267.42 (1.982×) |
| MT-Bench | 258.54 (1.916×) |
| **Aggregate** | **501.60 (3.717×)** |

<a id="training-data"></a>

## 📚 Training Data

All corpora use [Open-PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend)
prompts. See each dataset card for generation settings, statistics, and download
instructions. Released models and datasets are grouped in the
[D3D collection](https://huggingface.co/collections/fudan-generative-ai/d3d).

| Corpus | Target Model | Samples | Dataset |
|:---|:---|---:|:---|
| **PerfectBlend: Qwen3.8-27B Thinking** | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 1,417,675 | [PerfectBlend-Qwen3.8-27B-Thinking](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3.8-27B-Thinking) |
| **PerfectBlend: Qwen3-4B Thinking (Greedy)** | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 1,420,909 | [PerfectBlend-Qwen3-4B-Thinking-Greedy](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-4B-Thinking-Greedy) |
| **PerfectBlend: Qwen3-4B NoThink (Greedy)** | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 1,420,908 | [PerfectBlend-Qwen3-4B-NoThink-Greedy](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-4B-NoThink-Greedy) |
| **PerfectBlend: Qwen3-4B NoThink (Sampling)** | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 1,420,909 | [PerfectBlend-Qwen3-4B-NoThink-Sampling](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-4B-NoThink-Sampling) |
| **PerfectBlend: Qwen3-8B NoThink (Greedy)** | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 1,420,908 | [PerfectBlend-Qwen3-8B-NoThink-Greedy](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-8B-NoThink-Greedy) |
| **PerfectBlend: Qwen3-8B NoThink (Sampling)** | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 1,420,909 | [PerfectBlend-Qwen3-8B-NoThink-Sampling](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-8B-NoThink-Sampling) |
| **PerfectBlend: Qwen3-32B NoThink (Sampling)** | [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | 1,420,500 | [PerfectBlend-Qwen3-32B-NoThink-Sampling](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Qwen3-32B-NoThink-Sampling) |
| **PerfectBlend: Gemma-4-12B-IT (Greedy)** | [Gemma-4-12B-IT](https://huggingface.co/google/gemma-4-12B-it) | 1,420,898 | [PerfectBlend-Gemma-4-12B-IT-Greedy](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Gemma-4-12B-IT-Greedy) |
| **PerfectBlend: Gemma-4-31B-IT (Greedy)** | [Gemma-4-31B-IT](https://huggingface.co/google/gemma-4-31B-it) | 1,420,903 | [PerfectBlend-Gemma-4-31B-IT-Greedy](https://huggingface.co/datasets/fudan-generative-ai/PerfectBlend-Gemma-4-31B-IT-Greedy) |

Each corpus targets the listed model; use the matching tokenizer for training.

## 🏋️ Training and Layer Pruning

Training code lives in `training/train/` and `training/pruning/`, including
offline training on target-generated responses, layer selection, physical
pruning, recovery training, and recovery-checkpoint selection.

Use the **same environment** installed above with `requirements.txt` for
training and native inference.

```bash
export DRAFT_MODEL=checkpoints/Qwen3-1.7B
export TARGET_MODEL=checkpoints/Qwen3-8B
export DATA_PATH=data/train.jsonl
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/train.sh \
    --attn-implementation sdpa --disable-wandb
```

Download the initial checkpoints and prepare the jsonl dataset first. See the
[training and progressive-pruning guide](docs/training.md) for installation,
data schemas, loss settings, and all four pruning stages.

## Citation

If you use D3D, please cite:

```bibtex
@article{chen2026d3d,
  title   = {D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding},
  author  = {Baoyou Chen and Ruiqiao Mei and Guancong Lin and Tao Song and Weihao Yuan and Qun Liu and Siyu Zhu},
  journal = {arXiv preprint arXiv:2611.06425},
  year    = {2026}
}
```
