# SGLang backend

D3D provides two SGLang deployment modes. Both use a separate environment from
native D3D.

| Mode | GPU 0 | GPU 1 | Target/draft separation |
|:---|:---|:---|:---|
| Hybrid | Target plus a required local draft | Optional remote draft for branch preparation | Not strict |
| `remote` | Target only | Draft only | Strict |

## Default: Hybrid

Hybrid uses the fork's original `DECOUPLED` implementation. GPU 0 holds the
target and a local draft; GPU 1 holds a draft that prepares additional branches
asynchronously. If the remote draft does not cover a result, the local draft
produces it.

Install and launch from the D3D root:

```bash
bash scripts/sglang/install.sh
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=fudan-generative-ai/D3D-1.7B-8B
export TARGET_GPU=0 DRAFT_GPU=1 MAX_RUNNING_REQUESTS=1
bash scripts/sglang/launch.sh hybrid
```

Benchmark from another terminal:

```bash
export TARGET_MODEL=Qwen/Qwen3-8B MAX_RUNNING_REQUESTS=1
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset math500 --tokenizer "$TARGET_MODEL" \
  --samples 128 --concurrency "$MAX_RUNNING_REQUESTS" \
  --repetitions 1 --max-new-tokens 512 \
  --output outputs/sglang/hybrid-c${MAX_RUNNING_REQUESTS}.json
```

Set `MAX_RUNNING_REQUESTS` to `1`, `4`, `8`, or `16` and restart the backend
after changing it. Use a new output filename for each run. Datasets are
downloaded and cached automatically.

Hybrid defaults: block size 16, one anchor, FP16 remote draft, BF16 target,
1024-token local packed graph, remote packed CUDA graph enabled, 12 ms
covered-result timeout, and local prefix reuse disabled. Target decode CUDA
graph is enabled; prefill CUDA graph is disabled. `--dry-run` prints commands
without starting GPU workers.

`DETERMINISTIC=1` enables diagnostic batch-invariant operators. It is not the
performance default; do not mix deterministic and standard runs in a speedup
denominator.

## Strict remote

The `remote` mode preserves strict GPU separation. GPU 0 runs only the target
and its KV cache; GPU 1 runs only the draft and its KV cache. There is no
target-local draft and no Hybrid fallback.

Scope: single node, tensor parallel 1 and data parallel 1 (TP1/DP1),
concurrency 1, greedy decoding. The RPC endpoint is a trusted localhost
channel, not a public serving API.

## Environment

Do not install SGLang into the native D3D environment. The backend uses
`.venv-sglang`, built by `scripts/sglang/install.sh`; `scripts/sglang/runtime.lock`
pins the tested runtime.

Tested versions: Python 3.12, PyTorch `2.13.0+cu129`, Transformers `5.12.1`,
FlashInfer `0.6.17`, and `sglang-kernel==0.4.6.post1+cu129`. Installation uses
`--no-deps` to avoid replacing the pinned CUDA 12 PyTorch build. Do not run an
unpinned resolver over this environment. See [environment.md](environment.md)
for the native environment.

## Launch

```bash
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=fudan-generative-ai/D3D-1.7B-8B
export TARGET_GPU=0 DRAFT_GPU=1
export BLOCK_SIZE=16 ANCHORS=1
```

Terminal 1, draft on GPU 1:

```bash
bash scripts/sglang/launch.sh draft --metrics-file outputs/sglang/draft.json
```

Wait for `READY tcp://127.0.0.1:17891`, then start the target on GPU 0:

```bash
bash scripts/sglang/launch.sh remote
```

The launcher rejects `--speculative-draft-model-path` in `remote` mode to prevent
accidental Hybrid placement. Target settings: BF16, FA3 attention, greedy, no
radix cache, decode CUDA graph enabled, prefill CUDA graph disabled, context
length 8192, one running request.

`MAX_BRANCHES` defaults to `BLOCK_SIZE * ANCHORS`. Draft graph capacity defaults
to at least 1024 tokens and rounds up to a power of two sufficient for branch
blocks and the pending proposal. Override it with `--packed-graph-max-tokens`.
`DRAFT_CUDA_GRAPH=0` disables the draft graph for ablation.

## Matched AR comparison

Stop the remote target, launch the native autoregressive (AR) baseline on the
same GPU, and run the same workload:

```bash
bash scripts/sglang/launch.sh ar
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset /path/to/math500-test.jsonl --tokenizer "$TARGET_MODEL" \
  --samples 8 --repetitions 3 --max-new-tokens 512 \
  --output outputs/sglang/ar.json
```

Stop AR, launch `remote`, and run the same command with:

```bash
  --baseline outputs/sglang/ar.json --output outputs/sglang/remote.json
```

Both sides use the same source indices, rendered chat token IDs, thinking
setting, temperature, token cap, and a disjoint warmup prompt.

| Metric | Definition |
|:---|:---|
| `speedup` | Remote output tokens/s divided by native SGLang AR output tokens/s |
| `acclen` | Generated tokens minus one prefill token per request, divided by target verification calls |
| `exact_text_matches` | Greedy output equality against AR; not an accuracy score |

`acclen` includes the target bonus token. Subtract one per call for
draft-only accepted tokens. EOS and token-cap trimming can reduce the final
verification's contribution. AR uses one GPU; remote D3D uses two.

Integration checks:

```bash
.venv-sglang/bin/python scripts/sglang/test_remote.py
python -m unittest discover -s tests -p test_sglang_benchmark.py -v
```

## Anchor sweep

Stop the existing draft, keep the remote target running, then sweep anchor
counts:

```bash
unset MAX_BRANCHES
.venv-sglang/bin/python scripts/sglang/sweep_anchors.py \
  --anchors 1 2 3 4 6 \
  --dataset /path/to/math500-test.jsonl --tokenizer "$TARGET_MODEL" \
  --baseline outputs/sglang/ar.json --output-dir outputs/sglang/anchor-sweep
```

The sweep uses block size 16. Treat the selected anchor count as a
workload-specific result.

## Source

The SGLang snapshot is from `wintem0531/sglang-speedup-nvidia`, commit
`db686d124840ebfdd61e787b3a7fbf543244b6f0`, based on SGLang `v0.5.18`.
See `../third_party/sglang/UPSTREAM_D3D.md` for attribution and the export
boundary.
