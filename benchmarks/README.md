# Benchmarks

## Datasets

Dataset files are downloaded and cached at runtime; they are not stored in this
repository.

| ID | Dataset | Split | Source |
|:---|:---|:---|:---|
| `gsm8k` | GSM8K | `test` | `openai/gsm8k` (`main`) |
| `math500` | MATH-500 | `test` | `HuggingFaceH4/MATH-500` |
| `humaneval` | HumanEval | `test` | `openai/openai_humaneval` |
| `mbpp` | MBPP | `test` | `google-research-datasets/mbpp` (`sanitized`) |
| `mt-bench` | MT-Bench prompts | `train` | `HuggingFaceH4/mt_bench_prompts` |
| `aime25` | AIME 2025 | `train` | `MathArena/aime_2025` |
| `livecodebench-v6` | LiveCodeBench v6 | `test6.jsonl` | `livecodebench/code_generation_lite` |

## Protocol

- 2 × NVIDIA H800 80 GB: target on `cuda:0`, draft on `cuda:1`.
- BF16, greedy decoding, `max_new_tokens=2048`.
- First 32 samples per dataset, or all rows when the split is shorter.
- Concurrency: `1`, `4`, `8`, `16`.
- Warmup, model loading, and tokenization are excluded from timed decoding.
- Record package versions, CUDA/PyTorch build, GPU names, model revisions,
  dataset revisions, seed, configuration, and source commit.

Dataset files use `D3D_DATA_CACHE` when a local jsonl cache is provided;
otherwise the Hugging Face cache is used.

## Metrics

| Field | Definition |
|:---|:---|
| `baseline_tps` | Target-only autoregressive tokens per second |
| `speedup` | D3D tokens per second divided by `baseline_tps` |
| `accept_length` | Mean tokens committed per target verification step |

The baseline uses the same target checkpoint, prompts, token budget, greedy
settings, and concurrency as the D3D run. It runs on the target GPU only.

Single-request timing excludes target/draft prefill and the first target token
emitted during prefill. `output_tokens` still counts every generated token.
Multi-request timing includes prefill, request admission, decode, and drain.
Neither measurement includes model loading, warmup, or tokenization.

Single-request summaries store the metrics under `headline`; multi-request cells
store them at the top level and under `summary`. `--no-baseline` skips the
baseline and leaves `speedup` empty.

## Scope

MT-Bench measures the first turn only, not the full multi-turn judge evaluation.
HumanEval, MBPP, and LiveCodeBench outputs are generated but not executed or
graded. GSM8K/MATH answer matching is diagnostic. These scripts measure speed,
not answer quality.

## Running

Scripts forward extra flags to their Python entry point. Override datasets with
`DATASETS="math500 humaneval"`. Select models with `--verifier` / `--drafter`
for single-request runs or `--target` / `--draft` for multi-request runs; the
scripts fall back to `TARGET_MODEL` / `DRAFT_MODEL`.

Defaults: block size 16, one denoising step, three anchor branches, greedy
decoding, thinking disabled, and causal draft-prefix attention. Concurrency 1
keeps three anchors; multi-request cells use one anchor because multiple branches
do not fit the paged KV budget when requests overlap. `--anchor-num` overrides
both.

`--max-new-tokens` overrides the 2048-token limit for smoke runs; shorter runs
are not formal protocol results. `--max-samples` changes the sample cap. The
multi-request script runs each concurrency separately and writes one result file
per cell (`benchmark_c1.json`, `benchmark_c4.json`, ...).

The paged KV cache defaults to 512 pages per device. A 32B target may require a
smaller pool even on an 80 GB GPU. The runner checks the requested context and
worst-case branch reservations before model loading. Record the chosen value
and peak memory for formal runs.
