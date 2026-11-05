# Architecture

D3D is a draft-and-verify decoding architecture. A Qwen3-derived diffusion
draft runs on one GPU; the target model verifies draft tokens and supplies
anchors on another GPU. The draft generates masked token blocks, while the
target remains autoregressive.

## Terms

| Term | Definition |
|:---|:---|
| Target | The autoregressive model whose output distribution is being approximated. |
| Draft | The Qwen3-derived diffusion model that proposes tokens. |
| Anchor | A target-selected token that conditions the next draft block. |
| Block | A group of draft tokens generated in one diffusion step. |
| Packed candidates | Multiple draft continuations generated for possible verification outcomes. |
| KV cache | Cached key and value tensors used for autoregressive decoding. |
| EOS | End-of-sequence token. |
| Accept length | Mean number of tokens committed per target verification step. |

## Components

| Component | GPU | Responsibility | Code |
|:---|:---:|:---|:---|
| Diffusion draft | 1 | Generate blocks and packed continuation candidates | `models/d3d/`, `single/eval_async_packed.py` |
| Target verifier | 0 | Verify pending drafts and provide anchor tokens | `single/eval_sync.py`, `serving/backend.py` |
| Coordinator | both | Commit accepted tokens and select reusable candidates from verification feedback | `single/eval_async_packed.py`, `serving/engine.py` |

## Decode cycle

1. Prefill the prompt and obtain an anchor from the target.
2. Generate a masked draft block conditioned on the anchor.
3. Send the pending block to the target for verification. While verification
   runs, prepare packed candidates for possible acceptance positions and anchors.
4. Commit the accepted prefix and the target-selected token, then reuse a
   matching candidate. If no candidate matches, generate a new draft.
5. Repeat until EOS or the output-token limit.

Drafting and verification run concurrently on separate GPUs. Verification
feedback determines which candidate can be reused.

## Execution paths

| Path | Entry point | Cache and scheduling |
|:---|:---|:---|
| Single request | `single/eval_async_packed.py` | Per-request draft and target caches |
| Concurrent requests | `python -m serving.bench` | Paged KV caches, request admission, draft microbatches |

`models/d3d/configuration_d3d.py` and `modeling_d3d.py` define `D3DConfig`,
`D3DModel`, and `D3DForCausalLM`. The loader accepts both D3D and legacy draft
type tags without executing Hub-hosted Python.

## Scope

Released checkpoints are draft models paired with specific target models. They
are not replacements for the target models. Throughput depends on the model
pair, workload, and hardware; see the [benchmark protocol](../benchmarks/README.md).
