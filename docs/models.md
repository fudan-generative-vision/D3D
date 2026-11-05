# Released checkpoints

All six checkpoints are grouped with the training datasets in the
[D3D collection](https://huggingface.co/collections/fudan-generative-ai/d3d).

| Hub repository | Target | Layers | Tensor parameters | FP32 weight bytes |
|:---|:---|---:|---:|---:|
| [D3D-1.7B-8B](https://huggingface.co/fudan-generative-ai/D3D-1.7B-8B) | Qwen3-8B | 28 | 2,031,739,904 | 8,126,995,168 |
| [D3D-1.7B-32B](https://huggingface.co/fudan-generative-ai/D3D-1.7B-32B) | Qwen3-32B | 28 | 2,031,739,904 | 8,126,995,168 |
| [D3D-8B-8B](https://huggingface.co/fudan-generative-ai/D3D-8B-8B) | Qwen3-8B | 36 | 8,190,735,360 | 32,762,987,240 |
| [D3D-8B-32B](https://huggingface.co/fudan-generative-ai/D3D-8B-32B) | Qwen3-32B | 36 | 8,190,735,360 | 32,762,987,240 |
| [D3D-24D-p1.7B-32B](https://huggingface.co/fudan-generative-ai/D3D-24D-p1.7B-32B) | Qwen3-32B | 24 | 1,830,395,904 | 7,321,613,968 |
| [D3D-31D-p8B-8B](https://huggingface.co/fudan-generative-ai/D3D-31D-p8B-8B) | Qwen3-8B | 31 | 7,226,003,200 | 28,904,052,320 |

Tensor parameters sum all tensor shapes in the safetensors headers, including
embeddings and output heads. Names identify the architecture family, not the
exact stored parameter count. Weight bytes refer to the original FP32 files,
not BF16 GPU memory.

## Contents

Each repository contains:

- the original safetensors shards and weight index;
- original FP32 tensor values;
- tokenizer and generation assets;
- a model card, license, and `release_metadata.json`;
- inference compatibility classes and an `AutoModelForCausalLM` mapping.

Training FSDP files, optimizer state, recovery manifests, logs, and datasets are
not included.

## Loading

Use the D3D entry points. They load the Hub snapshot through local compatibility
classes with remote code execution disabled. Hub-hosted Python is provided for
architecture inspection; arbitrary Transformers versions are not supported.

## Validation

Release checks cover repository access, shard completeness, weight sizes and
hashes, configuration dimensions, and tokenizer loading. They do not establish
generation quality or throughput for every checkpoint. To reproduce a result,
record the Hub commit SHA, dataset samples, token cap, attention backend, and
decoding configuration.
