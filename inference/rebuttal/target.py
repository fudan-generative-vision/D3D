"""H800 target-only greedy TP1/TP2 benchmark, including prefill and memory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--warmup-prompts", type=int, default=8)
    parser.add_argument("--warmup-new-tokens", type=int, default=128)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common.data import load_rows, build_prompt
    from common.text import apply_chat_template, eos_token_ids

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != args.tp:
        raise ValueError("Use torchrun with nproc_per_node equal to --tp")
    torch.cuda.set_device(local_rank)
    hardware = torch.cuda.get_device_name(local_rank)
    if "H800" not in hardware:
        raise RuntimeError(f"Formal inference is configured for H800, found {hardware}")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if world > 1:
        dist.init_process_group("nccl")
    kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation)
    if world > 1:
        kwargs["tp_plan"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.target, **kwargs).eval()
    if world == 1:
        model = model.to(local_rank)
    else:
        from torch.distributed.tensor import DTensor
        if not any(isinstance(p, DTensor) for p in model.parameters()):
            raise RuntimeError("TP2 requested but model parameters are not tensor-parallel")
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    eos = set(eos_token_ids(tokenizer, None, model_path=args.target))
    rows, data_config, source = load_rows(args.dataset, args.max_samples)
    prompts = []
    for row in rows:
        prompt = build_prompt(data_config["question"](row), False)
        text = apply_chat_template(tokenizer, prompt["messages"], False)
        prompts.append(tokenizer.encode(text, return_tensors="pt").to(local_rank))

    @torch.inference_mode()
    def generate(ids, max_tokens):
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        state = model(ids, use_cache=True)
        token = state.logits[:, -1].argmax(-1)
        cache = state.past_key_values
        torch.cuda.synchronize()
        first_ready = time.perf_counter()
        tokens = [int(token.item())]
        for _ in range(max_tokens - 1):
            if tokens[-1] in eos:
                break
            state = model(token[:, None], past_key_values=cache, use_cache=True)
            cache = state.past_key_values
            token = state.logits[:, -1].argmax(-1)
            tokens.append(int(token.item()))
        torch.cuda.synchronize()
        finished = time.perf_counter()
        local = {"e2e_s": finished - start, "decode_s": finished - first_ready,
                 "prefill_first_token_s": first_ready - start,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        all_ranks = [None] * world
        if world > 1:
            dist.all_gather_object(all_ranks, local)
        else:
            all_ranks[0] = local
        return {"output_tokens": len(tokens), "decode_tokens": max(0, len(tokens) - 1),
                "output_token_ids": tokens, "input_tokens": ids.shape[1],
                "e2e_s": max(x["e2e_s"] for x in all_ranks),
                "decode_s": max(x["decode_s"] for x in all_ranks), "per_device": all_ranks}

    if not prompts:
        raise ValueError("The evaluation dataset is empty")
    for _ in range(args.warmup_prompts):
        generate(prompts[0], args.warmup_new_tokens)
    results = []
    for i, ids in enumerate(prompts):
        result = generate(ids, args.max_new_tokens)
        result["sample_index"] = i
        result["prompt_token_ids"] = ids.tolist()[0]
        results.append(result)
        if rank == 0:
            print(json.dumps({k: result[k] for k in ["sample_index", "output_tokens", "e2e_s", "decode_s"]}), flush=True)
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        decode_s = sum(row["decode_s"] for row in results)
        payload = {"target": args.target, "tp": args.tp, "hardware": hardware, "dataset": args.dataset,
            "dataset_source": source, "torch": torch.__version__, "cuda": torch.version.cuda,
            "dtype": "bf16", "attention_implementation": args.attn_implementation,
            "temperature": 0.0, "max_new_tokens": args.max_new_tokens,
            "decode_tps": sum(row["decode_tokens"] for row in results) / decode_s if decode_s else None,
            "per_prompt": results}
        output.write_text(json.dumps(payload, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
