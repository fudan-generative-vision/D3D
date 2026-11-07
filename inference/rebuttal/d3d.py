"""H800 D3D sync/async/serialized-packed evaluation with common timing semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["sync", "async", "serialized-packed"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--drafter-kind", choices=["d3d", "pard2"], default="d3d")
    parser.add_argument("--warmup-repeats", type=int, default=8)
    parser.add_argument("--warmup-new-tokens", type=int, default=128)
    args, remainder = parser.parse_known_args()
    import torch
    from inference.native.single_request import eval_sync as sync
    from inference.native.single_request import eval_async_packed as asynchronous
    from common.metrics import summarize_timing_metrics

    if torch.cuda.device_count() < 2 or any("H800" not in torch.cuda.get_device_name(i) for i in [0, 1]):
        raise RuntimeError("Formal D3D inference requires two visible H800 GPUs")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    backend = sync if args.mode == "sync" else asynchronous
    native_args = backend.build_arg_parser().parse_args(remainder)
    native_args.output_json = None
    native_args.summary_json = None
    native_args.use_kvcache = True
    if args.drafter_kind == "pard2":
        from inference.rebuttal.pard import PardIndependentForD3D
        from inference.model.configuration_d3d import D3DConfig
        native_args.model_type = "d3d"
        native_loader = sync._load_model
        def load_pard(model_path, dtype, *, attn_implementation):
            if Path(model_path).resolve() == Path(native_args.drafter).resolve():
                configuration = D3DConfig.from_pretrained(model_path)
                return PardIndependentForD3D.from_pretrained(model_path, config=configuration,
                    torch_dtype=dtype, attn_implementation="sdpa")
            return native_loader(model_path, dtype, attn_implementation=attn_implementation)
        sync._load_model = load_pard
    if args.mode != "sync":
        native_args.warmup_new_tokens = 0
        native_args.baseline = False
    if args.mode == "serialized-packed":
        class SerializedVerifier(asynchronous.VerifierProcessClient):
            def submit_verify(self, *a, **kw):
                elapsed = super().submit_verify(*a, **kw)
                self._completed_verification = super().receive_verify()
                return elapsed

            def receive_verify(self):
                result = self._completed_verification
                self._completed_verification = None
                return result
        asynchronous.VerifierProcessClient = SerializedVerifier
    function_name = "run_d3d_generate_cached" if args.mode == "sync" else "run_d3d_generate_async"
    original = getattr(backend, function_name)
    records = []
    warmed = False

    def instrumented(*positional, **kwargs):
        nonlocal warmed
        if args.mode != "sync":
            kwargs["collect_round_timings"] = bool(native_args.collect_stage_timings)
        if not warmed:
            warm_args = list(positional)
            warm_args[3] = args.warmup_new_tokens
            for _ in range(args.warmup_repeats):
                original(*warm_args, **kwargs)
            warmed = True
        local_devices = [native_args.draft_device] if args.mode != "sync" else [native_args.verifier_device, native_args.draft_device]
        for device in local_devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        client = positional[1] if args.mode != "sync" else None
        if client is not None:
            client.memory_stats(reset=True)
        start = time.perf_counter()
        stats = original(*positional, **kwargs)
        for device in local_devices:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        memory = {device: {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)} for device in local_devices}
        if client is not None:
            memory[native_args.verifier_device] = client.memory_stats()
        ids = stats.output_ids[0, stats.num_input_tokens:].detach().cpu().tolist()
        records.append({"e2e_s": elapsed, "decode_tokens": max(0, int(stats.num_output_tokens) - 1),
            "output_token_ids": ids, "per_device_memory": memory,
            "prompt_token_ids": positional[2].detach().cpu().tolist()[0]})
        return stats

    setattr(backend, function_name, instrumented)
    try:
        payload = backend.run(native_args)
    finally:
        setattr(backend, function_name, original)
    if len(records) != len(payload["prompts"]):
        raise RuntimeError("Instrumentation sample count mismatch")
    for row, measurement in zip(payload["prompts"], records, strict=True):
        row.update(measurement)
        seconds = row["decode_elapsed_s"]
        row["tokens_per_s"] = row["decode_tokens"] / seconds if seconds else 0.0
        row["time_per_output_token"] = seconds / row["decode_tokens"] if row["decode_tokens"] else 0.0
    payload["summary"].update(summarize_timing_metrics(payload["prompts"], elapsed_s=sum(r["e2e_s"] for r in records)))
    payload["summary"]["total_e2e_s"] = sum(r["e2e_s"] for r in records)
    payload["summary"]["mode"] = args.mode
    payload["summary"]["drafter_kind"] = args.drafter_kind
    payload["summary"]["timing_scope"] = "E2E includes prefill and first token; decode excludes both; model loading and tokenization excluded"
    payload["summary"]["warmup_repetitions"] = args.warmup_repeats
    payload["summary"]["warmup_prompt_policy"] = "first prompt, fresh KV each repetition"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
