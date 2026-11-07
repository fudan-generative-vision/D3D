#!/usr/bin/env python3
"""Print or execute the H800 inference queue; all model paths are portable."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.rebuttal import config as cfg


def draft_path(case, root):
    name = Path(root) / case["drafter"]
    if "*" in str(name):
        return max(name.parent.glob(name.name), key=lambda path: int(path.name.split("-")[-1]))
    return name


def make_command(case, dataset, repetition, args, suite):
    target = Path(args.models_root) / case["target"]
    output = Path(args.output_root) / case["id"] / f"{dataset}.r{repetition}.json"
    common = ["--dataset", dataset, "--max-new-tokens", str(suite["max_new_tokens"])]
    limit = args.max_samples if args.max_samples is not None else suite.get("max_samples")
    if limit is not None:
        common += ["--max-samples", str(limit)]
    if case["method"] == "target":
        command = [args.python, "-m", "torch.distributed.run", "--nproc_per_node", str(case["tp"]),
            "--master_port", str(args.master_port), "--module", "inference.rebuttal.target",
            "--target", str(target), "--tp", str(case["tp"]), "--output", str(output), *common]
    elif case["method"] == "d3d":
        draft = draft_path(case, args.checkpoints_root) if args.execute else Path(args.checkpoints_root) / case["drafter"]
        command = [args.python, "-m", "inference.rebuttal.d3d", "--mode", case["mode"],
            "--output", str(output), "--drafter-kind", case.get("drafter_kind", "d3d"), "--verifier", str(target), "--drafter", str(draft),
            "--tokenizer", str(target), "--verifier-device", "cuda:0", "--draft-device", "cuda:1",
            "--dtype", "bf16", "--temperature", "0", "--block-size", "16", "--denoising-steps", "1",
            "--attention-mask-mode", "causal", "--use-kvcache", "--target-attn-implementation", "flash_attention_2",
            "--draft-attn-implementation", "sdpa", *common]
        if case["mode"] != "sync":
            command += ["--async-blocks", "1", "--anchor-num", "2", "--packed-accepted-counts", "1,2,3,4,5,6,7,8,9,10,11,16"]
        if args.profile:
            command.append("--collect-stage-timings")
    elif case["method"] == "ssd":
        command = [args.python, "-m", "inference.rebuttal.ssd", "--ssd-root", args.ssd_root,
            "--mode", case["mode"], "--target", str(target), "--output", str(output), *common]
        if case["mode"] == "async":
            draft_root = args.models_root if case.get("drafter_source") == "models" else args.checkpoints_root
            command += ["--drafter", str(Path(draft_root) / case["drafter"]),
                "--lookahead", str(case.get("lookahead", 15)), "--fan-out", str(case.get("fan_out", 2))]
    else:
        raise ValueError(f"Unsupported H800 case method: {case['method']}")
    return command, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rebuttal/h800/run_existing.yaml")
    parser.add_argument("--models-root", required=True, help="Directory containing Qwen3-8B, Qwen3-32B, etc.")
    parser.add_argument("--checkpoints-root", required=True, help="Directory containing the named D3D checkpoints")
    parser.add_argument("--output-root", default="outputs/rebuttal_h800")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--case", action="append", help="Restrict to named cases; can be repeated")
    parser.add_argument("--group", action="append", help="Run one delivered group, such as llama or pruning_uniform")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--max-samples", type=int, help="Use small values only for smoke checks")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--master-port", type=int, default=29900)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--profile", action="store_true", help="Use a separate output root for instrumented diagnostics")
    parser.add_argument("--ssd-root", default="third_party/ssd", help="Official SSD checkout, installed in --python's environment")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    suite = cfg.load(args.config)
    if len(args.gpus.split(",")) != 2:
        parser.error("Select exactly two H800 GPUs")
    if args.group and set(args.group) - {case.get("group") for case in suite["cases"]}:
        parser.error("Unknown delivery group")
    cases = [case for case in suite["cases"]
             if (not args.case or case["id"] in args.case)
             and (not args.group or case.get("group") in args.group)]
    if not cases or (args.case and set(args.case) - {case["id"] for case in cases}):
        parser.error("Unknown or empty case selection")
    if args.execute:
        for case in cases:
            if not (Path(args.models_root) / case["target"] / "config.json").exists():
                raise FileNotFoundError(case["target"])
            if case["method"] == "d3d":
                resolved = draft_path(case, args.checkpoints_root).resolve()
                if not (resolved / "config.json").exists():
                    raise FileNotFoundError(resolved)
                # Freeze a wildcard selection once, before any repetitions begin.
                case["drafter"] = str(resolved)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": args.gpus, "OMP_NUM_THREADS": "4",
                        "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    datasets = args.dataset or suite["datasets"]
    repeats = args.repeats if args.repeats is not None else suite["repeats"]
    for repetition in range(repeats):
        # Reverse method order on alternating repetitions to reduce order bias.
        order = cases if repetition % 2 == 0 else list(reversed(cases))
        for case in order:
            for dataset in datasets:
                command, output = make_command(case, dataset, repetition, args, suite)
                print("CUDA_VISIBLE_DEVICES=" + shlex.quote(args.gpus) + " " + shlex.join(command), flush=True)
                if args.execute:
                    if output.exists():
                        raise FileExistsError(output)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.with_suffix(".log").open("w") as log:
                        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
