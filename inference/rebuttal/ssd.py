"""Run official SSD with existing AR weights and the paper's common prompts.

SSD is an inference method; this entry does not train a model. Run it inside
an environment with the official https://github.com/tanishqkumar/ssd installed.
Its optimized backend is recorded separately from the Transformers comparison.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time


class StopTokens:
    def __init__(self, ids):
        self.ids = set(ids)

    def __eq__(self, token):
        return token in self.ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssd-root', required=True)
    parser.add_argument('--mode', choices=['async', 'target-tp1', 'target-tp2'], default='async')
    parser.add_argument('--target', required=True)
    parser.add_argument('--drafter')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--lookahead', type=int, default=15)
    parser.add_argument('--fan-out', type=int, default=2)
    parser.add_argument('--max-model-len', type=int, default=16384)
    parser.add_argument('--warmup-repeats', type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.ssd_root).resolve()))
    import torch
    from ssd import LLM, SamplingParams
    from transformers import AutoTokenizer
    from common.data import load_rows, build_prompt
    from common.text import apply_chat_template, eos_token_ids

    if any('H800' not in torch.cuda.get_device_name(i) for i in [0, 1]):
        raise RuntimeError('Formal SSD evaluation uses two visible H800 GPUs')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    rows, data_config, source = load_rows(args.dataset, args.max_samples)
    prompts = [tokenizer.encode(apply_chat_template(tokenizer,
        build_prompt(data_config['question'](row), False)['messages'], False)) for row in rows]
    speculative = args.mode == 'async'

    class BenchmarkLLM(LLM):
        def exit(self, hard=True):
            # Scope shutdown to this engine's own children. The upstream exit
            # implementation also unlinks unrelated /dev/shm semaphores.
            if self._exiting:
                return
            self._exiting = True
            children = list(self.ps)
            if speculative and self.draft_ps is not None:
                children.append(self.draft_ps)
            for child in children:
                if child.is_alive():
                    child.terminate()
                child.join()

    engine = BenchmarkLLM(args.target, draft=args.drafter,
        num_gpus=1 if args.mode == 'target-tp1' else 2,
        speculate=speculative, draft_async=speculative,
        speculate_k=args.lookahead, async_fan_out=args.fan_out,
        jit_speculate=True, sampler_x=None, enforce_eager=False,
        max_num_seqs=1, max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len, kvcache_block_size=256)
    engine.scheduler.eos = StopTokens(eos_token_ids(tokenizer, None, model_path=args.target))
    engine.log_metrics = lambda: None
    # Return token IDs; text decoding is outside every timed comparison.
    engine.tokenizer.decode = lambda *a, **kw: ''

    def generate(ids, limit):
        engine.scheduler.block_manager.hash_to_block_id.clear()
        if speculative:
            engine.scheduler.draft_block_manager.hash_to_block_id.clear()
        torch.cuda.synchronize(0)
        start = time.perf_counter()
        outputs, metrics = engine.generate([ids], SamplingParams(temperature=0.0,
            max_new_tokens=limit, ignore_eos=False), use_tqdm=False)
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start
        tokens = outputs[0]['token_ids']
        return {'prompt_token_ids': ids, 'output_token_ids': tokens,
            'input_tokens': len(ids), 'output_tokens': len(tokens),
            'decode_tokens': max(0, len(tokens) - 1), 'e2e_s': elapsed,
            'decode_s': float(metrics['decode_total_time']),
            'ssd_metrics': copy.deepcopy(metrics)}

    try:
        for _ in range(args.warmup_repeats):
            generate(prompts[0], 128)
        results = [generate(ids, args.max_new_tokens) for ids in prompts]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'backend': 'official SSD optimized engine',
            'hardware': [torch.cuda.get_device_name(i) for i in [0, 1]],
            'mode': args.mode, 'target': args.target, 'drafter': args.drafter,
            'dataset': args.dataset, 'dataset_source': source,
            'configuration': vars(args), 'training_required': False,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'per_prompt': results}, indent=2) + '\n')
    finally:
        engine.exit(hard=False)


if __name__ == '__main__':
    main()
