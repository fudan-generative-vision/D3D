#!/usr/bin/env python3
"""Compare two formal H800 outputs using paired prompts; print, do not save, results."""
import argparse
import json
from pathlib import Path
import random
import statistics


def rows(path):
    data = json.loads(Path(path).read_text())
    return data['per_prompt'] if 'per_prompt' in data else data['prompts']


def percentile(values, fraction):
    return sorted(values)[round(fraction * (len(values) - 1))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    baseline, method = rows(args.baseline), rows(args.method)
    pairs = list(zip(baseline, method, strict=True))
    if any(a['prompt_token_ids'] != b['prompt_token_ids'] for a, b in pairs):
        raise ValueError('The two runs use different prompts or order')
    mismatches = [i for i, (a, b) in enumerate(pairs) if a['output_token_ids'] != b['output_token_ids']]
    if mismatches:
        raise ValueError(f'Greedy outputs differ on {len(mismatches)} prompts, first indices {mismatches[:10]}; resolve before interpreting speedups')
    def seconds(row, metric):
        return row['e2e_s'] if metric == 'e2e' else row.get('decode_s', row.get('decode_elapsed_s'))
    result = {'prompts': len(pairs), 'greedy_outputs_equal': True}
    for metric in ['e2e', 'decode']:
        a = [seconds(x, metric) for x, _ in pairs]
        b = [seconds(x, metric) for _, x in pairs]
        randomizer = random.Random(20261106)
        sampled = []
        for _ in range(args.bootstrap):
            indices = randomizer.choices(range(len(pairs)), k=len(pairs))
            sampled.append(sum(a[i] for i in indices) / sum(b[i] for i in indices))
        result[metric] = {'baseline_total_s': sum(a), 'method_total_s': sum(b),
            'speedup': sum(a) / sum(b), 'paired_prompt_95pct': [percentile(sampled, .025), percentile(sampled, .975)]}
    rounds = [record for row in method for record in row.get('timing', {}).get('async', {}).get('round_timings', [])]
    if rounds:
        result['rounds'] = len(rounds)
        for key in ['verify_gpu_s', 'packed_draft_gpu_s', 'round_wall_s', 'wait_for_verify_s']:
            values = [row[key] for row in rounds]
            result[key] = {'mean_ms': 1000 * statistics.mean(values),
                'p50_ms': 1000 * percentile(values, .5), 'p95_ms': 1000 * percentile(values, .95)}
        eligible = [row for row in rounds if not row['terminal']]
        misses = [row for row in eligible if not row['next_branch_available']]
        result['nonterminal_cache_miss_rate'] = len(misses) / len(eligible) if eligible else None
        result['conditional_redraft_ms'] = 1000 * statistics.mean(row['redraft_wall_s'] for row in misses) if misses else 0
        result['redraft_ms_per_nonterminal_round'] = 1000 * sum(row['redraft_wall_s'] for row in misses) / len(eligible) if eligible else None
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
