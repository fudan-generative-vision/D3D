#!/usr/bin/env python3
"""Prepare completed inference checkpoints for the author's ModelScope upload."""
import argparse
import json
import os
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text())


def checkpoint_for(artifact, source_root):
    output = Path(source_root) / 'outputs/rebuttal' / artifact['job']
    if 'stage' in artifact:
        stage = output / f"stage-{artifact['stage']:02d}"
        selected = stage / 'chosen/selected_checkpoint.json'
        if not selected.is_file() or not (stage / 'recovery/training_summary.json').is_file():
            return None
        # The recorded absolute training-host path is not portable.
        step = int(read_json(selected)['selected_step'])
        checkpoint = stage / 'recovery' / f'step-{step}'
        if read_json(checkpoint / 'config.json')['num_hidden_layers'] != artifact['depth']:
            raise ValueError(f"Wrong pruning depth for {artifact['id']}")
        return checkpoint
    completion = output / 'completion.json'
    if not completion.is_file() or read_json(completion)['returncode'] != 0:
        return None
    experiment = read_json(output / 'experiment.json')
    training = experiment['training']
    expected = experiment['data_info']['train_rows'] * training['epochs'] // training['global_batch_size']
    progress = read_json(output / 'resume/progress.json')
    if training['epochs'] != 3 or progress['optimizer_step'] != expected:
        raise ValueError(f"Full three-epoch budget is not complete for {artifact['id']}")
    name = 'epoch-3' if artifact['method'] == 'pard2' else f"step-{progress['micro_step']}"
    return output / name


def inference_files(checkpoint):
    checkpoint = Path(checkpoint)
    files = {checkpoint / 'config.json'}
    indexes = list(checkpoint.glob('model.safetensors.index.json')) + list(checkpoint.glob('pytorch_model.bin.index.json'))
    if indexes:
        index = indexes[0]
        files.add(index)
        files.update(checkpoint / name for name in read_json(index)['weight_map'].values())
    else:
        weights = checkpoint / 'model.safetensors'
        if not weights.is_file():
            weights = checkpoint / 'pytorch_model.bin'
        files.add(weights)
    for pattern in ('generation_config.json', 'tokenizer*', 'special_tokens_map.json',
                    'added_tokens.json', 'chat_template*', 'vocab.*', 'merges.txt',
                    '*.model', '*.tiktoken', '*.py', 'LICENSE*'):
        files.update(path for path in checkpoint.glob(pattern) if path.is_file())
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(files)


def prepare_artifacts(catalog, source_root, destination_root, *, groups=None, execute=False, copy=False):
    destination_root = Path(destination_root)
    artifacts = catalog['artifacts']
    if groups:
        unknown = set(groups) - {item['group'] for item in artifacts}
        if unknown:
            raise ValueError(f'Unknown delivery groups: {sorted(unknown)}')
        artifacts = [item for item in artifacts if item['group'] in groups]
    result = {}
    for item in artifacts:
        destination = destination_root / item['id']
        if destination.is_dir():
            result[item['id']] = {'state': 'ready', 'directory': str(destination)}
            continue
        checkpoint = checkpoint_for(item, source_root)
        if checkpoint is None:
            result[item['id']] = {'state': 'waiting_for_checkpoint'}
            continue
        files = inference_files(checkpoint)
        result[item['id']] = {'state': 'available', 'source': str(checkpoint),
                              'directory': str(destination), 'bytes': sum(path.stat().st_size for path in files)}
        if not execute:
            continue
        destination_root.mkdir(parents=True, exist_ok=True)
        pending = destination_root / ('.' + item['id'] + '.pending')
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        for source in files:
            target = pending / source.name
            if not copy and source.suffix in ('.safetensors', '.bin'):
                os.link(source.resolve(), target)
            else:
                shutil.copy2(source, target)
        selection = f"development-selected checkpoint at depth {item['depth']}" if 'stage' in item else 'final checkpoint after three epochs'
        (pending / 'README.md').write_text(
            f"# {item['id']}\n\nTarget: {item['target']}. Training seed: 42.\n\n"
            f"Selection: {selection}. Source: {checkpoint.relative_to(source_root)}.\n\n"
            "Use the inference implementation on Git branch `rebuttal/h800-trained-checkpoints`.\n"
            "This directory contains inference weights and tokenizer files. Training optimizer/RNG states are kept on the training host.\n"
        )
        pending.rename(destination)
        result[item['id']]['state'] = 'ready'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', default=str(ROOT / 'configs/rebuttal/h800/delivery.yaml'))
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/rebuttal_h800_upload')
    parser.add_argument('--group', action='append')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--copy', action='store_true', help='Copy weights when exporting to another filesystem')
    args = parser.parse_args()
    catalog = yaml.safe_load(Path(args.catalog).read_text())
    result = prepare_artifacts(catalog, args.source_root.resolve(), args.output_root.resolve(),
        groups=args.group, execute=args.execute, copy=args.copy)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
