"""CPU-only checks for final-checkpoint delivery; all fixtures are temporary."""
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts.rebuttal.export_h800 import checkpoint_for, prepare_artifacts


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='d3d-delivery-check-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifact = dict(id='model', job='job', group='llama', method='d3d', target='target')
        self.output = self.root / 'outputs/rebuttal/job'
        self.output.mkdir(parents=True)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def checkpoint(self, directory, depth=24):
        directory.mkdir(parents=True, exist_ok=True)
        self.write(directory / 'config.json', {'num_hidden_layers': depth})
        (directory / 'model.safetensors').write_bytes(b'temporary test weights')
        self.write(directory / 'tokenizer.json', {})
        return directory

    def completed_training(self):
        self.write(self.output / 'completion.json', {'returncode': 0})
        self.write(self.output / 'experiment.json', {'training': {'epochs': 3, 'global_batch_size': 16},
                                                    'data_info': {'train_rows': 32}})
        self.write(self.output / 'resume/progress.json', {'micro_step': 24, 'optimizer_step': 6})

    def test_running_checkpoint_is_not_exported(self):
        self.checkpoint(self.output / 'step-24')
        self.assertIsNone(checkpoint_for(self.artifact, self.root))

    def test_failed_or_shortened_training_is_not_a_final_result(self):
        self.completed_training()
        self.write(self.output / 'completion.json', {'returncode': 1})
        self.assertIsNone(checkpoint_for(self.artifact, self.root))
        self.write(self.output / 'completion.json', {'returncode': 0})
        self.write(self.output / 'resume/progress.json', {'micro_step': 20, 'optimizer_step': 5})
        with self.assertRaises(ValueError):
            checkpoint_for(self.artifact, self.root)

    def test_final_progress_selects_checkpoint_and_omits_optimizer(self):
        self.completed_training()
        checkpoint = self.checkpoint(self.output / 'step-24')
        self.checkpoint(self.output / 'step-999')
        (checkpoint / 'optimizer.bin').write_bytes(b'not an inference artifact')
        (checkpoint / 'training_config.json').write_text('{}')
        destination = self.root / 'delivery'
        result = prepare_artifacts({'artifacts': [self.artifact]}, self.root, destination, execute=True)
        self.assertEqual(result['model']['state'], 'ready')
        exported = destination / 'model'
        self.assertEqual((exported / 'model.safetensors').stat().st_ino,
                         (checkpoint / 'model.safetensors').stat().st_ino)
        self.assertFalse((exported / 'optimizer.bin').exists())
        self.assertFalse((exported / 'training_config.json').exists())
        self.assertTrue((exported / 'README.md').exists())

    def test_pard_uses_epoch_three_and_does_not_ship_teacher_projection(self):
        self.completed_training()
        self.artifact['method'] = 'pard2'
        checkpoint = self.checkpoint(self.output / 'epoch-3')
        (checkpoint / 'target_projection.safetensors').write_bytes(b'train-only')
        result = prepare_artifacts({'artifacts': [self.artifact]}, self.root, self.root / 'delivery', execute=True)
        self.assertEqual(result['model']['source'], str(checkpoint))
        self.assertFalse((self.root / 'delivery/model/target_projection.safetensors').exists())

    def test_pruning_uses_selected_step_not_latest_or_host_absolute_path(self):
        self.artifact.update(stage=2, depth=34)
        stage = self.output / 'stage-02'
        self.write(stage / 'recovery/training_summary.json', {'optimizer_steps': 10000})
        self.write(stage / 'chosen/selected_checkpoint.json',
                   {'selected_step': 500, 'selected_checkpoint': '/old/host/step-500'})
        selected = self.checkpoint(stage / 'recovery/step-500', depth=34)
        self.checkpoint(stage / 'recovery/step-10000', depth=34)
        self.assertEqual(checkpoint_for(self.artifact, self.root), selected)

    def test_catalog_and_inference_suite_match(self):
        root = Path(__file__).resolve().parents[2] / 'configs/rebuttal/h800'
        catalog = yaml.safe_load((root / 'delivery.yaml').read_text())['artifacts']
        cases = yaml.safe_load((root / 'run_delivery.yaml').read_text())['cases']
        self.assertEqual(len(catalog), 10)
        self.assertEqual(len(cases), 20)
        self.assertEqual({x['id'] for x in catalog}, {x['drafter'] for x in cases})
        self.assertTrue(all(x['method'] == 'd3d' for x in cases))
        self.assertFalse(any('*' in x['drafter'] for x in cases))


if __name__ == '__main__':
    unittest.main()
