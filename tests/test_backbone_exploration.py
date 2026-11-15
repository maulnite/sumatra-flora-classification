"""Guards against accidental pretraining, recipe mutation and incompatible resume."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import explore_backbones as runner


class BackboneExplorationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.yaml.safe_load((runner.ROOT / 'configs/LOCAL-PROP-007-BACKBONES.yaml').read_text())

    def test_every_model_is_explicitly_scratch_with_30_classes(self):
        with patch.object(runner.timm, 'create_model', return_value=object()) as create:
            for name, settings in self.cfg['backbones'].items():
                runner.make_model(self.cfg, name)
                self.assertEqual(create.call_args.args, (settings['architecture'],))
                self.assertIs(create.call_args.kwargs['pretrained'], False)
                self.assertEqual(create.call_args.kwargs['num_classes'], 30)
            self.cfg['training']['pretrained'] = True
            before = create.call_count
            with self.assertRaises(ValueError):
                runner.make_model(self.cfg, next(iter(self.cfg['backbones'])))
            self.assertEqual(create.call_count, before)

    def test_backbone_optimizer_configs_do_not_contaminate_each_other(self):
        before = copy.deepcopy(self.cfg)
        efficient = runner.optimizer_config(self.cfg, 'efficientnetv2_rw_t')
        convnext = runner.optimizer_config(self.cfg, 'convnext_tiny')
        self.assertEqual(self.cfg, before)
        self.assertEqual(efficient['training']['learning_rate'], .002)
        self.assertEqual(convnext['training']['learning_rate'], .0005)
        self.assertEqual(convnext['training']['warmup_epochs'], 20)
        efficient['training']['epochs'] = 1
        self.assertEqual(convnext['training']['epochs'], 600)
        self.assertEqual(self.cfg['training']['epochs'], 600)

    def test_wrong_scene_fold_and_target_training_are_rejected(self):
        with patch.object(runner.source, 'verify') as verify:
            self.cfg['fold'] = 1
            with self.assertRaisesRegex(ValueError, 'fold'):
                runner.validate_protocol(self.cfg)
            self.cfg['fold'] = 2
            self.cfg['execution']['target_images_used'] = True
            with self.assertRaisesRegex(ValueError, 'source-only'):
                runner.validate_protocol(self.cfg)
            verify.assert_not_called()

    def test_resume_rejects_checkpoint_from_another_specification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'spec.json').write_text('{}\n')
            checkpoint = root / 'paired_checkpoint.pt'
            runner.torch.save(dict(spec_sha256='different-experiment'), checkpoint)
            with self.assertRaises(AssertionError):
                runner.source.restore_paired({}, checkpoint, root)


if __name__ == '__main__':
    unittest.main()
