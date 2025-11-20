"""Tests for emulation_trainer."""
import sys
import time

import jax
from absl import flags, logging
from absl.testing import absltest

from axlearn.common import emulation_trainer, test_utils
from axlearn.common.base_model import BaseModel
from axlearn.common.config import config_class
from axlearn.common.input_base import Input
from axlearn.common.test_utils import TestCase


class MockInput(Input):
    """A mock input."""

    @config_class
    class Config(Input.Config):
        pass

    def dispatch_global_batch(self, global_physical_batch):
        return global_physical_batch

    def dataset(self):
        while True:
            yield {"input": jax.numpy.zeros((1, 1))}

    def element_spec(self):
        return {"input": jax.ShapeDtypeStruct((1, 1), jax.numpy.float32)}


class SleepTrainerTest(TestCase):
    """Tests SleepTrainer."""

    def test_sleep_trainer(self):
        # Mock config
        base_cfg = test_utils.mock_trainer_config(
            input_config=MockInput.default_config(),
            model_config=BaseModel.default_config().set(name="model", dtype=jax.numpy.float32),
        )
        cfg = emulation_trainer.SleepTrainer.default_config()
        cfg.input = base_cfg.input
        cfg.learner = base_cfg.learner
        cfg.model = base_cfg.model
        cfg.mesh_shape = base_cfg.mesh_shape
        cfg.mesh_axis_names = base_cfg.mesh_axis_names
        cfg.dir = self.create_tempdir().full_path
        cfg.sleep_seconds = 1.0
        cfg.name = "test_sleep_trainer"

        # Mock trainer
        trainer = cfg.instantiate(parent=None)

        # Create dummy state matching specs
        def create_dummy(spec):
            return jax.numpy.zeros(spec.shape, spec.dtype)

        # pylint: disable=protected-access
        trainer._trainer_state = jax.tree.map(create_dummy, trainer._trainer_state_specs)
        trainer._step = 0

        # Run a step and measure time
        start = time.time()
        trainer._run_step({"input": jax.numpy.zeros((1, 1))})
        # pylint: enable=protected-access
        end = time.time()

        duration = end - start
        logging.info("Step duration: %s", duration)

        # Should be at least 1 second
        assert duration >= 1.0


if __name__ == "__main__":
    flags.FLAGS(sys.argv)
    absltest.main()
