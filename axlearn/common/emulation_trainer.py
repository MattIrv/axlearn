# Copyright © 2023 Apple Inc.

"""Emulation trainer."""

import time
from typing import Any

import jax
from absl import logging

from axlearn.common import utils
from axlearn.common.config import config_class
from axlearn.common.trainer import SpmdTrainer, TrainerState


class SleepTrainer(SpmdTrainer):
    """A trainer that sleeps instead of training."""

    @config_class
    class Config(SpmdTrainer.Config):
        """Configures SleepTrainer."""

        # How many seconds to sleep per step.
        sleep_seconds: float = 0.0

    def _train_step(
        self,
        state: TrainerState,
        input_batch: dict[str, Any],
    ) -> tuple[TrainerState, dict[str, Any]]:
        """A train step that sleeps."""
        cfg: SleepTrainer.Config = self.config

        def sleep_callback(seconds):
            seconds = float(seconds)
            if seconds > 0:
                logging.info("Sleeping for %s seconds...", seconds)
                time.sleep(seconds)

        # Use jax.debug.callback to ensure the sleep is not optimized out.
        jax.debug.callback(sleep_callback, cfg.sleep_seconds)

        # Dispatch the input batch like the real trainer, this will synchronize appropriately.
        def train_cast(in_tree):
            per_param_train_dtype = self._per_param_train_dtype(in_tree)
            return utils.cast_floats_per_param(in_tree, per_param_train_dtype)

        # Cast before dispatching to speed up matmul and decrease memory imprint.
        input_batch = train_cast(input_batch)

        # Shard and (possibly) dispatch the input batch.
        input_batch = self.input.dispatch_global_batch(input_batch)

        dummy_val = jax.numpy.array(1.0, dtype=jax.numpy.float32)

        new_prng_key, _ = jax.random.split(state.prng_key)

        updated_state = state._replace(prng_key=new_prng_key)

        summaries = {
            "loss": jax.numpy.array(0.0, dtype=jax.numpy.float32),
            "batch_size": len(input_batch),  # Return this to ensure it's not optimized out
        }
        return updated_state, {
            "summaries": summaries,
            "loss": jax.numpy.array(0.0, dtype=jax.numpy.float32),
            "aux": {},
        }
