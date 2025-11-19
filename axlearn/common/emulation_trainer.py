# Copyright © 2023 Apple Inc.

"""Emulation trainer."""

import time
from typing import Any

import jax
from absl import logging

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
        del input_batch

        def sleep_callback(seconds):
            seconds = float(seconds)
            if seconds > 0:
                logging.info("Sleeping for %s seconds...", seconds)
                time.sleep(seconds)

        jax.debug.callback(sleep_callback, cfg.sleep_seconds)

        # Return same state and dummy metrics.
        # We need to update PRNG key to simulate state update?
        # It's better to split it to avoid reusing same key if that matters.
        new_prng_key, _ = jax.random.split(state.prng_key)

        updated_state = state._replace(prng_key=new_prng_key)

        summaries = {
            "loss": jax.numpy.array(0.0, dtype=jax.numpy.float32),
        }
        return updated_state, {
            "summaries": summaries,
            "loss": jax.numpy.array(0.0, dtype=jax.numpy.float32),
            "aux": {},
        }
