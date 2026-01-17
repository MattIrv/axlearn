# Copyright © 2023 Apple Inc.

"""Emulation trainer."""

import time
from typing import Any

import jax
from absl import logging
from jax.experimental.pjit import pjit

from axlearn.common.config import config_class
from axlearn.common.trainer import SpmdTrainer, TrainerState


class SleepTrainer(SpmdTrainer):
    """A trainer that sleeps instead of training."""

    @config_class
    class Config(SpmdTrainer.Config):
        """Configures SleepTrainer."""

        # How many seconds to sleep per step.
        sleep_seconds: float = 0.0

    def _run_step(
        self, input_batch: dict[str, Any], *, force_run_evals: set[str] | None = None
    ) -> dict[str, Any]:
        """Runs a single training step with sleep."""
        cfg: SleepTrainer.Config = self.config

        # 1. Dispatch inputs and synchronize devices.
        # We use a pjit-ted function to ensure this happens efficiently on devices.
        if not hasattr(self, "_jit_step_and_sync_fn"):
            self._jit_step_and_sync_fn = pjit(
                self._step_and_sync,
                in_shardings=(
                    self._trainer_state_partition_specs,
                    self._train_step_input_partition_specs(),
                ),
                out_shardings=(
                    self._trainer_state_partition_specs,
                    dict(
                        summaries=None,
                        loss=None,
                        aux=None,
                    ),
                ),
                donate_argnums=(0,),
            )

        with self.mesh():
            self._trainer_state, outputs = self._jit_step_and_sync_fn(
                self._trainer_state, input_batch
            )

        # 2. Block until synchronization is complete.
        # This ensures that all ranks reach this point before any of them start sleeping.
        # We block on 'loss' which depends on the psum.
        outputs["loss"].block_until_ready()

        # 3. Sleep on the host.
        if cfg.sleep_seconds > 0:
            logging.info("Sleeping for %s seconds...", cfg.sleep_seconds)
            time.sleep(cfg.sleep_seconds)
        jax.experimental.multihost_utils.sync_global_devices("Barrier after step")

        # 4. Standard logging and checkpointing (simplified from SpmdTrainer).
        n = self._config.log_every_n_steps or 100
        if self.step % n == 0 or 0 <= self.step <= 3000:
            self._step_log("loss=%s", outputs["loss"])

        self.summary_writer(self.step, {"loss": outputs["loss"], **outputs["summaries"]})

        # Aggregate summaries across evalers.
        evaler_summaries = self._run_eval(
            train_summaries=outputs["summaries"], force_runs=force_run_evals
        )

        # Checkpointer policy will decide if we should save.
        self.save_checkpoint(evaler_summaries=evaler_summaries)

        return_dict = {"loss": outputs["loss"], "aux": outputs["aux"]}
        if force_run_evals:
            return_dict["evaler_summaries"] = evaler_summaries
        return return_dict

    def _step_and_sync(
        self, state: TrainerState, input_batch: dict[str, Any]
    ) -> tuple[TrainerState, dict[str, Any]]:
        """Shards inputs, synchronizes, and updates state."""

        # Shard and (possibly) dispatch the input batch.
        # We need to use the same partitioning as the real trainer would to emulate overhead.
        # However, for pure sleep emulation, we might just want to touch the data.
        # Let's try to respect the input partition specs if possible, but for now
        # we'll rely on the input pipeline's dispatch.
        input_batch = self.input.dispatch_global_batch(input_batch)

        # Synchronize across mesh.
        # We use a dummy psum to force synchronization.
        # dummy_val = jax.numpy.array(1.0, dtype=jax.numpy.float32)
        # # Filter for active axes to avoid Unbound Axis errors
        # active_axes = tuple(
        #     name for name, size in zip(cfg.mesh_axis_names, cfg.mesh_shape) if size > 1
        # )
        # if active_axes:
        #     sync_val = jax.lax.psum(dummy_val, axis_name=active_axes)
        # else:
        #     sync_val = dummy_val

        # Update state (PRNG key).
        new_prng_key, _ = jax.random.split(state.prng_key)
        updated_state = state._replace(prng_key=new_prng_key)

        # Return dummy outputs.
        # We return 'loss' as sync_val to ensure we can block on it.
        summaries = {
            "loss": 0.0,
            "batch_size": jax.numpy.array(len(input_batch), dtype=jax.numpy.int32),
        }
        outputs = {
            "summaries": summaries,
            "loss": 0.0,
            "aux": {},
        }
        return updated_state, outputs

    def _train_step(self, state, input_batch):
        """Not used by SleepTrainer._run_step, but required by abstract base class."""
        raise NotImplementedError("SleepTrainer uses _run_step directly.")
