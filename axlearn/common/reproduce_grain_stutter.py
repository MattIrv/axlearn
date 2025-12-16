
"""
Reproduction script for Grain pipeline stutter.
"""
import time
import sys
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

import jax
import grain.python as grain
from axlearn.common import input_grain
from axlearn.common import input_grain_lm
from axlearn.common import utils
from axlearn.common import input_grain_text
from axlearn.common.config import config_class

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "gs://mirvine-datasets-usc1-hns/array-record", "Data directory.")
flags.DEFINE_integer("num_components", 100, "Number of data mixture components.")
flags.DEFINE_integer("vocab_size", 32000, "Vocab size.")
flags.DEFINE_integer("max_sequence_length", 2048, "Max sequence length.")
flags.DEFINE_integer("window_size", 2048, "Window size for packing.")
flags.DEFINE_integer("num_steps", 1000, "Number of steps to run.")
flags.DEFINE_float("stutter_threshold", 1.0, "Threshold in seconds to report stutter.")


class DummyVocabulary:
    """A dummy vocabulary for testing."""
    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size
    
    @property
    def pad_id(self):
        return 0

    @property
    def eos_id(self):
        return 1

    def encode(self, s: str) -> Sequence[int]:
        # Dummy encoding: just return a list of 1s of length len(s)
        # This is fast and sufficient for pipeline testing where we don't care about content.
        return [2] * min(len(s), 100) # Cap length to avoid huge processing if not needed

    def _decode(self, ids: Sequence[int]) -> str:
        return ""

    def decode(self, ids: Sequence[int]) -> str:
        return ""

class DataMixtureComponent:
    def __init__(self, name, split, shuffle_buffer_size, weight):
        self.name = name
        self.split = split
        self.shuffle_buffer_size = shuffle_buffer_size
        self.weight = weight

def main(_):
    # Setup data components
    components = [
        DataMixtureComponent(
            name="c4/en:3.0.1",
            split="train",
            shuffle_buffer_size=8192,
            weight=1.0,
        )
    ] * FLAGS.num_components

    # Setup Preprocessor
    # mimicking _convert_tf_data_to_grain_source in trainer_config_modifier.py
    # and _train_input_source in c4_trainer.py
    
    # We use config_class to mimic ConfigOr behavior expected by input_grain
    # But input_grain.mixture_train_input_source expects ConfigOr or list.
    # Actually, let's just use the direct function and config objects if needed, 
    # or better yet, use the same structure as trainer_config_modifier.
    
    # We need to properly mock ConfigOr behavior if we are not using the config system fully,
    # or just use simple objects if dynamic instantiation supports it.
    # input_grain uses maybe_instantiate.
    
    # Let's create a simple Config compatible object for preprocessor
    # The preprocessor arg in mixture_train_input_source expects a Config that instantiates 
    # to a function/transform_fn. 
    # In trainer_config_modifier it creates a config_for_function(partial_text_to_lm_training_input)
    
    # We can just define the partial function directly and wrap it in a class with .instantiate()
    # or just passed as is if maybe_instantiate handles callables (it usually doesn't, it handles Config objects).
    
    # Let's import config utils to be safe.
    from axlearn.common.config import config_for_function, config_for_class

    vocab_cfg = config_for_class(DummyVocabulary).set(vocab_size=FLAGS.vocab_size)
    
    def partial_text_to_lm_training_input(ds):
        return input_grain_lm.text_to_lm_training_input(
            ds,
            vocab=vocab_cfg,
            max_len=FLAGS.max_sequence_length,
            window_size=FLAGS.window_size,
            max_padding_fraction=0.5, # Matching c4_trainer
            read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=0), # Matching GrainConfigModifier default
            packing_fn=input_grain_lm.windowed_packing,
        )

    # In trainer_config_modifier, it sets preprocessor to a config.
    # IMPORTANT: The preprocessor config MUST NOT have 'ds' bound yet, 
    # because input_grain will inject it via partial application or simply by calling the config instantiator with ds?
    # No, input_grain calls `processor_fn = maybe_instantiate(processor_cfg)` and then `source_ds = processor_fn(source_ds)` OR `source_ds = processor_fn(mixed_ds)` if it returns a callable.
    
    # Wait, in trainer_config_modifier:
    # preprocessor = config_for_function(partial_text_to_lm_training_input).set(...)
    # partial_text_to_lm_training_input returns a LAMBDA taking ds.
    
    # So our definition of partial_text_to_lm_training_input ABOVE was wrong if we want to match trainer_config_modifier behavior exactly.
    # In trainer_config_modifier, it returns `lambda ds: input_grain_lm.text_to_lm_training_input(...)`.
    # BUT here effectively we want the same. 
    
    # However, the error was: missing value for required field 'ds'.
    # This implies that `maybe_instantiate(processor_cfg)` tried to CALL the function, but it required `ds`.
    
    # If we want `maybe_instantiate` to return a callable that TAKES `ds`, 
    # then the target function of the config should return a callable.
    
    def get_preprocessor_fn(
        vocab,
        max_len,
        window_size=128,
        max_padding_fraction=1.0,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=16),
        packing_fn=input_grain_lm.streaming_packing,
    ):
        return lambda ds: input_grain_lm.text_to_lm_training_input(
            ds,
            vocab=vocab,
            max_len=max_len,
            max_padding_fraction=max_padding_fraction,
            window_size=window_size,
            read_options=read_options,
            packing_fn=packing_fn,
        )

    preprocessor_cfg = config_for_function(get_preprocessor_fn).set(
         vocab=vocab_cfg,
         max_len=FLAGS.max_sequence_length,
         window_size=FLAGS.window_size,
         max_padding_fraction=0.5,
         read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=0),
         packing_fn=input_grain_lm.windowed_packing,
    )

    # Patch utils.get_data_dir to return our flag
    original_get_data_dir = utils.get_data_dir
    utils.get_data_dir = lambda: FLAGS.data_dir

    try:
        logging.info("Building dataset...")
        ds_builder = input_grain.mixture_train_input_source(
            is_training=True,
            vocab_cfg=vocab_cfg,
            preprocessor=preprocessor_cfg,
            data_mixture_components=components,
            max_sequence_length=FLAGS.max_sequence_length,
            enable_broadcast_instructions=True,
        )
        
        # Dispatch config
        # Simulate single host for reproduction
        read_config = dict(shard_index=[jax.process_index()], num_shards=[jax.process_count()], batch_size=16)
        ds = ds_builder(input_grain.DispatchConfig(**read_config))
        
        ds_iter = iter(ds)
        
        logging.info("Starting iteration...")
        start_time = time.time()
        for i in range(FLAGS.num_steps):
            step_start = time.time()
            _ = next(ds_iter)
            time.sleep(0.65)
            step_duration = time.time() - step_start
            
            if step_duration > FLAGS.stutter_threshold:
                logging.warning(f"Step {i}: Stutter detected: {step_duration:.4f}s")
            
            if i % 1 == 0:
                 logging.info(f"Step {i}: {step_duration:.4f}s")
                 
        total_time = time.time() - start_time
        logging.info(f"Finished {FLAGS.num_steps} steps in {total_time:.2f}s (Avg: {total_time/FLAGS.num_steps:.4f}s/step)")

    finally:
        utils.get_data_dir = original_get_data_dir

if __name__ == "__main__":
    app.run(main)
