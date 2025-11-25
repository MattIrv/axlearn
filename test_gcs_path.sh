   mkdir -p /tmp/test_sleep_trainer && \
   XLA_FLAGS="--xla_force_host_platform_device_count=8" python3 -m axlearn.common.launch_trainer_main \
       --module=text.gpt.c4_trainer \
       --config=fuji-test-v1-sleep-grain \
       --trainer_dir="gs://mirvine-benchmark-central1-hns/nov24-1" \
       --mesh_selector=tpu-v4-8 \
       --jax_backend=cpu \
       --data_dir="gs://mirvine-datasets-usc1-hns/array-record" \
       --max_step=100
