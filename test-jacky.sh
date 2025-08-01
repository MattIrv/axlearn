# Use this branch exactly - https://github.com/SujeethJinesh/axlearn/pull/1

export NUM_REPLICAS=128;

export BASTION_TIER=disabled

export CLUSTER=bodaborg-v6e-256;

export NAME=jackyf-128-orbax-old-2 && export OUTPUT_DIR=gs://tess-checkpoints-flat-us-east5/jackyf/$NAME && export DATA_DIR=gs://tess-dataloading-us-east5/tensorflow_datasets;

axlearn gcp bundle --name=${NAME} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=image=tpu \
        --bundler_spec=target=tpu;

axlearn gcp launch run --cluster=${CLUSTER} \
        --runner_name=gke_tpu_single \
        --name=${NAME} \
        --instance_type=tpu-v6e-256 \
        --max_tries=100 \
        --queue=multislice-queue \
        --priority_class=very-high \
        --service_account=axlearn-scale-testing \
        --num_replicas=$NUM_REPLICAS \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "ulimit -n 1048576; ulimit -c 0; patch /opt/venv/lib/python3.10/site-packages/jax/experimental/shard_map.py -p0 < patches/shard_map.py.patch; python3 -m axlearn.common.launch_trainer_main \
          --module=text.gpt.c4_trainer \
          --config=fuji-150B-v2-flash \
          --trainer_dir=${OUTPUT_DIR} \
          --data_dir=${DATA_DIR}  \
          --jax_backend=tpu \
          --trace_at_steps=99,100,101,109,110 \
          --recorder_type=axlearn.cloud.gcp.measurement:goodput \
          --recorder_spec=name=goodput_$NAME \
          --recorder_spec=upload_dir=$OUTPUT_DIR/summaries \
          --recorder_spec=upload_interval=30 \
          --recorder_spec=rolling_window_size=3600,7200,10800,86400"
          
