# Modify these
export NUM_REPLICAS=2;
export NAME=${USER}-pygrain-test-0802-1;

# Keep these
export OUTPUT_DIR=gs://hns-largescale-axlearn-testing/$USER/$NAME
export DATASET_BUCKET=gs://tess-apple-southamerica-west1
export GKE_CLUSTER=$(axlearn gcp config | grep gke_cluster | awk '{ print $3 }' | tr -d '"')
export PROJECT_ID=$(gcloud config get project)
export BASTION_TIER=disabled

axlearn gcp bundle --name=${NAME} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=image=tpu \
        --bundler_spec=target=tpu;

axlearn gcp launch run --cluster=${GKE_CLUSTER} \
        --runner_name=gke_tpu_single \
        --name=${NAME} \
        --instance_type=tpu-v6e-256 \
        --max_tries=100 \
        --priority_class=very-high \
        --service_account=axlearn-scale-testing \
        --num_replicas=$NUM_REPLICAS \
	    --gcsfuse_mount_spec=gcs_path=${DATASET_BUCKET}/tensorflow_datasets,mount_path=/tmp/tensorflow_datasets \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "patch /opt/venv/lib/python3.10/site-packages/jax/experimental/shard_map.py -p0 < patches/shard_map.py.patch; python3 -m axlearn.common.launch_trainer_main" \
          --module=text.gpt.c4_trainer \
          --config=fuji-150B-v2-flash-grain \
          --trainer_dir=${OUTPUT_DIR} \
          --data_dir=${DATA_DIR}  \
          --jax_backend=tpu \
          --trace_at_steps=99,100,101,109,110 \
          --recorder_type=axlearn.cloud.gcp.measurement:goodput \
          --recorder_spec=name=goodput_$NAME \
          --recorder_spec=upload_dir=$OUTPUT_DIR/summaries \
          --recorder_spec=upload_interval=30 \
          --recorder_spec=rolling_window_size=3600,7200,10800,86400 \
          --mesh_selector=tpu-v6e-256