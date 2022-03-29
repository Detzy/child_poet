#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing an experiment id"
    exit 1
fi

experiment=poet_$1

mkdir -p ~/tmp/ipp/$experiment
mkdir -p ~/tmp/logs/$experiment
mkdir -p ~/tmp/niche_encodings/$experiment

python -u child_poet_master_script.py \
  ~/tmp/logs/$experiment \
  ~/tmp/niche_encodings/$experiment \
  --init=random \
  --learning_rate=0.01 \
  --lr_decay=0.9999 \
  --lr_limit=0.001 \
  --batch_size=1 \
  --batches_per_chunk=256 \
  --eval_batch_size=1 \
  --eval_batches_per_step=5 \
  --master_seed=24582922 \
  --repro_threshold=200 \
  --mc_lower=25 \
  --mc_upper=340 \
  --noise_std=0.1 \
  --noise_decay=0.999 \
  --noise_limit=0.01 \
  --normalize_grads_by_noise_std \
  --returns_normalization=centered_ranks \
  --envs stump pit roughness \
  --max_num_envs=40 \
  --adjust_interval=2 \
  --propose_with_adam \
  --steps_before_transfer=15 \
  --max_children=8 \
  --max_admitted=1 \
  --num_workers 2 \
  --run_child_poet \
  --child_success_reward=0.5 \
  --agent_tracker_certainty_threshold=0.8 \
  --start_from=/uio/hume/student-u31/eirikolb/tmp/logs/poet_dec2_168h/poet_dec2_168h.flat.best.json \
  --n_iterations=60000 2>&1 | tee ~/tmp/ipp/$experiment/run.log
#  --omit_simulation \
