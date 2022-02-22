#!/bin/bash
# Job name:
#SBATCH --job-name=poet_dataset_collector
#
# Project:
#SBATCH --account=nn9740k
#
# Wall clock limit (hh:mm:ss):
#SBATCH --time=168:00:00
#
## Allocates 20 cpus
#SBATCH --ntasks=1 --cpus-per-task=20
#
## allocates 1 GB ram per cpu
#SBATCH --mem-per-cpu=1G

## Set up job environment:
module --quiet purge   # clear any inherited modules
set -o errexit # exit on errors
set -o nounset # treat unset variables as error

## Load python (make sure you load the same python version on saga before creating your virtual environment!!!!!!!!)
module load Python/3.6.6-foss-2018b
export PS1=\$
source child_poet_virtualenv/bin/activate

## Copy input files to the work directory:
## copy a file
#cp $SUBMITDIR/poet_test.py $SCRATCH

## copy a directory
cp -r $SUBMITDIR/child_poet $SCRATCH
cp -r $SUBMITDIR/img $SCRATCH
cp -r $SUBMITDIR/tmp $SCRATCH

## Go to your work folder
cd $SCRATCH

## Use this to save files generated by the program back to your home area
savefile img
savefile tmp

## Run your program
if [ -z "$1" ]
then
    echo "Missing an experiment id"
    exit 1
fi

experiment=poet_$1

mkdir -p $SCRATCH/tmp/logs/$experiment
mkdir -p $SCRATCH/tmp/niche_encodings/$experiment
mkdir -p $SCRATCH/img/$experiment

cd $SCRATCH/child_poet

srun python -u dataset_collector_master_script.py \
  $SCRATCH/tmp/logs/$experiment \
  $SCRATCH/tmp/niche_encodings/$experiment \
  $SCRATCH/img/$experiment \
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
  --adjust_interval=1 \
  --propose_with_adam \
  --steps_before_transfer=25 \
  --max_children=16 \
  --max_admitted=3 \
  --num_workers 20 \
  --n_iterations=60000 2>&1 | tee ~/tmp/ipp/$experiment/run.log

## Exit 
exit 0
