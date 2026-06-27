#!/bin/sh
#SBATCH --job-name="ff-baselines"
#SBATCH --account="ksandor"
#SBATCH --partition="general"      # Request partition.
#SBATCH --time=04:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                  # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=2          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=8GB                  # Request 8 GB of RAM in total
#SBATCH --mail-type=END            # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --array=0-7                # One array task per config below
#SBATCH --output=slurm-%x-%a-%j.out   # %x=job-name, %a=array idx, %j=jobId
#SBATCH --error=slurm-%x-%a-%j.err

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/eccmorl/essentially-contested-morl/image.sif"

# The 8 firefighters baseline configs (envelope + gpipd x 4 objective setups).
# SLURM_ARRAY_TASK_ID picks one per array task.
CONFIGS="\
scripts/config/firefighters_envelope_4obj.json \
scripts/config/firefighters_envelope_rescue.json \
scripts/config/firefighters_envelope_fire.json \
scripts/config/firefighters_envelope_weighted.json \
scripts/config/firefighters_gpipd_4obj.json \
scripts/config/firefighters_gpipd_rescue.json \
scripts/config/firefighters_gpipd_fire.json \
scripts/config/firefighters_gpipd_weighted.json"

# Pick the config for this array index (0-based).
CONFIG=$(echo $CONFIGS | tr ' ' '\n' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
echo "[array $SLURM_ARRAY_TASK_ID] running config: $CONFIG"

# Run script
srun apptainer exec \
  --nv \
  --env-file ~/.env \
  -B $HOME:$HOME \
  -B /tudelft.net/staff-umbrella/eccmorl/ \
  --cwd /tudelft.net/staff-umbrella/eccmorl/essentially-contested-morl \
  $APPTAINER_IMAGE \
  /opt/venv/bin/python scripts/train_reach_goal.py --config "$CONFIG"

# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B mounts host file-system inside container
# APPTAINER_IMAGE is the full path to the container.sif file
# Submit with:  sbatch job_firefighters_baselines.sh
