#!/bin/sh
#SBATCH --job-name="gpipd-four-room-job"
#SBATCH --account="ksandor"
#SBATCH --partition="general"      # Request partition.
#SBATCH --time=01:00:00            # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1.                 # Request 1 node
#SBATCH --tasks-per-node=1         # Set one task per node
#SBATCH --cpus-per-task=2          # Request number of CPUs (threads) per task.
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=4GB                  # Request 4 GB of RAM in total
#SBATCH --mail-type=END            # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=slurm-%x-%j.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err    # Set name of error log. %j is the Slurm jobId

export APPTAINER_IMAGE="\tudelft.net\staff-umbrella\eccmorl\essentially-contested-morl\run_code\gpipd_four_room.sif"

# Run script
srun apptainer exec \
  --nv \
  --env-file ~/.env \                 
  -B $HOME:$HOME \
  -B /tudelft.net/staff-umbrella/eccmorl/:/project-storage/ \
  $APPTAINER_IMAGE \
  uv run python run_code/gpipd_four_room.py

# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /$HOME:/$HOME/ mounts host file-sytem inside container
# The home folder should be mounted by default, but sometimes it is not
# -B can be used several times, change this to match your cluster file-system
# APPTAINER_IMAGE is the full path to the container.sif file
# python script.py is the command that you want to use inside the container
