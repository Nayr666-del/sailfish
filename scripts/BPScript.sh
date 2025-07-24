#!/bin/bash
#SBATCH -J ProcessArray
#SBATCH -o out_%A_%a.log
#SBATCH -e err_%A_%a.log
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 110
#SBATCH --array=0-20  # adjust based on number of files

# Define base directories
BASE_INPUT_DIR="/scratch/10824/hxiong1/Runs/KeplerReferenceZoom"
BASE_FOLDER="/scratch/10824/hxiong1/Runs/ZoomedRuns"

# Get the task ID assigned by SLURM
TASK_ID=${SLURM_ARRAY_TASK_ID}
PARAM=$(( (TASK_ID + 1) * 5 ))

# Create the folder for this task
FOLDER="${BASE_FOLDER}/${TASK_ID}"
mkdir -p "$FOLDER"

mv "${BASE_INPUT_DIR}/chkpt.$(printf '%04d' $TASK_ID)_zoomed.pk" "$FOLDER/"
mv "$FOLDER/chkpt.$(printf "%04d" $TASK_ID)_zoomed.pk" "$FOLDER/chkpt.$(printf '%04d' $TASK_ID).pk"

# Load environment
module load python
export OMP_NUM_THREADS=110
source ~/RyanKick_env/bin/activate

# Run python script with this folder
python ~/sailfish/bin/sailfish "$FOLDER" \
  -o "$FOLDER" \
  -p --restart-dir -n 1000 -c 0.5 -t 0.01 -e "$PARAM" \
  --cfl 0.2 \
  --fold 100 \
  --new-timestep-cadence 1 \
  --model central_mass_msun=1e6 \
    domain_radius=5. retrograde=False alpha=0.1 sink_model="torque_free" sink_rate=1. sink_radius=0.04 vkick=0.0 epsilon=0.0 OpticalDepthFloor=1.0\
    softening_length=0.04 init_eccentricity=0.0 init_separation_rg=50. GM=1. mass_ratio=1. target_accretion_rate=1.\
    integration_timestep=0.005 which_diagnostics="david" beta=1. mach_number_a=10 kick_start_time=1000.0 eos="gamma-law" buffer_is_enabled=False
