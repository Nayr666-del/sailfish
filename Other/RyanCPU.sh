#SBATCH -J hxiong_test
#SBATCH -o hxiong1.o%j
#SBATCH -e hxiong1.e%j
#SBATCH -p spr
#SBATCH -N 1                    # Number of nodes
#SBATCH -n 1                    # Number of tasks (just 1 for OpenMP)
#SBATCH -c 8                  # CPUs per task (16 threads)
#SBATCH -t 00:30:00             # Walltime


module load python

source ~/RyanKick_env/bin/activate

python ~/sailfish/bin/sailfish cool-inspiral   -o /scratch/10824/hxiong1/Output  -p -n 2000 -c 1 -t 0.01 -e 30   --cfl 0.2   --fold 100   --new-timestep-cadence 1   --model central_mass_msun=1e6     domain_radius=10. retrograde=False alpha=0.1 sink_model="torque_free" sink_rate=1. sink_radius=0.04 OpticalDepthFloor=1.0    softening_length=0.04 init_eccentricity=0.4 init_separation_rg=50. GM=1. mass_ratio=1. target_accretion_rate=1. integration_timestep=0.01 which_diagnostics="david" beta=1. mach_number_a=10 inspiral_start_time=5. eos="gamma-law"