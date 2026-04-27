#!/usr/bin/env python3
"""
Generate minimal SLURM script that works on most clusters.
"""

import os

def generate_minimal_sbatch(num_jobs=90, output_file="eval_cross_dataset_minimal.sbatch"):
    """Generate a minimal SLURM script without GPU requirements."""
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=cross_eval
#SBATCH --array=0-{num_jobs-1}%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/eval_%a.log
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/eval_%a.err

# Create log directory
mkdir -p /home1/eibl/gfm/prodigy/logs

# Load environment (adjust based on your cluster)
# module load cuda/11.8  # Uncomment if needed

# Activate conda
source $HOME/.bashrc
conda activate prodigy

# Set working directory
cd /home1/eibl/gfm/prodigy

# Run job from the list
JOB_COMMANDS=(
"""
    
    # Read job commands from eval_jobs.txt
    if os.path.exists("eval_jobs.txt"):
        with open("eval_jobs.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    sbatch_content += f'  "{line}"\n'
    
    sbatch_content += f"""
)

# Get the command for this array task
COMMAND="${{JOB_COMMANDS[${{SLURM_ARRAY_TASK_ID}}]}}"

echo "Running task ${{SLURM_ARRAY_TASK_ID}}: $COMMAND"
eval "$COMMAND"
EXIT_CODE=$?

echo "Task ${{SLURM_ARRAY_TASK_ID}} completed with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
    
    with open(output_file, 'w') as f:
        f.write(sbatch_content)
    
    print(f"Generated minimal SLURM script: {output_file}")
    print(f"\nTry submitting with:")
    print(f"  sbatch {output_file}")
    print(f"\nTo monitor:")
    print(f"  squeue -u $USER")
    
    return sbatch_content


if __name__ == "__main__":
    generate_minimal_sbatch()
