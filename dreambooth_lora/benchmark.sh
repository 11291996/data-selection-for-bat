#!/bin/bash

# Define gamma and sample_ratio arrays
gamma=(0.95)
sample_ratio=(0.1 0.3 0.5 0.75 1)

# Define number of GPUs
num_gpus=4

# Loop over the gamma and sample_ratio combinations
for g in "${gamma[@]}"; do
  for s in "${sample_ratio[@]}"; do
    
    # Set the output directory based on gamma and sample_ratio
    output_dir="../dsbat/models/military_pilot/military_pilot_strong_rand_gamma_${g}_sample_ratio_${s}"

    # Set the output directory to be used by benchmark.py
    for gpu_id in {0..3}; do  # Loop to assign one GPU per process
        
      # Launch the benchmark with Accelerate, specifying the GPU to use
      CUDA_VISIBLE_DEVICES=$gpu_id python benchmark.py \
        --instance="military pilot" \
        --instance_data_path="/nfs/home/jaewan/data-selection-for-bat/dsbat/datasets/military_pilot/military_pilot_instance/val" \
        --model_path="/nfs/home/jaewan/data-selection-for-bat/dsbat/models/military_pilot/military_pilot_strong_rand_gamma_${g}_sample_ratio_${s}" \
        --num_samples=1 \
        --gpu_id=0 &  # The '&' runs the command in the background for each process
      
      # Add a small delay to stagger the processes slightly for clarity, optional.
      sleep 1

    done
    
    # Wait for all background jobs to finish before starting the next combination of gamma and sample_ratio
    wait
    
  done
done