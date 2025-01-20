#!/bin/bash

# Define gamma and sample_ratio arrays
gamma=(0.5 0.75 0.9 0.95 0.99)
sample_ratio=(0.1 0.3 0.5 0.75 1)

# Initialize the GPU ID to 0
gpu_id=0

# Loop over the gamma and sample_ratio combinations
for g in "${gamma[@]}"; do
  for s in "${sample_ratio[@]}"; do
    
    # Set the output directory based on gamma and sample_ratio
    output_dir="../dsbat/models/military_pilot/military_pilot_strong_rand_gamma_${g}_sample_ratio_${s}"

    # Loop through the GPUs and assign one process to each GPU
    for i in {0..3}; do
      # Launch the training with Accelerate, specifying the GPU to use
      accelerate launch --gpu_ids $i ./train_model.py \
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
        --instance_data_dir="../dsbat/datasets/military_pilot/military_pilot_instance/train" \
        --class_data_dir="../dsbat/datasets/military_pilot/military_pilot_class" \
        --output_dir=$output_dir \
        --train_text_encoder \
        --with_prior_preservation --prior_loss_weight=1.0 \
        --num_dataloader_workers=1 \
        --instance_prompt="a photo of sks military pilot" \
        --class_prompt="a photo of military pilot" \
        --resolution=512 \
        --train_batch_size=1 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=100 \
        --use_lora \
        --lora_r 16 \
        --lora_alpha 27 \
        --lora_text_encoder_r 16 \
        --lora_text_encoder_alpha 17 \
        --learning_rate=1e-4 \
        --gradient_accumulation_steps=1 \
        --gradient_checkpointing \
        --max_train_steps=800 \
        --backbone_data_dir="../dsbat/datasets/laion_coco/laion_dataset_img" \
        --backbone_prompt="../dsbat/datasets/laion_coco/laion_dataset_text/metadata.json" \
        --score_data_dir="../dsbat/models/military_pilot/military_pilot_strong/score.json" \
        --gamma=$g \
        --sample_ratio $s &  # The '&' runs the command in the background for each process
    done

    # Wait for all background jobs to finish before starting the next combination of gamma and sample_ratio
    wait

  done
done

# Wait for all background jobs to finish before exiting the script
wait