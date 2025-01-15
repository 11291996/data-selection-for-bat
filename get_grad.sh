accelerate launch "./diffusion_lora.py" \
  --train_data="/scratch2/paneah/dsbat/datasets/military_pilot/military_pilot_instance/train" \
  --test_data="/scratch2/paneah/dsbat/datasets/military_pilot/military_pilot_instance/val" \
  --prompt="a photo of sks military pilot" \
  --class_prompt="a photo of military pilot" \
  --class_data="/scratch2/paneah/dsbat/datasets/military_pilot/military_pilot_class" \
  --model_path="/scratch2/paneah/dsbat/models/military_pilot/military_pilot_strong" \
  --backbone_data="/scratch2/paneah/dsbat/datasets/laion_dataset/laion_dataset_img" \
  --backbone_prompt="/scratch2/paneah/dsbat/datasets/laion_dataset/laion_dataset_text/metadata.json" \
  --output_path="/scratch2/paneah/dsbat/models/military_pilot/military_pilot_strong/score.json" \