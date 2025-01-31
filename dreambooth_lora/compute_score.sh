accelerate launch "./compute_score.py" \
  --train_data="../dsbat/datasets/military_pilot/military_pilot_instance/train" \
  --test_data="../dsbat/datasets/military_pilot/military_pilot_instance/val" \
  --prompt="a photo of sks military pilot" \
  --class_prompt="a photo of military pilot" \
  --class_data="../dsbat/datasets/military_pilot/military_pilot_class" \
  --model_path="../dsbat/models/military_pilot/military_pilot_strong" \
  --backbone_data="../dsbat/datasets/laion_coco/laion_dataset_img" \
  --backbone_prompt="../dsbat/datasets/laion_coco/laion_dataset_text/metadata.json" \
  --output_path="../dsbat/models/military_pilot/military_pilot_strong/score.json" \
  --sampling_count=3 \