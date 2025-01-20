python ./train_lora.py --train_path="/home/paneah/Desktop/data-selection-for-bat/dsbat/datasets/nlp_datasets/hellaswag.train.json" \
    --eval_path="/home/paneah/Desktop/data-selection-for-bat/dsbat/datasets/nlp_datasets/hellaswag.val.json" \
    --epochs=1 \
    --batch_size=1 \
    --use_dora \
    --output_dir="/home/paneah/Desktop/data-selection-for-bat/dsbat/models/nlp" \