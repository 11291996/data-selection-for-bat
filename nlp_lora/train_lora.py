import os
import json
import torch
import random
import argparse
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorWithPadding, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from functools import partial
from tqdm import tqdm
from dreambooth_lora.train_model import compare_score_add_data

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', '-D', type=str, required=True, help="Path to the JSONL file containing training data")
    parser.add_argument('--eval_path', '-d', type=str, required=True, help="Path to the JSONL file containing evaluation data")
    parser.add_argument('--output_dir', '-o', type=str, default="./output", help="Directory to save the trained model")
    parser.add_argument('--epochs', '-e', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', '-b', type=int, default=1, help="Batch size per device")
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--use_gpu', '-gpu', type=str, default="0,1,2,3")
    parser.add_argument('--use_dora', '-dora', action='store_true', help="Use DoRA for training")
    parser.add_argument('--backbone_path', '-bpath', type=str, default=None, help="Path to the backbone data")
    parser.add_argument('--score_path', '-spath', type=str, default="./dsbat", help="Path to the score")
    parser.add_argument('--gamma', '-g', type=float, default=0.5, help="Gamma value for backbone data")
    parser.add_argument('--sample_ratio', '-sr', type=float, default=0.5, help="Sample ratio for backbone data")
    args = parser.parse_args()
    return args

def preprocess_function(examples, data_name, tokenizer):
    """Preprocess data for training with batching support.
    Download dataset here: https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset"""
    input_texts = []
    full_prompts = []

    for instruction, answer in zip(examples["instruction"], examples["answer"]):
        input_text = f"{instruction}\n"
        full_prompt = input_text + f"Answer: {answer}"
        input_texts.append(input_text)
        full_prompts.append(full_prompt)        

    # Tokenization
    model_inputs = tokenizer(
        full_prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    prompt_inputs = tokenizer(
        input_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    # ignore the prompt tokens in the loss
    labels = model_inputs["input_ids"].clone()
    for i in range(len(full_prompts)):
        prompt_len = prompt_inputs["attention_mask"][i].sum().item()
        labels[i][:prompt_len] = -100

    # # Validate token IDs
    # invalid_token_ids = (model_inputs["input_ids"] >= tokenizer.vocab_size).nonzero(as_tuple=True)
    # if invalid_token_ids[0].numel() > 0:
    #     print("Invalid token IDs found in input_ids:", model_inputs["input_ids"][invalid_token_ids])
    #     raise ValueError("Found invalid token IDs in input_ids.")

    model_inputs["labels"] = labels

    return model_inputs

def load_jsonl_data(file_path):
    """Load data from a JSONL or JSON file."""
    if file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            # if isinstance(data, dict):
            #     # If the JSON file is a single object, wrap it in a list
            #     data = [data]
    else:
        raise ValueError('Not a valid dataset path. Supported formats: .jsonl, .json')
    return data

def collate_fn(batch, tokenizer):
    # Initialize lists to store the tensors
    input_ids = []
    attention_masks = []
    labels = []

    # Loop over the batch and extract the tensors
    for item in batch:
        input_ids.append(torch.tensor(item['input_ids']))
        attention_masks.append(torch.tensor(item['attention_mask']))
        labels.append(torch.tensor(item['labels']))

    # Stack the tensors along a new dimension (batch size)
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    labels = torch.stack(labels, dim=0)

    # Create a dictionary to return
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

def train_model(train_path, eval_path, backbone_path, score_path, gamma, sample_ratio, output_dir, epochs, batch_size, learning_rate, use_dora=False):
    """Train the model using the provided JSON data."""

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.01,
        target_modules=["q_proj", "v_proj"],
        use_dora=use_dora
    )

    model = LlamaForCausalLM.from_pretrained("lainshower/Llama3-8b-alpaca",
                                            torch_dtype=torch.float16,
                                            device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained("lainshower/Llama3-8b-alpaca")
    tokenizer.pad_token = tokenizer.eos_token

    train_name = os.path.basename(train_path).split('.')[0]
    eval_name = os.path.basename(eval_path).split('.')[0]
    # Load and preprocess data
    if args.backbone_path:
        from compute_score import preprocess_function_backbone, collate_fn_backbone

        backbone_name = os.path.basename(args.backbone_path).split(".")[0]
        
        train_data = load_jsonl_data(train_path)
        eval_data = load_jsonl_data(eval_path)
        #apply gamma and sample ratio
        backbone_data = load_jsonl_data(backbone_path)
        sample_score = compare_score_add_data(score_path, sample_ratio)
        num_backbone_data = round(len(train_data) * (1 - gamma)) 
        score_items = list(sample_score.keys())
        backbone_index = score_items[:num_backbone_data]

        backbone_data_new = []

        for idx in backbone_index:
            idx = int(idx)
            backbone_data_new.append(backbone_data[idx])

        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        backbone_dataset = Dataset.from_list(backbone_data_new)
        train_preprocess_with_args = partial(preprocess_function, data_name=train_name, tokenizer=tokenizer)
        eval_preprocess_with_args = partial(preprocess_function, data_name=eval_name, tokenizer=tokenizer)
        backbone_preprocess_with_args = partial(preprocess_function_backbone, data_name=backbone_name, tokenizer=tokenizer)
        train_tokenized_dataset = train_dataset.map(train_preprocess_with_args, batched=True)
        eval_tokenized_dataset = eval_dataset.map(eval_preprocess_with_args, batched=True)
        backbone_tokenized_dataset = backbone_dataset.map(backbone_preprocess_with_args)
    else:
        train_data = load_jsonl_data(train_path)
        eval_data = load_jsonl_data(eval_path)
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        train_preprocess_with_args = partial(preprocess_function, data_name=train_name, tokenizer=tokenizer)
        eval_preprocess_with_args = partial(preprocess_function, data_name=eval_name, tokenizer=tokenizer)
        train_tokenized_dataset = train_dataset.map(train_preprocess_with_args, batched=True)
        eval_tokenized_dataset = eval_dataset.map(eval_preprocess_with_args, batched=True)

    model.config.use_cache = False
    model = get_peft_model(model, lora_config)

    adapter_name = "dora" if args.use_dora else "lora"
    output_dir = os.path.join(output_dir, adapter_name, train_name)
    if args.backbone_path:
        output_dir = os.path.join(output_dir, f"bat_gamma_{gamma}_sample_ratio_{sample_ratio}")

    collate_fn_with_args = partial(collate_fn, tokenizer=tokenizer)

    #set data loaders
    train_loader = DataLoader(train_tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_args)
    eval_loader = DataLoader(eval_tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_args)
    if args.backbone_path:
        backbone_collate_fn_with_args = partial(collate_fn_backbone, tokenizer=tokenizer)
        train_loader = DataLoader(train_tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_args)
        backbone_loader = DataLoader(backbone_tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=backbone_collate_fn_with_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.06 * (len(train_loader) * epochs), num_training_steps=len(train_loader) * epochs)

    model = model.cuda()

    results = {}
    train_losses = []
    eval_losses = []

    total_steps = len(train_loader) * epochs

    if epochs == 1 and backbone_path:
        train_step = len(train_loader) - len(backbone_loader)
        print(f"Train step: {train_step}")

    current_step = 0

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

            current_step += 1

            if current_step > total_steps:
                break
            
            if epochs == 1 and backbone_path and current_step > train_step:
                break

        if args.backbone_path:
            for step, batch in enumerate(tqdm(backbone_loader)):
                if current_step > total_steps:
                    break   
                model.train()
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_losses.append(loss.item())

                current_step += 1

        eval_loss = 0

        for step, batch in enumerate(tqdm(eval_loader)):
            model.eval()
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()

        # Calculate average loss

        eval_losses.append(eval_loss / len(eval_loader))

        if current_step > total_steps:
            break

    
    results["train_losses"] = train_losses
    results["eval_losses"] = eval_losses

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu
    train_model(args.train_path, 
    args.eval_path, 
    args.backbone_path,
    args.score_path,
    args.gamma,
    args.sample_ratio, 
    args.output_dir, 
    args.epochs, 
    args.batch_size, 
    args.learning_rate
    )
    print("Training complete.")