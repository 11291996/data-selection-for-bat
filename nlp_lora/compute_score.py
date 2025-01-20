import os
import json
import torch
import random
import argparse
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorWithPadding, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from functools import partial
from tqdm import tqdm
from train_lora import load_jsonl_data, collate_fn, preprocess_function

def preprocess_function_backbone(examples, data_name, tokenizer):
    """Preprocess data for training with batching support.
    Download dataset here: https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset"""

    instruction = examples["instruction"] + " " + examples["input"]
    answer = examples["output"]


    input_texts = f"{instruction}\n"
    full_prompts = input_texts + f"Answer: {answer}"   


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
    prompt_len = prompt_inputs["attention_mask"].sum().item()
    labels[0][:prompt_len] = -100

    model_inputs["labels"] = labels

    return model_inputs

def collate_fn_backbone(batch, tokenizer):
    # Initialize lists to store the tensors

    input_ids = torch.tensor(batch[0]['input_ids'])
    attention_masks = torch.tensor(batch[0]['attention_mask'])
    labels = torch.tensor(batch[0]['labels'])

    # Create a dictionary to return
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", type=str, default="../dsbat")
    parser.add_argument("--adapter_path", type=str, default="../dsbat")
    parser.add_argument("--train_path", type=str, default="../dsbat")
    parser.add_argument("--eval_path", type=str, default="../dsbat")
    return parser.parse_args()

def compute_gradient(args):

    model = LlamaForCausalLM.from_pretrained("lainshower/Llama3-8b-alpaca",
                                            torch_dtype=torch.float16,
                                            device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained("lainshower/Llama3-8b-alpaca")
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    backbone_name = os.path.basename(args.backbone_path).split(".")[0]
    train_name = os.path.basename(args.train_path).split(".")[0]
    eval_name = os.path.basename(args.eval_path).split(".")[0]

    # Load the dataset
    backbone_data = load_jsonl_data(args.backbone_path)
    train_data = load_jsonl_data(args.train_path)
    eval_data = load_jsonl_data(args.eval_path)
    backbone_dataset = Dataset.from_list(backbone_data)
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    backbone_tokenized_dataset = backbone_dataset.map(partial(preprocess_function_backbone, data_name=backbone_name, tokenizer=tokenizer))
    train_tokenized_dataset = train_dataset.map(partial(preprocess_function, data_name=train_name, tokenizer=tokenizer), batched=True)
    eval_tokenized_dataset = eval_dataset.map(partial(preprocess_function, data_name=eval_name, tokenizer=tokenizer), batched=True)

    # Data loader 
    backbone_dataloader = DataLoader(
        backbone_tokenized_dataset,
        collate_fn=partial(collate_fn_backbone, tokenizer=tokenizer),
        batch_size=1,
        shuffle=False
    )

    train_dataloader = DataLoader(
        train_tokenized_dataset,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        batch_size=1,
        shuffle=False
    )

    eval_dataloader = DataLoader(
        eval_tokenized_dataset,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        batch_size=1,
        shuffle=False
    )


    config = PeftConfig.from_pretrained(args.adapter_path)
    lora_model = PeftModel.from_pretrained(model, args.adapter_path)
    lora_model.eval()

    # # Safely move model to CPU to free GPU memory
    # model.to('cpu')
    # del model  # Optional, to clear memory explicitly
    # torch.cuda.empty_cache()  # Clears unused memory on the GPU

    for k, v in lora_model.named_parameters():
        if 'lora_A' in k:
            v.requires_grad = True
        elif 'lora_B' in k:
            v.requires_grad = True
        else:
            v.requires_grad = False

    lora_model.cuda()

    tr_grad_dict = {}
    eval_grad_dict = {}
    back_grad_dict = {}

    test_count = 0


    for step, batch in enumerate(tqdm(train_dataloader)):
        lora_model.zero_grad()
        batch = {k: v.cuda() for k, v in batch.items()}
        model_outputs = lora_model(**batch)
        loss = model_outputs.loss
        loss.backward()
        print(loss)
        grad_dict={}
        for k, v in lora_model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k]=v.grad.cpu()
            elif 'lora_B' in k:
                # first index of shape indicates low-rank
                grad_dict[k]=v.grad.cpu().T
            else:
                pass
        tr_grad_dict[step]=grad_dict
        del grad_dict

        test_count += 1
        if test_count > 10:
            test_count = 0
            break

    for step, batch in enumerate(tqdm(eval_dataloader)):
        lora_model.zero_grad()
        batch = {k: v.to(lora_model.device) for k, v in batch.items()}
        model_outputs = lora_model(**batch)
        loss = model_outputs.loss
        print(loss)
        loss.backward()

        grad_dict={}
        for k, v in lora_model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k]=v.grad.cpu()
            elif 'lora_B' in k:
                # first index of shape indicates low-rank
                grad_dict[k]=v.grad.cpu().T
            else:
                pass
        eval_grad_dict[step]=grad_dict
        del grad_dict

        test_count += 1
        if test_count > 10:
            test_count = 0
            break

    for step, batch in enumerate(backbone_dataloader):
        lora_model.zero_grad()
        batch = {k: v.to(lora_model.device) for k, v in batch.items()}
        model_outputs = lora_model(**batch)
        loss = model_outputs.loss
        print(loss)
        loss.backward()

        grad_dict = {}

        for k, v in lora_model.named_parameters():
            if 'lora_A' in k:
                grad_dict[k]=v.grad.cpu()
            elif 'lora_B' in k:
                # first index of shape indicates low-rank
                grad_dict[k]=v.grad.cpu().T
            else:
                pass
        back_grad_dict[step]=grad_dict
        del grad_dict

        test_count += 1
        if test_count > 100:
            test_count = 0
            break

    return tr_grad_dict, eval_grad_dict, back_grad_dict

def compute_score_cuda(tr_grad_dict, val_grad_dict, backbone_grad_dict, lambda_const_param=10):
    # Move data to CUDA
    for data in tr_grad_dict:
        for layer in tr_grad_dict[data]:
            tr_grad_dict[data][layer] = tr_grad_dict[data][layer].cuda()
    for data in val_grad_dict:
        for layer in val_grad_dict[data]:
            val_grad_dict[data][layer] = val_grad_dict[data][layer].cuda()
    for data in backbone_grad_dict:
        for layer in backbone_grad_dict[data]:
            backbone_grad_dict[data][layer] = backbone_grad_dict[data][layer].cuda()

    # Compute Q
    Q = {}
    for layer in val_grad_dict[0]:
        Q[layer] = torch.zeros(val_grad_dict[0][layer].shape, device="cuda")
        for data in val_grad_dict:
            Q[layer] += val_grad_dict[data][layer] / len(val_grad_dict.keys())

    # Compute G
    G = {}
    for layer in tr_grad_dict[0]:
        G[layer] = torch.zeros(tr_grad_dict[0][layer].shape, device="cuda")
        for data in tr_grad_dict:
            G[layer] += tr_grad_dict[data][layer] / len(tr_grad_dict.keys())

    # Compute lambda, QG^-1, and GG^-1
    q_g_inv = {}
    g_g_inv = {}
    lambda_list = []

    for layer in G:
        # Compute lambda
        l = torch.zeros(len(tr_grad_dict.keys()), device="cuda")
        for i, data in enumerate(tr_grad_dict):
            tmp_grad = tr_grad_dict[data][layer]
            l[i] = torch.mean(tmp_grad**2)
        lambda_const = torch.mean(l) / lambda_const_param
        lambda_list.append(lambda_const)

        # Compute G^-1 and QG^-1 and GG^-1
        hvp_1 = torch.zeros(Q[layer].shape, device="cuda")
        hvp_2 = torch.zeros(G[layer].shape, device="cuda")
        for data in tr_grad_dict:
            tmp_grad = tr_grad_dict[data][layer]
            c_tmp_1 = torch.sum(Q[layer] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
            c_tmp_2 = torch.sum(G[layer] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
            hvp_1 += (Q[layer] - c_tmp_1 * tmp_grad) / (len(val_grad_dict.keys()) * lambda_const)
            hvp_2 += (G[layer] - c_tmp_2 * tmp_grad) / (len(tr_grad_dict.keys()) * lambda_const)

        q_g_inv[layer] = hvp_1
        g_g_inv[layer] = hvp_2

    # Calculate G(x)G^-1
    b_g_inv = {}
    for data in backbone_grad_dict:
        tmp_dict = {}
        for layer, lambda_const in zip(q_g_inv, lambda_list):
            tmp_grad = G[layer]
            c_tmp = torch.sum(backbone_grad_dict[data][layer] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
            hvp = (backbone_grad_dict[data][layer] - c_tmp * tmp_grad) / (len(tr_grad_dict.keys()) * lambda_const)
            tmp_dict[layer] = hvp
        b_g_inv[data] = tmp_dict

    # Calculate -tr(G(x)G^-1HG^-1) + 2 tr(G(x)G^-1HG^-1GG^-1)
    score_dict = {}
    for data in b_g_inv:
        score = 0
        for layer in b_g_inv[data]:
            tmp_grad = torch.sum(b_g_inv[data][layer] * q_g_inv[layer])
            score += -tmp_grad + 2 * torch.sum(tmp_grad * g_g_inv[layer])
        score_dict[data] = score.item()

    return score_dict


if __name__ == "__main__":
    args = parse_args()
    tr_grad_dict, eval_grad_dict, back_grad_dict = compute_gradient(args)
    import time
    start = time.time()
    score = compute_score_cuda(tr_grad_dict, eval_grad_dict, back_grad_dict)
    end = time.time()
    print("Time taken: ", end - start)
    #save the dict 
    with open(args.adapter_path + "/score.json", 'w') as fp:
        json.dump(score, fp)
