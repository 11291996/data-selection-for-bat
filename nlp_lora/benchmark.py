import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import os
import json
from tqdm import tqdm
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned causal language model on a specified dataset.")
    parser.add_argument(
        '--data_root', '-r', 
        type=str, 
        default="./dataset", 
        help="Root directory containing the dataset. Each dataset should be in a subdirectory with a '.val.json' file."
    )
    parser.add_argument(
        '--model_root', '-m', 
        type=str, 
        default="./output/", 
        help="Root directory containing the dataset. Each dataset should be in a subdirectory with a '.val.json' file."
    )
    parser.add_argument(
        '--batch_size', '-b', 
        type=int, 
        default=1, 
        help="Batch size for processing dataset examples. Default is 1."
    )
    parser.add_argument(
        '--epoch', '-e',
        type=int,
        default=1
    )
    parser.add_argument(
        '--validate_dora', '-dora',
        action='store_true',
        default=False
    )
    args = parser.parse_args()
    return args

def main(args):
    data_path = args.data_root
    model_path = args.model_root
    data_name = os.path.basename(model_path)


    # Load LoRA configuration to get base model path
    config_path = os.path.join(model_path, "adapter_config.json")
    
    # Load PEFT configuration
    config = PeftConfig.from_pretrained(model_path)
    base_model_name_or_path = config.base_model_name_or_path

    # Load base model from the original pre-trained model
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA parameters from output_dir
    model = PeftModel.from_pretrained(base_model, model_path)

    # Set use_cache to True
    model.config.use_cache = True

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

    # Set to evaluation mode
    model.eval()
    
    # Load dataset
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    # Evaluate model
    correct = 0
    total = len(dataset)
    num_batches = (total + args.batch_size - 1) // args.batch_size  # Calculate total number of batches

    results = []

    for i in tqdm(range(num_batches), desc="Evaluating Batches"):
        batch = dataset[i * args.batch_size:(i + 1) * args.batch_size]
        
        instructions = [entry['instruction'] for entry in batch]
        expected_answers = [entry['answer'] for entry in batch]

        # Build prompts with "Answer:" appended
        prompts = [f"{instruction}\nAnswer:" for instruction in instructions]

        # Tokenize input
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to("cuda")

        # 명시적으로 attention_mask 전달
        attention_mask = inputs['attention_mask']

        # 입력 시퀀스 길이 계산
        input_lengths = [len(input_ids) for input_ids in inputs['input_ids']]

        # Generate model output
        try:
            outputs = model.generate(  # greedy decoding
                inputs['input_ids'],
                attention_mask=attention_mask,  # Add attention_mask
                max_new_tokens=64,
                do_sample=False,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                temperature=1.0,
                top_p=1.0,
                top_k=0
            )
        except RuntimeError as e:
            print(f"Error encountered in batch {i}:")
            print(f"Inputs: {instructions}")
            raise e

        # 생성된 텍스트에서 입력 부분 제외하고 추출
        generated_texts = []
        for output, input_length in zip(outputs, input_lengths):
            generated_text = tokenizer.decode(
                output[input_length:],  # 입력 길이 이후의 토큰만 추출
                skip_special_tokens=True
            ).strip()
            generated_texts.append(generated_text)

        # Extract answers and compare
        for idx, (gen_text, expected_answer) in enumerate(zip(generated_texts, expected_answers)):
            gen_text = str(gen_text).strip()
            pred_answer = extract_answer(data_name, gen_text)
            is_correct = pred_answer.strip().lower() == expected_answer.strip().lower()
            correct += int(is_correct)

            # Collect individual results
            results.append({
                "Example": i * args.batch_size + idx + 1,
                "Instruction": instructions[idx],
                "Expected Answer": expected_answer,
                "Generated Text": gen_text,
                "Predicted Answer": pred_answer,
                "Correct": is_correct
            })

    # Calculate accuracy
    accuracy = correct / total * 100

    # save path for results
    save_path = os.path.join(model_path, "benchmark.txt")

    # Write results to file
    with open(save_path, "w") as f:
        # Write overall accuracy at the beginning
        f.write("Evaluation Results:\n")
        f.write(f"Dataset: {data_name}\n")
        f.write(f"Total Examples: {total}\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})\n\n")

        # Write individual results
        for result in results:
            f.write(f"Example {result['Example']}:\n")
            f.write(f"> Instruction: {result['Instruction']}\n")
            f.write(f"> Expected Answer: {result['Expected Answer']}\n")
            f.write(f"> Generated Text: {result['Generated Text']}\n")
            f.write(f"> Predicted Answer: {result['Predicted Answer']}\n")
            f.write(f"> Correct: {result['Correct']}\n\n")

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

def extract_answer(dataset_name, sentence: str) -> str:
    sentence_ = sentence.strip().lower()
    # if "bat" in dataset_name:
    #     dataset_name = dataset_name.split('_')[0]
    if dataset_name in ['boolq', 'winogrande', 'toy']:
        pred_answers = re.findall(r'true|false|option1|option2', sentence_)
    elif dataset_name == 'piqa':
        pred_answers = re.findall(r'solution1|solution2', sentence_)
    elif dataset_name in ['social_i_qa', 'arc-c', 'arc-e', 'openbookqa']:
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
    elif dataset_name == 'hellaswag':
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
    else:
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)

    return pred_answers[0] if pred_answers else ""

if __name__ == '__main__':
    args = parse_args()
    main(args)