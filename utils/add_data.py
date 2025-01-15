import json
import numpy as np
from collections import defaultdict

def compare_score_add_data(json_path, sample_ratio):
    # Load the JSON file
    with open(json_path, 'r') as f:
        score = json.load(f)

    print(f"Total scores: {len(score)}")

    # Convert the score dictionary into a list of tuples (key, value)
    score_items = list(score.items())
    
    # Stratify the data by the score values (or bins of scores)
    bins = np.linspace(min([float(v) for v in score.values()]), max([float(v) for v in score.values()]), num=10)  # Create 10 bins
    stratified_scores = defaultdict(list)
    for k, v in score_items:
        bin_index = np.digitize(float(v), bins)
        stratified_scores[bin_index].append((k, v))
    
    # Perform proportional sampling from each bin
    sample_score = {}
    for bin_index, items in stratified_scores.items():
        num_samples = max(1, round(len(items) * sample_ratio))  # Ensure at least one sample per bin
        sampled_items = np.random.choice(len(items), num_samples, replace=False)
        for idx in sampled_items:
            key, value = items[idx]
            sample_score[key] = value

    print(f"Sampled scores: {len(sample_score)}")

    # Order the sampled score dictionary via the score values
    sample_score = dict(sorted(sample_score.items(), key=lambda item: float(item[1]), reverse=True))

    return sample_score

if __name__ == '__main__':
    json_path = "/scratch2/paneah/dsbat/models/military_pilot/military_pilot_strong/score.json"
    score_dict = compare_score_add_data(json_path, 1)
    import os 
    #get path list from a path
    img_path = "/scratch2/paneah/dsbat/datasets/laion_dataset/laion_dataset_img"
    from pathlib import Path
    img_path = Path(img_path)
    path_list = sorted(img_path.iterdir(), key=lambda x: int(''.join(filter(str.isdigit, x.name))))

    #get the path of the images
    for i in list(score_dict.keys())[:10]:
        idx = int(i)
        print(path_list[idx])


    

            
        