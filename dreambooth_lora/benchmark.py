#find dinov2 image similarity 
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import numpy as np
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
import json
import os

def generate_safe_image(prompt, pipeline, seed, num_inference_steps=50, guidance_scale=7.5, max_retries=5):
    for i in range(max_retries):
        # Set the seed for reproducibility with variation
        torch.manual_seed(seed * i * 999)
        
        # Generate an image
        output = pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        image = output.images[0]
        
        # Check for NSFW content
        if hasattr(output, "nsfw_content_detected") and not output.nsfw_content_detected[0]:
            return image  # Return the safe image
        else:
            print(f"NSFW content detected. Retrying with seed {i + 1}... ({i + 1}/{max_retries})")

    raise ValueError("Failed to generate a safe image after maximum retries.")


def calculate_squared_centroid_distance_torch(points1, points2):
        """
        Calculate the squared centroid distance between two sets of points using PyTorch.
        
        Parameters:
        points1, points2: torch.Tensor of shape (n, d), where n is the number of points and d is the dimension.

        Returns:
        Squared distance between the centroids of the two point sets.
        """
        # Compute the centroids of both sets
        centroid1 = torch.mean(points1, dim=0)
        centroid2 = torch.mean(points2, dim=0)
        
        # Calculate the squared distance between the centroids
        squared_distance = torch.sum((centroid1 - centroid2) ** 2)
        
        return squared_distance

def vendi_score_torch(samples, similarity_func):
    """
    Compute the Vendi Score for a given set of samples using a similarity function,
    leveraging the eigenvalues of the similarity matrix.

    Args:
        samples (list): List of samples.
        similarity_func (function): A function that computes the similarity between two samples.

    Returns:
        torch.Tensor: The Vendi Score.
    """
    # Normalize the samples
    samples = [x / torch.norm(x) for x in samples]

    n = len(samples)
    
    # Compute the similarity matrix K
    K = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = similarity_func(samples[i], samples[j])
    
    # Normalize the similarity matrix by dividing by its trace
    K /= torch.trace(K)
    
    # Compute the eigenvalues of K
    eigenvalues = torch.linalg.eigvalsh(K)  # Use eigvalsh since K is symmetric
    eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Clamp to avoid log(0) issues
    
    # Compute the Vendi Score using the eigenvalues
    vendi_score = torch.exp(-torch.sum(eigenvalues * torch.log(eigenvalues)))
    
    return vendi_score

def calculate_clip_score(image: Image.Image, prompt: str, device: str) -> float:
    # Load pre-trained CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Preprocess the image and prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(device)

    # Forward pass through the model to get image and text embeddings
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeddings = outputs.image_embeds
        text_embeddings = outputs.text_embeds

    # Calculate cosine similarity between image and text embeddings
    clip_score = torch.cosine_similarity(image_embeddings, text_embeddings).item()

    # Return the CLIP score
    return round(clip_score, 4)

def main(args):
    result = {}
    images = []

    instance = args.instance
    model_path = args.model_path
    num_samples = args.num_samples
    gpu_id = args.gpu_id
    instance_data_path = args.instance_data_path

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    pipeline = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

    unet = PeftModel.from_pretrained(pipeline.unet, model_path + "/unet").to(device)
    text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, model_path + "/text_encoder").to(device)

    pipeline.unet = unet
    pipeline.text_encoder = text_encoder

    # Move remaining components to the correct GPU
    pipeline.vae.to(device)

    # find dinov2 image similarity
    # Load the DINOv2 model and processor
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

    instance_prompt = f"A photo of a sks {instance}"

    #get the original image path
    path = Path(instance_data_path)
    path_list = list(path.iterdir())

    #get lists of scores
    cosine_similarities = []
    centroid_distances = []
    clip_scores = []

    for i in range(len(path_list) * num_samples):

        original_image = Image.open(path_list[i % len(path_list)])
        torch.manual_seed(i)
        model_image = generate_safe_image(instance_prompt, pipeline , seed=i)

        # Define the transform to preprocess the image

        # Preprocess the image
        original = processor(images=original_image, return_tensors="pt")
        original = original.to(device)

        original_embedding = model(**original).last_hidden_state.mean(dim=1)

        # Calculate the cosine similarity between the embeddings

        image = processor(images=model_image, return_tensors="pt")
        image = image.to(device)
        embedding = model(**image).last_hidden_state.mean(dim=1)
        images.append(embedding)

        cosine_similarity = torch.nn.functional.cosine_similarity(embedding, original_embedding, dim=1)
        cosine_similarities.append(cosine_similarity.item())

        # print(f"cosine similarity: {cosine_similarity.item()}")

        # find dinov2 image centroid distance
        centroid_distance = calculate_squared_centroid_distance_torch(embedding, original_embedding)
        centroid_distances.append(centroid_distance.item())

        # print(f"centroid distance: {centroid_distance.item()}")

        # find clip score
        clip_score = calculate_clip_score(model_image, instance_prompt, args.gpu_id)
        clip_scores.append(clip_score)

        # print(f"CLIP Score: {random_clip_score}")

    #calcuate scores 
    print(f"Average cosine similarity: {np.mean(cosine_similarities)}")
    print(f"Average centroid distance: {np.mean(centroid_distances)}")
    print(f"Average CLIP Score: {np.mean(clip_scores)}")
    vendi_score = vendi_score_torch(images, torch.nn.functional.cosine_similarity)
    print(f"Vendi Score: {vendi_score.item()}")

    result["cosine_similarity"] = np.mean(cosine_similarities)
    result["centroid_distance"] = np.mean(centroid_distances)
    result["clip_score"] = np.mean(clip_scores)
    result["vendi_score"] = vendi_score.item()

    with open(f"{model_path}/benchmark.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, default="military pilot")
    parser.add_argument("--instance_data_path", type=str, default="/nfs/home/jaewan/data-selection-for-bat/dsbat/datasets/military_pilot/military_pilot_instance/val")
    parser.add_argument("--model_path", type=str, default="/nfs/home/jaewan/data-selection-for-bat/dsbat/models/military_pilot/military_pilot_strong")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    main(args)

    

