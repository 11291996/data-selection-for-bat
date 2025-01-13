import torch
import torch.nn.functional as F
import itertools
import argparse
from accelerate import Accelerator
from utils.dreambooth_lora import DreamBoothDataset, import_model_class_from_model_name_or_path, collate_fn, BackboneDreamBoothDataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

UNET_TARGET_MODULES = ["to_q", "to_v", "query", "value"]  # , "ff.net.0.proj"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "v_proj"]

def arg_parse():
    parser = argparse.ArgumentParser(description='Influence function for Diffusion LORA')
    #training data path
    parser.add_argument('--train_data', type=str, default='data/20ng_train.csv', help='path to training data')
    #test data path
    parser.add_argument('--test_data', type=str, default='data/20ng_test.csv', help='path to test data')
    #model path
    parser.add_argument('--model_path', type=str, default='model/20ng_model.pkl', help='path to trained model')
    #output path
    parser.add_argument('--output_path', type=str, default='output/20ng_output.csv', help='path to output file')
    #instance prompt
    parser.add_argument('--prompt', type=str, default='instance', help='prompt for instance')
    #class data path
    parser.add_argument('--class_data', type=str, default='data/20ng_class.csv', help='path to class data')
    #class prompt
    parser.add_argument('--class_prompt', type=str, default='class', help='prompt for class')
    #backbone path 
    parser.add_argument('--backbone_data', type=str, default='backbone/20ng_backbone.pkl', help='path to backbone model')
    #backbone prompt
    parser.add_argument('--backbone_prompt', type=str, default='backbone', help='prompt for backbone')
    args = parser.parse_args()
    return args

def compute_gradient(args):

    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    accelerator = Accelerator(
        gradient_accumulation_steps=1
    )

    if accelerator.is_main_process:
        # Load the tokenizer
    
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        # Load scheduler and models
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        ) 

        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=None)

        pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path)

        unet = PeftModel.from_pretrained(pipe.unet, args.model_path + "/unet")

        vae.requires_grad_(False)

        text_encoder = PeftModel.from_pretrained(pipe.text_encoder, args.model_path + "/text_encoder")

        # Dataset and DataLoaders creation:
        train_dataset = DreamBoothDataset(
            instance_data_root=args.train_data,
            instance_prompt=args.prompt,
            class_data_root=args.class_data if True else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=512,
            center_crop=False,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda examples: collate_fn(examples, True),
            num_workers=1,
        )

        # Validation dataset
        test_dataset = DreamBoothDataset(
            instance_data_root=args.test_data,
            instance_prompt=args.prompt,
            class_data_root=args.class_data if True else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=512,
            center_crop=False,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda examples: collate_fn(examples, True),
            num_workers=1,
        )

        #backbone dataset
        backbone_dataset = BackboneDreamBoothDataset( 
            instance_data_root=args.train_data,
            instance_prompt=args.backbone_prompt,
            class_data_root=args.class_data if True else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=512,
            center_crop=False,
        )

        backbone_dataloader = torch.utils.data.DataLoader(
            backbone_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda examples: collate_fn(examples, True),
            num_workers=1,
        )

        unet, text_encoder, train_dataloader, test_dataloader, backbone_dataloader = accelerator.prepare(
            unet, text_encoder, train_dataloader, test_dataloader, backbone_dataloader
        )

        # Move vae and text_encoder to device and cast to torch.float32
        vae.to(accelerator.device, dtype=torch.float32)

        # Set model and text_encoder to eval mode
        vae.eval()
        text_encoder.eval()
        unet.eval()

        for param in unet.parameters():
            param.requires_grad = True

        for param in text_encoder.parameters():
            param.requires_grad = True

        tr_grad_dict = {}
        val_grad_dict = {}

        # for step, batch in enumerate(tqdm(backbone_dataloader)):
        #     # Forward pass
        #      with accelerator.accumulate(unet):
        #         # Convert images to latent space

        #         unet.zero_grad()
        #         text_encoder.zero_grad()

        #         latents = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
        #         latents = latents * 0.18215

        #         # Sample noise that we'll add to the latents
        #         noise = torch.randn_like(latents)
        #         bsz = latents.shape[0]
        #         # Sample a random timestep for each image
        #         timesteps = torch.randint(
        #             0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        #         )
        #         timesteps = timesteps.long()

        #         # Add noise to the latents according to the noise magnitude at each timestep
        #         # (this is the forward diffusion process)
        #         noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        #         # Get the text embedding for conditioning
        #         encoder_hidden_states = text_encoder(batch["input_ids"])[0]

        #         # Predict the noise residual
        #         model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        #         # Get the target for loss depending on the prediction type
        #         if noise_scheduler.config.prediction_type == "epsilon":
        #             target = noise
        #         elif noise_scheduler.config.prediction_type == "v_prediction":
        #             target = noise_scheduler.get_velocity(latents, noise, timesteps)
        #         else:
        #             raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        #         # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
        #         model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        #         target, target_prior = torch.chunk(target, 2, dim=0)

        #         # Compute instance loss
        #         loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        #         # Compute prior loss
        #         prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

        #         # Add the prior loss to the instance loss.
        #         loss = loss + 1 * prior_loss

        #         accelerator.backward(loss)

        #         if accelerator.sync_gradients:
        #             params_to_clip = (
        #                 itertools.chain(unet.parameters(), text_encoder.parameters())
        #                 if True
        #                 else unet.parameters()
        #             )
        #             accelerator.clip_grad_norm_(params_to_clip, 1)


        for step, batch in enumerate(tqdm(train_dataloader)):
            # Forward pass
             with accelerator.accumulate(unet):
                # Convert images to latent space

                unet.zero_grad()
                text_encoder.zero_grad()

                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + 1 * prior_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if True
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1)
                
                grad_dict = {}
                for k, v in unet.named_parameters():
                    if "lora_A" in k:
                        grad_dict[k] = v.grad.cpu()
                    elif "lora_B" in k:
                        grad_dict[k] = v.grad.cpu().T
                    else:
                        pass
                tr_grad_dict[step] = grad_dict
                del grad_dict

        for step, batch in enumerate(tqdm(test_dataloader)):
            # Forward pass
             with accelerator.accumulate(unet):
                # Convert images to latent space

                unet.zero_grad()
                text_encoder.zero_grad()

                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + 1 * prior_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if True
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1)
                
                grad_dict = {}
                for k, v in unet.named_parameters():
                    if "lora_A" in k:
                        grad_dict[k] = v.grad.cpu()
                    elif "lora_B" in k:
                        grad_dict[k] = v.grad.cpu().T
                    else:
                        pass
                val_grad_dict[step] = grad_dict
                del grad_dict

    return tr_grad_dict, val_grad_dict

def compute_score(tr_grad_dict, val_grad_dict, backbone_grad_dict, lambda_const_param = 10):

    #TODO: replace later
    backbone_grad_dict = tr_grad_dict

    #compute Q
    Q = {}
    for layer in val_grad_dict[0].keys():
        Q[layer] = torch.zeros(val_grad_dict[0][layer].shape)
        for val_id in val_grad_dict:
            Q[layer] += val_grad_dict[val_id][layer] / len(val_grad_dict.keys())
    
    #compute G 
    G = {}
    for layer in tr_grad_dict[0].keys():
        G[layer] = torch.zeros(tr_grad_dict[0][layer].shape)
        for data in tr_grad_dict:
            G[layer] += tr_grad_dict[data][layer] / len(tr_grad_dict.keys())

    #compute lambda, QG^-1 and GG^-1
    q_g_inv = {}
    g_g_inv = {}
    lambda_list = []

    for layer in G: 
        #compute lambda
        l = torch.zeros(len(tr_grad_dict.keys()))
        for data in tr_grad_dict:
            tmp_grad = tr_grad_dict[data][layer]
            l[data] = torch.mean(tmp_grad**2)
        lambda_const = torch.mean(l) / lambda_const_param
        lambda_list.append(lambda_const)

        #compute G^-1 and QG^-1 and GG^-1
        hvp_1 = torch.zeros(Q[layer].shape)
        hvp_2 = torch.zeros(G[layer].shape)
        for data in tr_grad_dict:
            tmp_grad = tr_grad_dict[data][layer]
            c_tmp_1 = torch.sum(Q[layer] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
            c_tmp_2 = torch.sum(G[layer] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
            #identity matrix is mulitplied with Q[layer] 
            #calculate the backbone gradient likewise
            hvp_1 += (Q[layer] - c_tmp_1 * tmp_grad) / (len(tr_grad_dict.keys()) * lambda_const)
            hvp_2 += (G[layer] - c_tmp_2 * tmp_grad) / (len(tr_grad_dict.keys()) * lambda_const)

        q_g_inv[layer] = hvp_1
        g_g_inv[layer] = hvp_2

    #calculate G(x)G^-1
    b_g_inv = {}
    for data in backbone_grad_dict:
        tmp_dict = {}
        for layer, lambda_const in zip(q_g_inv, lambda_list):
            tmp_grad = tr_grad_dict[data][layer]
            c_tmp = torch.sum(backbone_grad_dict[data][layer] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
            hvp = (backbone_grad_dict[data][layer] - c_tmp * tmp_grad) / (len(tr_grad_dict.keys()) * lambda_const)
            tmp_dict[layer] = hvp
        b_g_inv[data] = tmp_dict

    #calculate -tr(G(x)G^-1HG^-1) + 2 tr(G(x)G^-1HG^-1GG^-1)
    score_dict = {}
    for data in b_g_inv: 
        score = 0
        for layer in b_g_inv[data]:
            tmp_grad = b_g_inv[data][layer] * q_g_inv[layer]
            score += -torch.trace(tmp_grad) + 2 * torch.trace(tmp_grad * g_g_inv[layer])
        score_dict[data] = score

    return score_dict

if __name__ == '__main__':
    import time 
    start = time.time()
    args = arg_parse()
    tr_grad_dict, val_grad_dict = compute_gradient(args)
    score_dict = compute_score(tr_grad_dict, val_grad_dict, None)
    print(score_dict)
    print(f"Time taken: {time.time() - start}")