

# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Reconstructions.ipynb --to python"
# call(command,shell=True)


import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse
from models import Voxel2StableDiffusionModel
# import pyiqa

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

import utils
from models import *
from utils import batchwise_cosine_similarity, EvalMetrics

seed=42
utils.seed_everything(seed=seed)





# # Configurations

def get_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="name of trained model",
    )
    parser.add_argument(
        "--subj",type=int, default=1, choices=[1,2,5,7],
    )
    parser.add_argument(
        "--recons_per_sample", type=int, default=1,
        help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
    )
    parser.add_argument(
        "--vd_cache_dir", type=str, default='shi-labs/versatile-diffusion',
        help="Where is cached Versatile Diffusion model",
    )
    parser.add_argument(
        "--verbose", type=str, default=False,
        help="Verbose debug",
    )
    parser.add_argument(
        "--text_model", type=str, default=None,
        help="text_model",
    )
    parser.add_argument(
        "--image_model", type=str, default=None,
        help="image_model",
    )
    parser.add_argument(
        "--plotting", type=str, default=True,
        help="image_model",
    )
    return parser.parse_args()


def main():
    assert args.text_model is not None or args.image_model is not None

    if args.subj == 1:
        num_voxels = 15724
    elif args.subj == 2:
        num_voxels = 14278
    elif args.subj == 3:
        num_voxels = 15226
    elif args.subj == 4:
        num_voxels = 13153
    elif args.subj == 5:
        num_voxels = 13039
    elif args.subj == 6:
        num_voxels = 17907
    elif args.subj == 7:
        num_voxels = 12682
    elif args.subj == 8:
        num_voxels = 14386
    print("subj",args.subj,"num_voxels",num_voxels)


    voxel_val = f"../prepare_nsd/processed_data/subj0{args.subj}/nsd_test_fmriavg_nsdgeneral_sub{args.subj}.npy"
    img_val = f"../prepare_nsd/processed_data/subj0{args.subj}/nsd_test_stim_sub{args.subj}.npy"
    coco_val = f"../prepare_nsd/processed_data/subj0{args.subj}/nsd_test_cap_sub{args.subj}.npy"
    val_data = utils.MyDataset(voxel_val, img_val, coco_val)
    val_data.calc_text_feature(None, f'../train_logs/{args.model_name}/nsd_test_cap_embed{args.subj}.pt')
    num_val = len(val_data)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=1, shuffle=False)


    # check that your data loader is working
    for val_i, (voxel, img_input, coco) in enumerate(val_dl):
        print("idx",val_i)
        print("voxel.shape",voxel.shape)
        print("img_input.shape",img_input.shape)
        break



    # # Load VD pipe

    print('Creating versatile diffusion reconstruction pipeline...')
    from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    try:
        vd_pipe =  utils.VDDualGuidedPipelineEmbed.from_pretrained(args.vd_cache_dir).to(device).to(torch.float16)
    except:
        print("Downloading Versatile Diffusion to", args.vd_cache_dir)
        vd_pipe =  utils.VDDualGuidedPipelineEmbed.from_pretrained(
                "shi-labs/versatile-diffusion",
                cache_dir = args.vd_cache_dir).to(device).to(torch.float16)
    vd_pipe.image_unet.eval()
    vd_pipe.vae.eval()
    vd_pipe.text_encoder.eval()
    vd_pipe.image_unet.requires_grad_(False)
    vd_pipe.vae.requires_grad_(False)
    vd_pipe.text_encoder.requires_grad_(False)

    vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(args.vd_cache_dir, subfolder="scheduler")
    num_inference_steps = 20

    # Set weighting of Dual-Guidance 
    if args.text_model is not None and args.image_model is not None:
        text_image_ratio = .5 # .5 means equally weight text and image, 0 means use only image
    elif args.text_model is not None:
        text_image_ratio = 1.
    else:
        text_image_ratio = .0
    for name, module in vd_pipe.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = text_image_ratio
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler


    # ## Load Versatile Diffusion model

    clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)

    if args.image_model is not None:
        out_dim = 257 * 768
        voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
        voxel2clip = BrainMambaNetwork(**voxel2clip_kwargs)
        voxel2clip.requires_grad_(False)
        voxel2clip.eval()

        out_dim = 768
        depth = 6
        dim_head = 64
        heads = 12 # heads * dim_head = 12 * 64 = 768
        timesteps = 100 #100

        prior_network = VersatileDiffusionPriorNetwork(
                dim=out_dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                causal=False,
                num_tokens=257,
                learned_query_mode="pos_emb"
            )

        image_diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            voxel2clip=voxel2clip,
        )

        print('image prior ckpt_path', args.image_model)
        checkpoint = torch.load(args.image_model, map_location=device)
        state_dict = checkpoint['model_state_dict']
        image_diffusion_prior.load_state_dict(state_dict,strict=False)
        image_diffusion_prior.eval().to(device)
    
    if args.text_model is not None:
        out_dim = 77 * 768
        voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
        voxel2clip = BrainMambaNetwork(**voxel2clip_kwargs)
        voxel2clip.requires_grad_(False)
        voxel2clip.eval()

        out_dim = 768
        depth = 6
        dim_head = 64
        heads = 12 # heads * dim_head = 12 * 64 = 768
        timesteps = 100 #100

        prior_network = VersatileDiffusionPriorNetwork(
                dim=out_dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                causal=False,
                num_tokens=77,
                learned_query_mode="pos_emb"
            )

        text_diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
            voxel2clip=voxel2clip,
        )

        print('text prior ckpt_path', args.text_model)
        checkpoint = torch.load(args.text_model, map_location=device)
        state_dict = checkpoint['model_state_dict']
        text_diffusion_prior.load_state_dict(state_dict,strict=False)
        text_diffusion_prior.eval().to(device)


    outdir = f'../train_logs/{args.model_name}'
    os.makedirs(f'{outdir}/evals', exist_ok=True)



    # # Reconstruct one-at-a-time
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # retrieve = False
    plotting = args.plotting
    saving = True
    verbose = args.verbose
    imsize = 512

    # if img_variations:
    #     guidance_scale = 7.5
    # else:
    #     guidance_scale = 3.5
        
    ind_include = np.arange(num_val)
    all_brain_recons = None
    
    
        
    for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=len(ind_include))):
        if val_i<np.min(ind_include):
            continue
        # voxel = torch.mean(voxel,axis=1).to(device)
        # voxel = voxel[:,0].to(device)
        voxel = voxel.to(device)
        img = img.to(device)
        coco = coco.to(device)
        
        with torch.no_grad():
            
            if args.image_model is not None:
                image_prior_embeds, image_proj_embeds = utils.voxel2prioremb(
                    voxel=voxel,
                    clip_extractor=clip_extractor,
                    diffusion_priors = image_diffusion_prior,
                    recons_per_sample=args.recons_per_sample
                )
            if args.text_model is not None:
                text_prior_embeds, text_proj_embeds = utils.voxel2prioremb(
                    voxel=voxel,
                    clip_extractor=clip_extractor,
                    diffusion_priors = text_diffusion_prior,
                    recons_per_sample=args.recons_per_sample
                )
            if args.image_model is None:
                image_prior_embeds = torch.zeros(args.recons_per_sample, 257, 768).to(device).to(unet.dtype)
            if args.text_model is None:
                text_prior_embeds = torch.zeros(args.recons_per_sample, 77, 768).to(device).to(unet.dtype)
            # Add classifier-free-guidance
            image_prior_embeds_cfg = torch.cat([torch.zeros_like(image_prior_embeds), image_prior_embeds]).to(device).to(unet.dtype)
            text_prior_embeds_cfg = torch.cat([torch.zeros_like(text_prior_embeds), text_prior_embeds]).to(device).to(unet.dtype)
            dual_prompt_embeddings = torch.cat([text_prior_embeds_cfg, image_prior_embeds_cfg], dim=1)  # (2*bs, 334, 768)

            recons = vd_pipe.recons_with_dual_embeds(dual_prompt_embeddings)
            grid, best_pick, best_recon = utils.result2grid(
                                            image=img[0],
                                            recons=recons,
                                            clip_extractor=clip_extractor,
                                            recons_per_sample=args.recons_per_sample
                                        )

            if plotting:
                plt.show()
                grid.savefig(f'{outdir}/evals/{args.model_name}_{val_i}.png')
                plt.close(grid)

            if all_brain_recons is None:
                all_brain_recons = recons.unsqueeze(0)
                all_brain_recons_best = best_recon.unsqueeze(0)
                all_images = img
            else:
                all_brain_recons = torch.vstack((all_brain_recons, recons.unsqueeze(0)))
                all_brain_recons_best = torch.vstack((all_brain_recons_best, best_recon.unsqueeze(0)))
                all_images = torch.vstack((all_images, img))

        if val_i>=np.max(ind_include):
            break

    # all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if saving:
        torch.save(all_images,f'{outdir}/all_images.pt')
        torch.save(all_brain_recons,f'{outdir}/{args.model_name}_recons_{args.recons_per_sample}samples.pt')
        torch.save(all_brain_recons_best,f'{outdir}/recons_best.pt')
    print(f'recon_path: {os.path.abspath(outdir)}')



if __name__ == '__main__':
    args = get_args()
    print(args)
    main()

