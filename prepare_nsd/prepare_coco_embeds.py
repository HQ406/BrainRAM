
import numpy as np
import torch
from diffusers import VersatileDiffusionDualGuidedPipeline
from tqdm import tqdm
import os

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vd_cache_dir = 'shi-labs/versatile-diffusion'       # or you could specify your own cache dir
test_coco_dir = 'processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub)
train_coco_dir = 'processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub)

guidance_scale = 3.5

@torch.no_grad()
def prepare_coco_embeds(dir, name):
    coco_data = np.load(dir)
    print(dir)
    txt_embeds_all = np.zeros((coco_data.shape[0], 2, 77, 768), dtype=np.float32)
    for i in tqdm(range(coco_data.shape[0])):
        # txt_embeds = vd_pipe._encode_text_prompt(list(coco_data[i]), device, 1, guidance_scale)
        txt_embeds = vd_pipe._encode_text_prompt([list(coco_data[i])[0]], device, 1, guidance_scale)
        txt_embeds_all[i] = txt_embeds.detach().cpu().numpy()

    print(f'{os.path.dirname(dir)}/nsd_{name}_cap_embeds.npy')
    np.save(f'{os.path.dirname(dir)}/nsd_{name}_cap_embeds.npy', txt_embeds_all)


if __name__ == '__main__':
    print(device)
    with torch.no_grad():
        vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device)
        prepare_coco_embeds(train_coco_dir, 'train0')
        prepare_coco_embeds(test_coco_dir, 'test0')


