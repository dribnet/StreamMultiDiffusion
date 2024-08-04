import torch
from model import StableMultiDiffusionPipeline, StableMultiDiffusionSDXLPipeline, StableMultiDiffusionKolorsPipeline
from util import seed_everything

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 2022
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionKolorsPipeline(
    device,
)
    # hf_key='ironjr/BlazingDriveV11m',

# Load prompts.
prompts = [
    # Background prompt.
    '1girl, 1boy, times square',
    # Foreground prompts.
    '1boy, looking at viewer, brown hair, casual shirt',
    '1girl, looking at viewer, pink hair, leather jacket',
]
negative_prompts = [
    '',
    '1girl', # (Optional) The first prompt is a boy so we don't want a girl.
    '1boy', # (Optional) The first prompt is a girl so we don't want a boy.
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/timessquare/timessquare_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (768, 768) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=5,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=1,
)
image.save('region_kolors.png')