import argparse
import sys
import torch
from model import StableMultiDiffusionPipeline
from model import StableMultiDiffusionSDXLPipeline
from model import StableMultiDiffusionFluxPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A photo of the dolomites")
    parser.add_argument("--outfile", type=str, default="outputs/sample.png")
    parser.add_argument("--pipeline", type=str, default="OG")
    args = parser.parse_args()

    device = 0

    # Load the module.
    device = torch.device(f'cuda:{device}')
    pipe = args.pipeline.lower()
    if pipe == 'og':
        smd = StableMultiDiffusionPipeline(device)
    elif pipe == 'xl':
        smd = StableMultiDiffusionSDXLPipeline(device)
    elif pipe == 'flux.dev' or pipe == 'flux':
        smd = StableMultiDiffusionFluxPipeline(device, hf_key="black-forest-labs/FLUX.1-dev")
    elif pipe == 'flux.schnell' or pipe == 'schnell':
        smd = StableMultiDiffusionFluxPipeline(device, hf_key="black-forest-labs/FLUX.1-schnell")
    else:
        print(f"Unknown pipeline: {args.pipeline}")
        sys.exit(1)

    # Sample an image.
    image = smd.sample(args.prompt)
    image.save(args.outfile)

if __name__ == '__main__':
    main()
