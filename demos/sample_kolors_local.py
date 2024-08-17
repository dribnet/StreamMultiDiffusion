import torch
from pipeline_kolors_subclass import KolorsMultiPipeline

def infer(prompt='A photo of Alps', height=512, width=2048, output='outputs/pano_kol1.png'):
	device = 0

	device = torch.device(f'cuda:{device}')
	pipe = KolorsMultiPipeline.from_pretrained(
	    "Kwai-Kolors/Kolors-diffusers", 
	    torch_dtype=torch.float16, 
	    variant="fp16"
	).to(device)

	# Sample a image.
	image = pipe.sample(prompt, height=height, width=width).images[0]
	image.save(output)

if __name__ == '__main__':
    import fire
    fire.Fire(infer)
