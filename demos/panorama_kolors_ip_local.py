import torch
from pipeline_kolors_subclass import KolorsMultiPipeline
from diffusers import DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

def infer(prompt='A photo of Alps', height=512, width=2048, output='outputs/pano_kol1.png', ip_adapter=None):
    device = 0

    device = torch.device(f'cuda:{device}')

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "Kwai-Kolors/Kolors-IP-Adapter-Plus",
        subfolder="image_encoder",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        revision="refs/pr/4",
    )

    pipe = KolorsMultiPipeline.from_pretrained(
        "Kwai-Kolors/Kolors-diffusers",
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    ip_adapter_image = None
    if ip_adapter is not None:
        pipe.load_ip_adapter(
            "Kwai-Kolors/Kolors-IP-Adapter-Plus",
            subfolder="",
            weight_name="ip_adapter_plus_general.safetensors",
            revision="refs/pr/4",
            image_encoder_folder=None,
        )
        # pipe.enable_model_cpu_offload()
        # settings for "style"
        scale = {
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipe.set_ip_adapter_scale(scale)
        ip_adapter_image = load_image(ip_adapter)

    # Sample a panorama image.
    image = pipe.sample_panorama(
        prompt, 
        height=height, 
        width=width,
        ip_adapter_image=ip_adapter_image,
        ).images[0]
    image.save(output)

if __name__ == '__main__':
    import fire
    fire.Fire(infer)
