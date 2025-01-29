#pip install diffusers transformers accelerate scipy

import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"  # 예시로 v1.5
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A majestic fantasy landscape, cinematic, 4k, trending on artstation"
image = pipe(prompt, height=512, width=1024).images[0]  # 2:1 비율
image.save("fantasy_skybox.jpg")
