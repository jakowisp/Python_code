from diffusers import DiffusionPipeline
import numpy as np
pipeline = DiffusionPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-X-v10", cache_dir="./cache"
)
prompt = "Generate a cowgirl on a horse with red hat"

# Set num_inference_steps as per your memory or computation
image = pipeline(prompt, num_inference_steps=100, guidance_strength=0.7)
image_data = np.array(image.images[0])
image_pil = Image.fromarray(image_data.astype(np.uint8))
image_pil.save(f"output_images/Example10.png")
image_pil
