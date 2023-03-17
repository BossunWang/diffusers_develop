import os
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler


def main():
    OUTPUT_DIR = "models_bird/"
    model_path = OUTPUT_DIR  # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16).to(
        "cuda")
    target_embeddings = torch.load(os.path.join(model_path, "target_embeddings.pt")).to("cuda")
    optimized_embeddings = torch.load(os.path.join(model_path, "optimized_embeddings.pt")).to("cuda")
    g_cuda = None
    # @markdown Can set random seed here for reproducibility.
    g_cuda = torch.Generator(device='cuda')
    seed = 4324  # @param {type:"number"}
    g_cuda.manual_seed(seed)

    # @title Run for generating images.

    alpha = 0.1  # @param {type:"number"}
    num_samples = 4  # @param {type:"number"}
    guidance_scale = 7.  # @param {type:"number"}
    num_inference_steps = 50  # @param {type:"number"}
    height = 256  # @param {type:"number"}
    width = 256  # @param {type:"number"}

    edit_embeddings = alpha * target_embeddings + (1 - alpha) * optimized_embeddings
    # bs_embed, seq_len, _ = edit_embeddings.shape
    # edit_embeddings = edit_embeddings.repeat(1, 2, 1)
    # edit_embeddings = edit_embeddings.view(bs_embed * 2, seq_len, -1)
    # print(edit_embeddings.shape)

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            text_embeddings=edit_embeddings,
            height=height,
            width=width,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    for i, img in enumerate(images):
        img.save("test_{}.png".format(i))



if __name__ == '__main__':
    main()