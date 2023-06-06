import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, AutoTokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipelineDDPM
from PIL import Image
import pickle
import numpy as np
from torch import autocast, inference_mode
import argparse

from inversion_utils import  inversion_forward_process


def embed_captions(sentences, tokenizer, text_encoder, device="cuda"):
    with torch.no_grad():
        embeddings = []
        for sent in sentences:
            text_inputs = tokenizer(
                sent,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            embeddings.append(prompt_embeds)
    return torch.cat(embeddings, dim=0).mean(dim=0).unsqueeze(0)


def load_512(image_path, left=0, right=0, top=0, bottom=0, device=None):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)

    return image


def DDPM_inversion(pipeline,
                   ldm_stable,
                   num_diffusion_steps,
                   raw_image,
                   x0,
                   eta,
                   cfg_scale_src,
                   skip,):
    print("get caption from image:")
    caption = pipeline.generate_caption(raw_image)
    # caption = "a photography of a black and white kitten in a field of daies"
    print("caption:", caption)

    print("DDPM invert:")
    # vae encode image
    with autocast("cuda"), inference_mode():
        w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

    wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=caption, cfg_scale=cfg_scale_src,
                                            prog_bar=True, num_inference_steps=num_diffusion_steps)
    if wts is None:
        print("inv_latents:", wt.shape)
    else:
        print("inv_latents:", wts[skip].shape)

    return caption, wts[skip].unsqueeze(0) if wts is not None else wt, zs


def zero_shot_I2I(caption,
                  inv_latents,
                  zs,
                  eta,
                  skip,
                  num_diffusion_steps,
                  pipeline,
                  generator,
                  source_captions_fp,
                  target_captions_fp,
                  output_path):

    with open(source_captions_fp, "rb") as fp:
        source_captions = pickle.load(fp)
    with open(target_captions_fp, "rb") as fp:
        target_captions = pickle.load(fp)

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    source_embeddings = embed_captions(source_captions, tokenizer, text_encoder)
    target_embeddings = embed_captions(target_captions, tokenizer, text_encoder)

    print("generate target image:")
    image = pipeline(
        caption,
        source_embeds=source_embeddings,
        target_embeds=target_embeddings,
        num_inference_steps=num_diffusion_steps,
        cross_attention_guidance_amount=0.15,
        generator=generator,
        latents=inv_latents,
        negative_prompt=caption,
        eta=eta,
        zs=zs[skip:]
    ).images[0]
    image.save(output_path)


def main(args):
    # load models
    captioner_id = args.captioner_id
    processor = BlipProcessor.from_pretrained(captioner_id)
    model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True)

    sd_model_ckpt = args.model_id
    pipeline = StableDiffusionPix2PixZeroPipelineDDPM.from_pretrained(
        sd_model_ckpt,
        caption_generator=model,
        caption_processor=processor,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    generator = torch.manual_seed(0)

    # load image
    img_path = args.img_path
    raw_image = Image.open(img_path).convert("RGB").resize((512, 512))

    # DDPM inversion
    device_num = args.device_num
    device = f"cuda:{device_num}"
    model_id = args.model_id
    num_diffusion_steps = args.num_diffusion_steps
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    ldm_stable.scheduler.set_timesteps(num_diffusion_steps)
    offsets = (0, 0, 0, 0)
    x0 = load_512(img_path, *offsets, device)

    source_captions_fp = args.source_captions_file_name
    target_captions_fp = args.target_captions_file_name

    eta = args.eta
    cfg_scale_src = args.cfg_src
    skip = args.skip
    caption, inv_latents, zs = DDPM_inversion(pipeline,
                                              ldm_stable,
                                              num_diffusion_steps,
                                              raw_image,
                                              x0,
                                              eta,
                                              cfg_scale_src,
                                              skip)
    inv_latents = inv_latents.half()

    del ldm_stable

    # Image-to-Image Translation
    output_path = args.output_path
    zero_shot_I2I(caption,
                  inv_latents,
                  zs,
                  eta,
                  skip,
                  num_diffusion_steps,
                  pipeline,
                  generator,
                  source_captions_fp,
                  target_captions_fp,
                  output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--captioner_id", type=str, default="")
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--cfg_src", type=float, default=3.5)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--skip", type=int, default=36)
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--source_captions_file_name", type=str, default="")
    parser.add_argument("--target_captions_file_name", type=str, default="")

    args = parser.parse_args()

    main(args)