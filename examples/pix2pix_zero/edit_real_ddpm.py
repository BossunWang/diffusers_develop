import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, AutoTokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipelineDDPM
from PIL import Image
import pickle
import numpy as np
from torch import autocast, inference_mode

from inversion_utils import  inversion_forward_process


def generate_captions(input_prompt, tokenizer, model):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


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


def Generating_embeddings(source_concept, target_concept):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto",
                                                       torch_dtype=torch.float16)

    source_text = f"Provide a caption for images containing a {source_concept}. "
    target_text = f"Provide a caption for images containing a {target_concept}. "

    source_captions = generate_captions(source_text, tokenizer, model)
    target_captions = generate_captions(target_concept, tokenizer, model)
    # target_captions = generate_captions(target_text, tokenizer, model)

    print("source_captions:", source_captions)
    print("target_captions:", target_captions)

    return source_captions, target_captions


def DDIM_inversion(pipeline,
                   generator,
                   raw_image,
                   source_prompt,
                   target_prompt,
                   source_captions_fp,
                   target_captions_fp):
    print("get caption from image:")
    caption = pipeline.generate_caption(raw_image)
    # caption = "a photography of a black and white kitten in a field of daies"
    print("caption:", caption)

    print("DDIM invert:")
    inv_latents = pipeline.invert(caption, image=raw_image, generator=generator).latents
    print("inv_latents:", inv_latents.shape)

    # # Generating source and target embeddings
    # print("Generating source and target embeddings:")
    # source_captions, target_captions = Generating_embeddings(source_prompt, target_prompt)
    #
    # with open(source_captions_fp, "wb") as fp:
    #     pickle.dump(source_captions, fp)
    # with open(target_captions_fp, "wb") as fp:
    #     pickle.dump(target_captions, fp)

    return caption, inv_latents


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
                   generator,
                   num_diffusion_steps,
                   raw_image,
                   x0,
                   source_prompt,
                   target_prompt,
                   source_captions_fp,
                   target_captions_fp,
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

    # Generating source and target embeddings
    # print("Generating source and target embeddings:")
    # source_captions, target_captions = Generating_embeddings(source_prompt, target_prompt)
    #
    # with open(source_captions_fp, "wb") as fp:
    #     pickle.dump(source_captions, fp)
    # with open(target_captions_fp, "wb") as fp:
    #     pickle.dump(target_captions, fp)

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


def main():
    # load models
    captioner_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(captioner_id)
    model = BlipForConditionalGeneration.from_pretrained(captioner_id, torch_dtype=torch.float16,
                                                         low_cpu_mem_usage=True)

    sd_model_ckpt = "CompVis/stable-diffusion-v1-4"
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
    img_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/pix2pix-zero/assets/test_images/cats/cat_6.png"
    # img_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/custom-diffusion_develop/data/bean_curd_cat/line_3487386995580014.jpg"
    # img_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/pix2pix-zero/assets/test_images/cats/cat_5.png"
    # img_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/art_dataset_v2/LOUIS WAIN/555.jpg"
    # img_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/art_dataset_v2/LOUIS WAIN/writing-a-letter.jpg"
    # img_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/art_dataset_v2/LOUIS WAIN/top-cat.jpg"

    raw_image = Image.open(img_path).convert("RGB").resize((512, 512))

    # DDPM inversion
    device_num = 0
    device = f"cuda:{device_num}"
    model_id = "CompVis/stable-diffusion-v1-4"
    num_diffusion_steps = 100
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")
    ldm_stable.scheduler.set_timesteps(num_diffusion_steps)
    offsets = (0, 0, 0, 0)
    x0 = load_512(img_path, *offsets, device)

    source_prompt = "cat"
    target_prompt = "dog"

    source_captions_fp = "cat_captions.pkl"
    target_captions_fp = "dog_captions.pkl"

    eta = 1
    cfg_scale_src = 3.5
    skip = 36
    caption, inv_latents, zs = DDPM_inversion(pipeline,
                                              ldm_stable,
                                              generator,
                                              num_diffusion_steps,
                                              raw_image,
                                              x0,
                                              source_prompt,
                                              target_prompt,
                                              source_captions_fp,
                                              target_captions_fp,
                                              eta,
                                              cfg_scale_src,
                                              skip)
    inv_latents = inv_latents.half()

    del ldm_stable

    # # DDIM inversion
    # # get captions and inversion
    # source_prompt = "cat"
    # target_prompt = "dog"
    #
    # source_captions_fp = "cat_captions.pkl"
    # target_captions_fp = "dog_captions.pkl"
    #
    # caption, inv_latents = DDIM_inversion(pipeline,
    #                                       generator,
    #                                       raw_image,
    #                                       source_prompt,
    #                                       target_prompt,
    #                                       source_captions_fp,
    #                                       target_captions_fp)

    # Image-to-Image Translation
    # output_path = "edited_image_flan-t5_cat2dog.png"
    output_path = "edited_image_flan-t5_cat2dog_DDPM.png"
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

    # ---------------------

    # print("generate target image:")
    # # See the "Generating source and target embeddings" section below to
    # # automate the generation of these captions with a pre-trained model like Flan-T5 as explained below.
    # source_prompts = ["a cat sitting on the street", "a cat playing in the field", "a face of a cat"]
    # target_prompts = ["a dog sitting on the street", "a dog playing in the field", "a face of a dog"]
    #
    # source_embeds = pipeline.get_embeds(source_prompts, batch_size=2)
    # target_embeds = pipeline.get_embeds(target_prompts, batch_size=2)
    #
    # image = pipeline(
    #     caption,
    #     source_embeds=source_embeds,
    #     target_embeds=target_embeds,
    #     num_inference_steps=50,
    #     cross_attention_guidance_amount=0.15,
    #     generator=generator,
    #     latents=inv_latents,
    #     negative_prompt=caption,
    # ).images[0]
    # image.save("edited_image.png")
    # # image.save("edited_image_from_bean_curd_cat.png")


if __name__ == '__main__':
    main()