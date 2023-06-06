import torch
from transformers import BlipForConditionalGeneration, BlipProcessor, AutoTokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipelineDDPM
from PIL import Image
import pickle
import numpy as np
from torch import autocast, inference_mode
import argparse

from inversion_utils import  inversion_forward_process


def generate_captions(input_prompt, tokenizer, model):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids, temperature=0.8, num_return_sequences=16, do_sample=True, max_new_tokens=128, top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


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


def main(args):
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt

    source_captions_fp = args.source_captions_file_name
    target_captions_fp = args.target_captions_file_name

    # Generating source and target embeddings
    print("Generating source and target embeddings:")
    source_captions, target_captions = Generating_embeddings(source_prompt, target_prompt)

    with open(source_captions_fp, "wb") as fp:
        pickle.dump(source_captions, fp)
    with open(target_captions_fp, "wb") as fp:
        pickle.dump(target_captions, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_prompt", type=str, default="")
    parser.add_argument("--target_prompt", type=str, default="")
    parser.add_argument("--source_captions_file_name", type=str, default="")
    parser.add_argument("--target_captions_file_name", type=str, default="")
    args = parser.parse_args()

    main(args)