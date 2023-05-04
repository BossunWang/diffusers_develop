from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

def main():
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

    pipe.load_lora_weights("checkpoints_bean_curd_cat/")

    generator = torch.Generator(device="cuda").manual_seed(87)
    prompt = "A picture of a sks cat in a bucket"
    images = [
        pipe(prompt, num_inference_steps=25, generator=generator).images[0]
        for _ in range(4)
    ]

    [img.save("bean_curd_cat_{}.png".format(i)) for i, img in enumerate(images)]


if __name__ == '__main__':
    main()