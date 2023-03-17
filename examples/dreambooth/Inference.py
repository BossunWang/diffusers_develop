from diffusers import StableDiffusionPipeline
import torch


def main():
    # model_id = "checkpoints/800"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    #
    # prompt = "A photo of sks dog in a bucket"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("dog-bucket.png")

    # model_id = "checkpoints_face/800"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    #
    # prompt = "A photo of sks woman face"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("angry_face.png")

    # model_id = "checkpoints_cat/800"
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    #
    # # prompt = "the Van Gogh of sks cat in the river"
    # prompt = "sks cat with Christmas hat stand on Christmas tree"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("bean_curd_cat.png")

    model_id = "checkpoints_louis_cat/800"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    # prompt = "a Louis Wain sks cat in the garden"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("Louis_Wain_cat1.png")
    #
    # prompt = "a Louis Wain sks cat in the beach"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("Louis_Wain_cat2.png")
    #
    # prompt = "a Louis Wain sks cat smiling"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("Louis_Wain_cat3.png")
    #
    # prompt = "a Louis Wain sks cat sitting"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("Louis_Wain_cat4.png")
    #
    # prompt = "a Louis Wain sks cat and apple"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("Louis_Wain_cat5.png")

    # prompt = "a Louis Wain sks cat angry"
    # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save("Louis_Wain_cat6.png")

    prompt = "a Louis Wain sks cat sleeping"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save("Louis_Wain_cat7.png")

    prompt = "a Louis Wain sks cat shock"
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save("Louis_Wain_cat8.png")


if __name__ == '__main__':
    main()