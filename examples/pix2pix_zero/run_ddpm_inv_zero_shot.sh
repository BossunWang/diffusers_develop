export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"

python ddpm_inv_zero_shot.py \
      --captioner_id "Salesforce/blip-image-captioning-base" \
      --model_id  "CompVis/stable-diffusion-v1-4" \
      --img_path "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/pix2pix-zero/assets/test_images/cats/cat_6.png" \
      --output_path "edited_image_flan-t5_cat2pig_DDPM.png" \
      --device_num 0 \
      --cfg_src 3.5 \
      --num_diffusion_steps 100 \
      --eta 1 \
      --skip 36 \
      --source_captions_file_name "cat_captions.pkl" \
      --target_captions_file_name "pig_captions.pkl"