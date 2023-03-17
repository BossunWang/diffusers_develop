export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

#export OUTPUT_DIR="models/"
#export INPUT_IMAGE="imgs/Official_portrait_of_Barack_Obama.jpg"

#CUDA_VISIBLE_DEVICES=0 accelerate launch train_imagic.py \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --output_dir=$OUTPUT_DIR \
#  --input_image=$INPUT_IMAGE \
#  --target_text="A photo of Barack Obama smiling with a big grin." \
#  --seed=3434554 \
#  --resolution=256 \
#  --mixed_precision="fp16" \
#  --use_8bit_adam \
#  --gradient_accumulation_steps=1 \
#  --emb_learning_rate=1e-3 \
#  --learning_rate=1e-6 \
#  --emb_train_steps=500 \
#  --max_train_steps=1000


export OUTPUT_DIR="models_bird/"
export INPUT_IMAGE="imgs/bird.jpg"
CUDA_VISIBLE_DEVICES=0 accelerate launch train_imagic.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --input_image=$INPUT_IMAGE \
  --target_text="A bird spreading wings." \
  --seed=3434554 \
  --resolution=256 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --emb_learning_rate=1e-3 \
  --learning_rate=1e-6 \
  --emb_train_steps=500 \
  --max_train_steps=1000 \
  --train_batch_size 1 \
  --gradient_checkpointing
