export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"
export MODEL_NAME="runwayml/stable-diffusion-v1-5"

#export INSTANCE_DIR="dog/"
#export OUTPUT_DIR="checkpoints_dog"
#
#accelerate launch train_dreambooth_lora.py \
#  --pretrained_model_name_or_path=$MODEL_NAME  \
#  --instance_data_dir=$INSTANCE_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --instance_prompt="a photo of sks dog" \
#  --resolution=256 \
#  --train_batch_size=1 \
#  --gradient_accumulation_steps=1 \
#  --checkpointing_steps=100 \
#  --learning_rate=1e-4 \
#  --report_to="wandb" \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --max_train_steps=500 \
#  --validation_prompt="A photo of sks dog in a bucket" \
#  --validation_epochs=50 \
#  --seed="87" \
#  --mixed_precision="fp16" \
#  --use_8bit_adam

export INSTANCE_DIR="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/custom-diffusion_develop/data/bean_curd_cat/"
export OUTPUT_DIR="checkpoints_bean_curd_cat"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks cat" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks cat in a bucket" \
  --validation_epochs=50 \
  --seed="87" \
  --mixed_precision="fp16" \
  --use_8bit_adam
