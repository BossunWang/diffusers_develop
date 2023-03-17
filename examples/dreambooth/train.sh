export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

#export INSTANCE_DIR="train_instance_dog/"
#export CLASS_DIR="train_class_dog"
#export OUTPUT_DIR="checkpoints"

#CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_dreambooth.py \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --instance_data_dir=$INSTANCE_DIR \
#  --class_data_dir=$CLASS_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --with_prior_preservation \
#  --prior_loss_weight=1.0 \
#  --instance_prompt="a photo of sks dog" \
#  --class_prompt="a photo of dog" \
#  --resolution=512 \
#  --train_batch_size=1 \
#  --sample_batch_size=1 \
#  --gradient_accumulation_steps=1 \
#  --gradient_checkpointing \
#  --learning_rate=5e-6 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --num_class_images=200 \
#  --max_train_steps=800 \
#  --use_8bit_adam

#export INSTANCE_DIR="train_instance_face/"
#export CLASS_DIR="train_class_face"
#export OUTPUT_DIR="checkpoints_face"
#
#CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision="fp16" train_dreambooth.py \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --instance_data_dir=$INSTANCE_DIR \
#  --class_data_dir=$CLASS_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --with_prior_preservation \
#  --prior_loss_weight=1.0 \
#  --instance_prompt="a photo of sks woman face" \
#  --class_prompt="a photo of woman face" \
#  --resolution=256 \
#  --train_batch_size=1 \
#  --sample_batch_size=1 \
#  --gradient_accumulation_steps=1 \
#  --gradient_checkpointing \
#  --learning_rate=5e-6 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --num_class_images=200 \
#  --max_train_steps=800 \
#  --use_8bit_adam


#export INSTANCE_DIR="train_instance_face/"
#export CLASS_DIR="train_class_face"
#export OUTPUT_DIR="checkpoints_face"
#
#CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision="fp16" train_dreambooth.py \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --instance_data_dir=$INSTANCE_DIR \
#  --class_data_dir=$CLASS_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --with_prior_preservation \
#  --prior_loss_weight=1.0 \
#  --instance_prompt="a photo of sks woman face" \
#  --class_prompt="a photo of woman face" \
#  --resolution=256 \
#  --train_batch_size=1 \
#  --sample_batch_size=1 \
#  --gradient_accumulation_steps=1 \
#  --gradient_checkpointing \
#  --learning_rate=5e-6 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --num_class_images=200 \
#  --max_train_steps=800 \
#  --use_8bit_adam


#export INSTANCE_DIR="train_instance_cat/"
#export CLASS_DIR="train_class_cat"
#export OUTPUT_DIR="checkpoints_cat"
#
#CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision="fp16" train_dreambooth.py \
#  --pretrained_model_name_or_path=$MODEL_NAME \
#  --instance_data_dir=$INSTANCE_DIR \
#  --class_data_dir=$CLASS_DIR \
#  --output_dir=$OUTPUT_DIR \
#  --with_prior_preservation \
#  --prior_loss_weight=1.0 \
#  --instance_prompt="a photo of sks cat" \
#  --class_prompt="a photo of cat" \
#  --resolution=256 \
#  --train_batch_size=1 \
#  --sample_batch_size=1 \
#  --gradient_accumulation_steps=1 \
#  --gradient_checkpointing \
#  --learning_rate=5e-6 \
#  --lr_scheduler="constant" \
#  --lr_warmup_steps=0 \
#  --num_class_images=200 \
#  --max_train_steps=800 \
#  --use_8bit_adam

export INSTANCE_DIR="train_instance_cat/"
export CLASS_DIR="train_class_louis_cat"
export OUTPUT_DIR="checkpoints_louis_cat"

CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision="fp16" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a Louis Wain sks cat" \
  --class_prompt="a Louis Wain cat" \
  --resolution=256 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --use_8bit_adam