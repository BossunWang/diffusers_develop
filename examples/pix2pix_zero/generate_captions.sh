export HF_HOME="/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/.cache/huggingface"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/"

python generate_captions.py \
      --source_prompt "cat" \
      --target_prompt "dog" \
      --source_captions_file_name "cat_captions.pkl" \
      --target_captions_file_name "dog_captions.pkl"


python generate_captions.py \
      --source_prompt "cat" \
      --target_prompt "pig" \
      --source_captions_file_name "cat_captions.pkl" \
      --target_captions_file_name "pig_captions.pkl"