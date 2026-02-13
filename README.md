# Step 1: convert train/test TIFFs
python scripts/step1_multi_subset_tif_to_png.py --config config/step1_subsets.yaml

# Step 2: build TRL datasets
python scripts/step2_build_trl_dataset.py

# Step 3: fine-tune VLM
python scripts/train_vlm_sft.py

# Step 4: evaluate
python scripts/infer_vlm_quality.py
