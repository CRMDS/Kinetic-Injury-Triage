#!/bin/bash
#SBATCH --job-name=BCBERTClassifier
#SBATCH --output=~/models/llch/%x_%A.txt
#SBATCH --error=~/models/llch/%x_%A.txt
#SBATCH --partition=ampere80
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:0

BASE_DIR="~/models/llch"
mkdir -p "${BASE_DIR}"

# Environment
source ~/.bashrc
conda activate bcb
module load PyTorch/Python3.10
python - <<'PY'
import torch; print("CUDA available:", torch.cuda.is_available())
PY

# Hyper-parameter grids
optimizers=("AdamW" "Adam" "SGD")
seeds=(14 13 108 777 21 90 22 12 7)
#test_splits=(0.2 0.3 0.25)
test_splits=(0.2)

# Globals
batch_size=64
learning_rate=1e-4
weight_decay=0.01
early_stop_patience=10
dropout_prob=0.2
unfreeze_layers=1

for opt in "${optimizers[@]}"; do
  for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    split="${test_splits[$i]}"

    out_dir="${BASE_DIR}/${opt}_unfreeze_${unfreeze_layers}_seed_${seed}_split_${split}"
    mkdir -p "${out_dir}"

    python3 train.py \
      --data_path train.csv \
      --save_model_path "${out_dir}/${opt}_seed_${seed}_ts_${split}_unfreeze_${unfreeze_layers}.pt" \
      --num_epochs 200 \
      --lr ${learning_rate} \
      --weight_decay ${weight_decay} \
      --optimizer_class ${opt} \
      --unfreeze_layers ${unfreeze_layers} \
      --batch_size ${batch_size} \
      --test_split ${split} \
      --seed ${seed} \
      --early_stop_patience ${early_stop_patience} \
      --dropout_prob ${dropout_prob} \
      --verbose \
      --debug \
      --print_every 1
  done
done
