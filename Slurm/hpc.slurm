#!/bin/bash
#SBATCH --job-name=BioClinBERT
#SBATCH --output=%x_%j.out              
#SBATCH --error=%x_%j.err               
#SBATCH --partition=ampere80
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:0

# 1. Environment
source ~/.bashrc
conda activate bcb                
module load PyTorch/Python3.10   

python - <<'PY'
import torch, platform, os, datetime as dt
print("CUDA available:", torch.cuda.is_available(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Started at:", dt.datetime.now(), "| Host:", platform.node())
PY

# 2. Fixed-value “global” arguments
data_path="$HOME/data/train.csv"               # --data_path
text_column="TEXT"                             # --text_column
label_column="LABEL"                           # --label_column
primary_key="HADM_ID"                          # --primary_key  (or leave blank)

model_name="emilyalsentzer/Bio_ClinicalBERT"   # --model_name
num_labels=2                                   # --num_labels
num_epochs=200                                 # --num_epochs
batch_size=64                                  # --batch_size
print_every=1                                  # --print_every
early_stop_patience=10                         # --early_stop_patience
dropout_prob=0.2                               # --dropout_prob
unfreeze_layers=1                              # --unfreeze_layers

#  (optional) path to resume from; leave empty ("") if training from scratch
model_weight_path=""                           # --model_weight_path

# 3. Hyper-parameter grids – extend or trim as required
optimizers=("AdamW" "Adam" "SGD")
learning_rates=(1e-5)
weight_decays=(0.01)
seeds=(14 13 108 777 21 90 22 12 7)
test_splits=(0.2)

# 4. Output base directory
BASE_DIR="$HOME/models/bcbert_runs"
mkdir -p "${BASE_DIR}"

# 5. Grid search
for opt in "${optimizers[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for wd in "${weight_decays[@]}"; do
      for seed in "${seeds[@]}"; do
        for split in "${test_splits[@]}"; do

          run_tag="${opt}_lr=${lr}_wd=${wd}_unf=${unfreeze_layers}_seed=${seed}_ts=${split}"
          out_dir="${BASE_DIR}/${run_tag}"
          mkdir -p "${out_dir}"

          # Construct CLI
          cli=(
            --data_path        "${data_path}"
            --save_model_path  "${out_dir}/model.pt"
            --model_name       "${model_name}"
            --num_labels       "${num_labels}"
            --lr               "${lr}"
            --weight_decay     "${wd}"
            --batch_size       "${batch_size}"
            --seed             "${seed}"
            --num_epochs       "${num_epochs}"
            --test_split       "${split}"
            --text_column      "${text_column}"
            --label_column     "${label_column}"
            --optimizer_class  "${opt}"
            --unfreeze_layers  "${unfreeze_layers}"
            --early_stop_patience "${early_stop_patience}"
            --dropout_prob     "${dropout_prob}"
            --print_every      "${print_every}"
            --verbose
            --debug
          )

          # optional path flags
          [[ -n "$primary_key"       ]] && cli+=( --primary_key "$primary_key" )
          [[ -n "$model_weight_path" ]] && cli+=( --model_weight_path "$model_weight_path" )

          echo "▶︎ Running:  ${run_tag}"
          python3 train.py "${cli[@]}"

        done
      done
    done
  done
done
