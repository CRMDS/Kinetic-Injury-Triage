#!/usr/bin/env python3
"""
@model: Bio_ClinicalBERTClassifier.py
@script: train
@author: Midhun Shyam (M.Shyam)
"""

import argparse
import pandas as pd
from Bio_ClinicalBERTClassifier import BioClinicalBERTClassifier
from torch.optim import AdamW, Adam, SGD
import os


def main(args):
    # Load dataset
    print("Loading dataset...", flush=True)
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns", flush=True)
    # Validate primary_key argument
    if args.primary_key is not None and args.primary_key not in df.columns:
        raise ValueError(
            f"primary_key '{args.primary_key}' not found in DataFrame columns: {list(df.columns)}"
        )

    # Display configuration
    print("-" * 27, flush=True)
    print("Configuration:", flush=True)
    print(f"  Optimizer: {args.optimizer_class}", flush=True)
    print(f"  Learning Rate: {args.lr}", flush=True)
    print(f"  Weight-Decay: {args.weight_decay}", flush=True)
    print(f"  Unfreeze Layers: {args.unfreeze_layers}", flush=True)
    print(f"  Seed: {args.seed}", flush=True)
    print(f"  Test Split: {args.test_split}", flush=True)
    print(f"  Batch Size: {args.batch_size}", flush=True)
    print(f"  Dropout Probability: {args.dropout_prob}", flush=True)
    print(f"  Primary Key: {args.primary_key}", flush=True)
    print(f"  Output Path: {args.save_results_path}", flush=True)
    print("-" * 27, flush=True)

    # Select optimizer
    optimizer_dict = {"AdamW": AdamW, "Adam": Adam, "SGD": SGD}
    if args.optimizer_class not in optimizer_dict:
        raise ValueError(
            f"Invalid optimizer class: {args.optimizer_class}. "
            f"Choose from {list(optimizer_dict.keys())}."
        )
    optimizer_class = optimizer_dict[args.optimizer_class]

    # Initialize classifier
    classifier = BioClinicalBERTClassifier(
        model_name=args.model_name,
        num_labels=args.num_labels,
        optimizer_class=optimizer_class,
        optimizer_params={"lr": args.lr, "weight_decay": args.weight_decay},
        verbose=args.verbose,
        seed=args.seed,
        batch_size=args.batch_size,
        dropout_prob=args.dropout_prob,
        output_path=args.save_results_path
    )

    # Unfreeze specified BERT layers
    if args.unfreeze_layers > 0:
        classifier.unfreeze_last_layers(args.unfreeze_layers)

    # Load existing model weights if provided
    if args.model_weight_path:
        classifier.load_model(args.model_weight_path)

    print("Everything set up, time to train", flush=True)
    # Run training and save validation predictions

    # Check if results_summary.csv exists in the output directory
    if args.save_results_path and os.path.exists(os.path.join(args.save_results_path, 'results_summary.csv')):
        print(f"Found existing results_summary.csv in output directory: {args.save_results_path}")
    else:
        results = classifier._run_train_epoch(
            df,
            num_epochs=args.num_epochs,
            primary_key=args.primary_key,
            test_split=args.test_split,
            early_stop_patience=args.early_stop_patience,
            shuffle_train=True,
            text_column=args.text_column,
            label_column=args.label_column,
            debug=args.debug,
            print_every=args.print_every
        )

        # Save fine-tuned model
        classifier.save_model(args.save_model_path)
        print(f"Fine-tuned model saved to {args.save_model_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BioClinicalBERT")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str,
                        default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--text_column", type=str, default="TEXT")
    parser.add_argument("--label_column", type=str, default="LABEL")
    parser.add_argument("--model_weight_path", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, required=True)
    parser.add_argument("--optimizer_class", type=str, default="AdamW")
    parser.add_argument("--unfreeze_layers", type=int, default=0)
    parser.add_argument(
        "--primary_key", type=str, default=None,
        help="Column name to use as primary key for CSV predictions"
    )
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--dropout_prob", type=float, default=None)
    parser.add_argument("--save_results_path", type=str, default=None)
    args = parser.parse_args()

    main(args)

# To run this script; use the below on terminal:
# python3 train.py \
#   --data_path path/to/your/data.csv \
#   --model_name emilyalsentzer/Bio_ClinicalBERT \
#   --num_labels 2 \
#   --lr 1e-5 \
#   --weight_decay 0.1 \
#   --batch_size 64 \
#   --seed 8 \
#   --num_epochs 100 \
#   --test_split 0.2 \
#   --text_column TEXT \
#   --label_column LABEL \
#   --model_weight_path path/to/existing_weights.pt \
#   --save_model_path path/to/fine_tuned_model.pt \
#   --optimizer_class AdamW \
#   --unfreeze_layers 0 \
#   --primary_key your_primary_key_column \
#   --verbose \
#   --debug \
#   --print_every 10 \
#   --early_stop_patience 10 \
#   --dropout_prob 0.1

