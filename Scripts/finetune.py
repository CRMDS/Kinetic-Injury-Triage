import os
import argparse
import pandas as pd
import warnings
from torch.optim import AdamW, Adam, SGD


warnings.filterwarnings("ignore", message=".*CUDA.*")
from Bio_ClinicalBERTClassifier import BioClinicalBERTClassifier




def main(args):
    # Load dataset
    try:
        dataset = pd.read_csv(args.data_file)
    except Exception as e:
        print(f"Error reading data file {args.data_file}: {e}")
        return

    # Verify that the required columns exist
    if args.text_column not in dataset.columns:
        print(f"Error: Text column '{args.text_column}' not found in the dataset.")
        return
    if args.label_column not in dataset.columns:
        print(f"Error: Label column '{args.label_column}' not found in the dataset.")
        return

    # Process each model weight file
    print(f"\nFine tuning model: {os.path.relpath(args.weight_file)}")

    optimizer_dict = {"AdamW": AdamW, "Adam": Adam, "SGD": SGD}
    if args.optimizer_class not in optimizer_dict:
        raise ValueError(
            f"Invalid optimizer class: {args.optimizer_class}. "
            f"Choose from {list(optimizer_dict.keys())}."
        )
    optimizer_class = optimizer_dict[args.optimizer_class]


    classifier = BioClinicalBERTClassifier(
        model_name=args.model_name,
        num_labels=args.num_labels,
        optimizer_class=optimizer_class,
        optimizer_params={"lr": args.lr, "weight_decay": args.weight_decay},
        verbose=args.verbose,
        seed=args.seed,
        batch_size=args.batch_size,
        dropout_prob=args.dropout_prob,
        output_path=args.save_results_path,
        fine_tune_run=args.finetune
    )

    # Unfreeze specified BERT layers
    if args.unfreeze_layers > 0:
        classifier.unfreeze_last_layers(args.unfreeze_layers)

    if args.save_results_path and os.path.exists(os.path.join(args.save_results_path, 'model_finetune.pt')):
        print(f"Found existing results_summary.csv in output directory: {args.save_results_path}")
    else:
        # Define the save path for the fine-tuned model (saved in the same directory as the original weight file)
        # Fine tune the model on the dataset
        training_results = classifier.fine_tune(
            model_wt_path=args.weight_file,
            dataset=dataset,
            text_column=args.text_column,
            label_column=args.label_column,
            save_model_path=args.save_model_path,
            num_epochs=args.num_epochs,
            debug=args.debug,
            print_every=args.print_every,
            early_stop_patience=args.early_stop_patience
        )
        
        print(f"\nFine tuning results for model {os.path.relpath(args.weight_file)}:")
        print(training_results)
        print(f"Fine-tuned model saved to: {args.save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to use Bio_ClinicalBERTClassifier for predictions and fine tuning."
    )
    parser.add_argument("--weight_file", type=str, required=True, help="Directory containing .pt file")
    parser.add_argument("--data_file", type=str, required=True, help="Path to new dataset CSV file")
    parser.add_argument("--text_column", type=str, default="TEXT", help="Name of text column in the dataset")
    parser.add_argument("--label_column", type=str, default="LABEL", help="Name of label column in the dataset")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience for fine tuning")
    parser.add_argument("--model_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT", 
                        help="Name or path of the pre-trained model to use")
    parser.add_argument("--num_labels", type=int, default=2, 
                        help="Number of classification labels")
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1, 
                        help="Weight decay (L2 penalty) for optimizer")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--seed", type=int, default=8, 
                        help="Random seed for reproducibility")
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Maximum number of training epochs")
    parser.add_argument("--test_split", type=float, default=0.2, 
                        help="Fraction of data to use for testing")
    parser.add_argument("--save_model_path", type=str, required=True, 
                        help="Path to save the trained model")
    parser.add_argument("--optimizer_class", type=str, default="AdamW", 
                        help="Optimizer class to use (AdamW, Adam, SGD)")
    parser.add_argument("--unfreeze_layers", type=int, default=0, 
                        help="Number of transformer layers to unfreeze for fine-tuning")
    parser.add_argument(
        "--primary_key", type=str, default=None,
        help="Column name to use as primary key for CSV predictions"
    )
    parser.add_argument("--verbose", action="store_true", default=True, 
                        help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", default=True, 
                        help="Enable debug information")
    parser.add_argument("--print_every", type=int, default=10, 
                        help="Print metrics every N epochs")
    parser.add_argument("--dropout_prob", type=float, default=None, 
                        help="Dropout probability for model layers")
    parser.add_argument("--save_results_path", type=str, default=None, 
                        help="Path to save training results and metrics")
    parser.add_argument("--finetune", default=False, action="store_true",
                        help="Flag to indicate if fine-tuning should be performed")
    parser.add_argument("--predict", default=False, action="store_true",
                        help="Flag to indicate if predictions should be made")

    args = parser.parse_args()

    main(args)