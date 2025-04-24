import argparse
import pandas as pd
from Bio_ClinicalBERTClassifier import BioClinicalBERTClassifier


def main(args):
    # Load input CSV
    df = pd.read_csv(args.input_path)

    # Extract texts
    if args.text_column not in df:
        raise ValueError(f"Text column '{args.text_column}' not found in input file.")
    texts = df[args.text_column].tolist()

    # Optional primary key
    primary = df[args.primary_key] if args.primary_key and args.primary_key in df else None

    # Optional true labels
    true_labels = df[args.true_label_column].tolist() if args.true_label_column and args.true_label_column in df else None

    # Initialize and load model
    clf = BioClinicalBERTClassifier(
        model_name=args.model_name
    )
    clf.load_model(args.model_weight_path)

    # Run prediction and write CSV 
    clf.predict(
        texts,
        primary_key=primary,
        true_labels=true_labels,
        output_csv=True,
        max_length=args.max_length
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Predict with BioClinicalBERTClassifier"
    )
    parser.add_argument("--input_path", required=True,
                        help="CSV file containing text for prediction")
    parser.add_argument("--model_weight_path", required=True,
                        help="Path to trained .pt weights")
    parser.add_argument("--model_name", default="emilyalsentzer/Bio_ClinicalBERT",
                        help="HuggingFace model name or local path")
    parser.add_argument("--text_column", default="TEXT",
                        help="Name of column containing input text")
    parser.add_argument("--true_label_column", default="LABEL",
                        help="Column name for true labels (optional)")
    parser.add_argument("--primary_key", default=None,
                        help="Column name for primary keys (optional)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for tokenizer")

    args = parser.parse_args()
    main(args)

# To run this script; use the below on terminal:
#  python3 predict.py \
#   --input_path      train.csv \          
#   --model_weight_path fine_tuned.pt \   
#   --primary_key     ROW_ID \ 
#   --text_column     TEXT \
#   --true_label_column LABEL
