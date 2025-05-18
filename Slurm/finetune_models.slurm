#!/bin/bash
#SBATCH --job-name=BioClinBERT-Finetune_CPU                       # Name of the job
#SBATCH --output=/bigdata/InghamProjects/kineticInjury/Kinetic-Injury-Triage/Slurm_Logs/%x_%A_%a.out      # Standard output file (%x=job name, %A=job ID, %a=array index)           
#SBATCH --error=/bigdata/InghamProjects/kineticInjury/Kinetic-Injury-Triage/Slurm_Logs/%x_%A_%a.err       # Error output file            
#SBATCH --partition=cpu                                       # Use the CPU partition
##SBATCH --partition=ampere40                                  # Use the GPU partition
##SBATCH --gres=gpu:0                                          # No GPU requested explicitly (may use default allocation)
#SBATCH --time=0-12:00:00                                     # Maximum job runtime (days-hours:minutes:seconds)
#SBATCH --cpus-per-task=1                                     # Request 1 CPU per task
#SBATCH --mem=4G                                              # Request 4GB of RAM
##SBATCH --mem=31G                                             # Request 31GB of RAM
#SBATCH --array=0-239                                         # Run 240 array jobs (indices 0-239)


# Set project directory path
PROJECT_DIR="/bigdata/InghamProjects/kineticInjury/Kinetic-Injury-Triage"

# Load PyTorch module with Python 3.10
module load PyTorch/Python3.10   

# Define input data path and copy to local /tmp for faster access
data_path="$PROJECT_DIR/Data/Kinetic_Injury_Finetune_Data.csv"
cp "${data_path}" "/tmp/train.csv"
data_path="/tmp/train.csv"

# Define data columns configuration
text_column="ED_Triage_Comment"                # Column containing text data
label_column="Label"                           # Column containing labels
primary_key="Encntr_ID"                        # Primary key column

# Define model configuration
model_name="emilyalsentzer/Bio_ClinicalBERT"   # Base model to use
num_labels=2                                   # Binary classification
num_epochs=200                                 # Maximum training epochs
batch_size=64                                  # Batch size for training
print_every=1                                  # Print metrics every epoch
early_stop_patience=10                         # Stop if no improvement after 10 epochs

                     

# Set fixed hyperparameters
weight_decays=0.01                            # L2 regularization strength
test_splits=0.2                               # 20% of data for testing

# Get array job index and extract parameters from CSV
LINE_NUM=$((${SLURM_ARRAY_TASK_ID} + 2))      # +2 to skip header row and 0-indexing
PARAMS=$(sed -n "${LINE_NUM}p" ${PROJECT_DIR}/Slurm/finetune_parameters.csv)

# Parse hyperparameters from the CSV row
OPTIMISER=$(echo $PARAMS | cut -d, -f1 | tr -d '\r')       # Column 1: optimizer type
LEARNING_RATE=$(echo $PARAMS | cut -d, -f2 | tr -d '\r')   # Column 2: learning rate
DROPOUT=$(echo $PARAMS | cut -d, -f3 | tr -d '\r')         # Column 3: dropout probability
LAYER_UNFREEZE=$(echo $PARAMS | cut -d, -f4 | tr -d '\r')  # Column 4: number of layers to unfreeze
SEED=$(echo $PARAMS | cut -d, -f5 | tr -d '\r')            # Column 5: random seed



# Print parameters for this run
echo "Running with parameters:"
echo "Optimiser: $OPTIMISER"
echo "Learning Rate: $LEARNING_RATE"
echo "Dropout: $DROPOUT"
echo "Layers to Unfreeze: $LAYER_UNFREEZE"
echo "Seed: $SEED"

# Create output directory structure for saving model and results
# BASE_DIR="$PROJECT_DIR/Outputs/models/bcbert_runs"
BASE_DIR="$PROJECT_DIR/Outputs/models/cpu_finetune"
mkdir -p "${BASE_DIR}"

# Create a unique directory name based on hyperparameters
run_tag="${OPTIMISER}_lr-${LEARNING_RATE}_dropout-${DROPOUT}_unf-${LAYER_UNFREEZE}_seed-${SEED}"
out_dir="${BASE_DIR}/${run_tag}"
mkdir -p "${out_dir}"


# Build command line arguments array for training script
cli=(
--data_file       "${data_path}"                    # Path to input data
--save_model_path  "${out_dir}/model_finetune.pt"   # Where to save model weights
--weight_file      "${out_dir}/model.pt"            # The model weights to load
--save_results_path "${out_dir}"                    # Where to save results
--model_name       "${model_name}"                  # Base model name
--num_labels       "${num_labels}"                  # Number of classification labels
--lr               "${LEARNING_RATE}"               # Learning rate from parameters
--weight_decay     "${weight_decays}"               # L2 regularization strength
--batch_size       "${batch_size}"                  # Training batch size
--seed             "${SEED}"                        # Random seed from parameters
--num_epochs       "${num_epochs}"                  # Maximum training epochs
--test_split       "${test_splits}"                 # Test set proportion
--text_column      "${text_column}"                 # Column containing text data
--label_column     "${label_column}"                # Column containing labels
--optimizer_class  "${OPTIMISER}"                   # Optimizer type from parameters
--unfreeze_layers  "${LAYER_UNFREEZE}"              # Layers to unfreeze from parameters
--early_stop_patience "${early_stop_patience}"      # Early stopping patience
--dropout_prob     "${DROPOUT}"                     # Dropout rate from parameters
--print_every      "${print_every}"                 # Print frequency
--verbose                                           # Enable verbose output
--debug                                             # Enable debug information
--finetune                                          # Enable fine-tuning
)

# Add optional arguments if variables are defined
[[ -n "$primary_key"       ]] && cli+=( --primary_key "$primary_key" )

# Print run tag and execute fine-tuning script with all parameters
echo "Running:  ${run_tag}"
python3.10 $PROJECT_DIR/Scripts/finetune.py "${cli[@]}"

# Print completion message
echo "All done at: $(date) | Host: $(hostname)"
echo "----------------------------------------"
# Clean up temporary data file
# rm -rf "${data_path}"