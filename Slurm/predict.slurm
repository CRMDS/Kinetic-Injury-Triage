#!/bin/bash
##SBATCH --job-name=BioClinBERT-Predict                       # Name of the job
#SBATCH --job-name=BioClinBERT-Finetune-Predict                       # Name of the job
#SBATCH --output=/bigdata/InghamProjects/kineticInjury/Kinetic-Injury-Triage/Slurm_Logs/%x_%A_%a.out      # Standard output file (%x=job name, %A=job ID, %a=array index)           
#SBATCH --error=/bigdata/InghamProjects/kineticInjury/Kinetic-Injury-Triage/Slurm_Logs/%x_%A_%a.err       # Error output file            
#SBATCH --partition=cpu                                       # Use the CPU partition
#SBATCH --time=0-13:00:00                                     # Maximum job runtime (days-hours:minutes:seconds)
#SBATCH --cpus-per-task=1                                     # Request 1 CPU per task
#SBATCH --mem=4G                                              # Request 4GB of RAM
##SBATCH --array=0-809                                       # Run 810 array jobs (indices 0-809)
#SBATCH --array=0-239                                         # Run 240 array jobs (indices 0-239)
##SBATCH -d afterok:13873                                         

echo "Starting at: $(date) | Host: $(hostname)"
echo "----------------------------------------"


# Set project directory path
PROJECT_DIR="/bigdata/InghamProjects/kineticInjury/Kinetic-Injury-Triage"

# Load PyTorch module with Python 3.10
module load PyTorch/Python3.10  

source ~/.bashrc

# Define input data path and copy to local /tmp for faster access
data_path="$PROJECT_DIR/Data/Kinetic_Injury_Test_Data.csv"
cp "${data_path}" "/tmp/test.csv"
data_path="/tmp/test.csv"

# Define data columns configuration
text_column="ED_Triage_Comment"                # Column containing text data
label_column="Label"                           # Column containing labels
primary_key="Encntr_ID"                        # Primary key column

# Get array job index and extract parameters from CSV
LINE_NUM=$((${SLURM_ARRAY_TASK_ID} + 2))      # +2 to skip header row and 0-indexing
# PARAMS=$(sed -n "${LINE_NUM}p" ${PROJECT_DIR}/Slurm/parameter_search.csv)
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
BASE_DIR="$PROJECT_DIR/Outputs/models/bcbert_runs"
mkdir -p "${BASE_DIR}"

# Create a unique directory name based on hyperparameters
run_tag="${OPTIMISER}_lr-${LEARNING_RATE}_dropout-${DROPOUT}_unf-${LAYER_UNFREEZE}_seed-${SEED}"
out_dir="${BASE_DIR}/${run_tag}"
mkdir -p "${out_dir}"


# Build command line arguments array for training script
cli=(
--data_file       "${data_path}"                    # Path to input data
# --weight_file      "${out_dir}/model.pt"            # The model weights to load
--weight_file      "${out_dir}/model_finetune.pt"   # The model weights to load
--save_results_path "${out_dir}"                    # Where to save results
--text_column      "${text_column}"                 # Column containing text data
--label_column     "${label_column}"                # Column containing labels
--predict                                           # Flag to indicate prediction mode
--finetune                                         # Flag to indicate fine-tuning mode
)

# Add optional arguments if variables are defined
[[ -n "$primary_key"       ]] && cli+=( --primary_key "$primary_key" )

# Print run tag and execute fine-tuning script with all parameters
echo "Running:  ${run_tag}"
python3.10 $PROJECT_DIR/Scripts/predict.py "${cli[@]}"

# Print completion message
echo "All done at: $(date) | Host: $(hostname)"
echo "----------------------------------------"
