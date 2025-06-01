#!/usr/bin/env nextflow

// using DSL2 -- the newer nextflow syntax
nextflow.enable.dsl=2

// Parameters 
params.base_dir = "${params.project_dir}/Outputs/models/bcbert_runs"
params.param_csv = "${params.project_dir}/parameter_search.csv"
//params.script = "${params.project_dir}/Scripts/train.py"
params.data_path = "${params.project_dir}/Data/labelled_kinetic_noteevents_2k.csv"
params.testdata_path = "${params.project_dir}/Data/Kinetic_Injury_Test_Data.csv"

// Set up channel
Channel
    .fromPath(params.param_csv)
    .splitCsv(skip: 1, header: false)
    .map { row -> 
   	def optimiser = row[0].trim()
        def learning_rate = row[1].trim()
        def dropout = row[2].trim()
        def unfreeze = row[3].trim()
        def seed = row[4].trim()
        def tag = "${optimiser}_lr-${learning_rate}_dropout-${dropout}_unf-${unfreeze}_seed-${seed}"

        return [
            optimiser: optimiser,
            learning_rate: learning_rate,
            dropout: dropout,
            unfreeze: unfreeze,
            seed: seed,
            tag: tag
        ]
    }
    .set { param_rows_ch } 

process train {

    memory = '48GB'
    time = '1h'
    cpus = 12
    gpus = 1

    clusterOptions = '-l jobfs=20GB,wd'

    input:
    val config 

    //output:
    //file "${config.tag}.out"

    // Possible output of the configuration to train results channel, so predict can use it
    //output: 
    //val(config) into train_results_ch

    script:
    """
   
    module load pytorch/1.10.0
    source $HOME/envs/kit/bin/activate

    # create output path
    mkdir -p ${params.base_dir}/${config.tag}

    # Print out all the numbers so we know things are running correctly
    echo "Running with parameters:"
    echo "Optimiser: ${config.optimiser}"
    echo "Learning Rate: ${config.learning_rate}"
    echo "Dropout: ${config.dropout}"
    echo "Layers to Unfreeze: ${config.unfreeze}"
    echo "Seed: ${config.seed}"

    python3 -u -B ${params.project_dir}/Scripts/train.py \\
        --data_path ${params.data_path} \\
        --save_model_path ${params.base_dir}/${config.tag}/model.pt \\
        --save_results_path ${params.base_dir}/${config.tag} \\
        --model_name emilyalsentzer/Bio_ClinicalBERT \\
	--local_model_path "/scratch/mp72/kineticInjury/huggingface/hub/models--emilyalsentzer--Bio_ClinicalBERT" \\
        --num_labels 2 \\
        --lr ${config.learning_rate} \\
        --weight_decay 0.01 \\
        --batch_size 64 \\
        --seed ${config.seed} \\
        --num_epochs 200 \\
        --test_split 0.2 \\
        --text_column TEXT \\
        --label_column LABEL \\
        --primary_key HADM_ID \\
        --optimizer_class ${config.optimiser} \\
        --unfreeze_layers ${config.unfreeze} \\
        --early_stop_patience 10 \\
        --dropout_prob ${config.dropout} \\
        --print_every 1 \\
        --verbose \\
        --debug \\
        > ${params.project_dir}/PBS_Logs/${config.tag}.out \\
	    2> ${params.project_dir}/PBS_Logs/${config.tag}.err

    echo "All done at: \$(date) | Host: \$(hostname)"

    deactivate

    """
}

process predict {

    memory = '4GB'
    time = '1h'
    cpus = 1
    queue = 'normal'

    clusterOptions = '-l jobfs=20GB,wd'

    input:
    val config 

    //output:
    //file "${config.tag}.out"

    script:
    """
   
    module load pytorch/1.10.0
    source $HOME/envs/kit/bin/activate

    # No need to create output path here, as it is already created in the train process

    # Print out all the numbers so we know things are running correctly
    echo "Running with parameters:"
    echo "Optimiser: ${config.optimiser}"
    echo "Learning Rate: ${config.learning_rate}"
    echo "Dropout: ${config.dropout}"
    echo "Layers to Unfreeze: ${config.unfreeze}"
    echo "Seed: ${config.seed}"

    python3 -u -B ${params.project_dir}/Scripts/predict.py \\
        --data_file ${params.testdata_path} \\
        --weight_file ${params.base_dir}/${config.tag}/model.pt \\
        --save_results_path ${params.base_dir}/${config.tag} \\
        --text_column ED_Triage_Comment \\
        --label_column Label \\
        --primary_key Encntr_ID \\
        --predict \\
        > ${params.project_dir}/PBS_Logs/${config.tag}_pred.out \\
        2> ${params.project_dir}/PBS_Logs/${config.tag}_pred.err

    echo "All done at: \$(date) | Host: \$(hostname)"

    deactivate

    """
}

// Call the process inside a workflow block
workflow {
    //train(param_rows_ch)
    predict(param_rows_ch)
}

// if you want to pipe the output of train to predict, do the following in the workflow block
// workflow {
//     train(param_rows_ch) | predict
// }
