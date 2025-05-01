#!/usr/bin/env python

import csv
import itertools
import random


# Define all possible values for each parameter
optimisers = ["AdamW", "Adam", "SGD"]
learning_rates = [0.005, 0.0001, 0.0005]
layer_unfreeze = [0, 1, 2]
random.seed(42)  # For reproducibility
seeds = random.sample(range(1, 100), 10)  # 10 random numbers between 1 and 999
dropout = [0.15, 0.2, 0.25]

# Open file for writing
with open('Slurm/parameter_search2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['optimiser', 'learning_rate', 'dropout' ,'layer_unfreeze', 'seed'])
    
    # Generate all combinations and write rows
    for opt, lr, drop, model, seed in itertools.product(optimisers, learning_rates, dropout, layer_unfreeze, seeds):
        writer.writerow([opt, lr, drop, model, seed])

print(f"CSV created with {len(optimisers) * len(learning_rates) * len(dropout) * len(layer_unfreeze) * len(seeds)} configurations")