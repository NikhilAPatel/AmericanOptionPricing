#!/bin/bash

# Set the number of threads and simulations as variables
NUM_THREADS=32
NUM_SIMULATIONS=10000

# File to store output
output_file="out.txt"

# Clear the output file if it already exists
> "$output_file"

echo "Running $NUM_SIMULATIONS simulations with $NUM_THREADS threads..." >> "$output_file"
echo "" >> "$output_file"

# Navigate and run each folder with specific command-line arguments
# AutoManager
echo "Processing AutoManager..."
cd AutoManager
g++-13 -o main -fopenmp main.cpp && ./main $NUM_THREADS $NUM_SIMULATIONS >> "../$output_file"
cd ..
echo "" >> "$output_file"

# ClassicalSequential
echo "Processing ClassicalSequential..."
cd ClassicalSequential
g++-13 -o main -fopenmp main.cpp && ./main $NUM_SIMULATIONS >> "../$output_file"
cd ..
echo "" >> "$output_file"

# EasyParallelization
echo "Processing EasyParallelization..."
cd EasyParallelization
g++-13 -o main -fopenmp main.cpp && ./main $NUM_THREADS $NUM_SIMULATIONS >> "../$output_file"
cd ..
echo "" >> "$output_file"

# HybridParallelization
echo "Processing HybridParallelization..."
cd HybridParallelization
g++-13 -o main -fopenmp main.cpp && ./main  $NUM_THREADS $NUM_SIMULATIONS >> "../$output_file"
cd ..
echo "" >> "$output_file"

echo "Execution completed. Check $output_file for outputs."
