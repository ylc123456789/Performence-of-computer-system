#!/bin/bash

# Configuration Parameters
OUTPUT_DIR="run_all(results)"
ALGOS=("reno" "cubic" "yeah" "vegas")
TCP_TRACE_DIR="./"
DT_DIRS=("runs_dt_500M" "runs_dt_2G")
RED_DIRS=("runs_red_500M" "runs_red_2G")

# Create Output Directory
mkdir -p $OUTPUT_DIR

# Step 1: Run Simulations to Generate Trace Files (Part A)
echo "=== Running TCP Algorithm Simulations ==="
for algo in "${ALGOS[@]}"; do
    tcl_script="${algo}Code.tcl"
    if [ -f "$tcl_script" ]; then
        echo "Running ns $tcl_script..."
        ns $tcl_script
        # Move Generated Trace File to Designated Directory
        trace_file=$(find . -maxdepth 1 -type f -name "*${algo}*.tr" | sort -r | head -n 1)
        if [ -n "$trace_file" ]; then
            mv "$trace_file" "${TCP_TRACE_DIR}/${algo}Trace.tr"
        fi
    else
        echo "Warning: $tcl_script not found, skipping this algorithm"
    fi
done

# Step 2: Perform Performance Analysis (Part A and Part B)
echo -e "\n=== Performing Data Analysis ==="

# Run Part A Analysis (Generate flows_summary.csv, algo_summary.csv, etc.)
echo "Running Part A Analysis..."
python3 analyser3.py $OUTPUT_DIR

# Run Part B Baseline Comparison (DropTail vs RED)
echo "Running Part B Baseline Comparison Analysis..."
python3 analyser3.py --compare runs_dt runs_red $OUTPUT_DIR

# Run Part B Sensitivity Analysis (Comparison Under Different Bandwidths)
echo "Running Part B Sensitivity Analysis..."
python3 analyser3.py --sensitivity "${DT_DIRS[@]}" "${RED_DIRS[@]}" $OUTPUT_DIR

# Step 3: Supplement with Other Analysis Scripts (e.g., End-to-End Delay, Jitter Analysis)
echo -e "\n=== Generating Additional Metric Analysis ==="

# Run End-to-End Delay Analysis
python3 analyser2.py

# Generate Jitter Data
echo "Calculating Jitter Metrics..."
if [ -f "out.tr" ]; then
    bash Test/jitter.sh
    mv jitter.txt $OUTPUT_DIR/
else
    echo "Warning: out.tr not found, unable to calculate jitter"
fi

# Step 4: Organize Output Files
echo -e "\n=== Analysis Completed ==="
echo "All results have been saved to: $OUTPUT_DIR"
ls -l $OUTPUT_DIR
