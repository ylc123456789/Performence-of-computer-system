#!/bin/bash

# Check and install required Python libraries
check_and_install_python_libs() {
    echo "=== Checking Python Dependencies ==="
    if ! command -v pip3 &> /dev/null; then
        echo "pip3 not found, installing python3-pip..."
        sudo apt-get update && sudo apt-get install -y python3-pip
    fi

    # Add matplotlib for plotting
    required_libs=("scipy" "pandas" "numpy" "matplotlib")
    for lib in "${required_libs[@]}"; do
        if ! python3 -c "import $lib" &> /dev/null; then
            echo "$lib not found, installing..."
            pip3 install $lib --quiet
        else
            echo "$lib installed"
        fi
    done
}

check_and_install_python_libs

if [ $# -ne 3 ] || [ "$1" != "all" ] || [ "$2" != "RED" ] || [ "$3" -ne 5 ]; then
    echo "Usage: $0 all RED 5"
    exit 1
fi

QUEUE="RED"
NUM_RUNS="$3"
ALGOS=("cubic" "reno" "vegas" "yeah")
BANDWIDTH="1000Mb"

# At the beginning of the script, define a main folder where all the results will be stored.
MAIN_OUTPUT_DIR="repeat_runs"
mkdir -p "${MAIN_OUTPUT_DIR}"

echo "=== Starting ${NUM_RUNS} Repeated Experiments for All Four Algorithms with RED Queue ==="
echo "All results will be stored in the '${MAIN_OUTPUT_DIR}' directory."

for ((run=1; run<=NUM_RUNS; run++)); do
    echo -e "\n--- Run ${run}/${NUM_RUNS} ---"
    
    SEED=$(( $(date +%s) + run * 1000 ))
    echo "Shared Random Seed for This Run: ${SEED}"
    
    for ALGO in "${ALGOS[@]}"; do
        echo -e "\n===== Processing Algorithm: ${ALGO} ====="
        SCENARIO="${ALGO}_${QUEUE}"
        
        # Update the original OUT_DIR path to be under the MAIN_OUTPUT_DIR
        OUT_DIR="${MAIN_OUTPUT_DIR}/repeat_runs_${SCENARIO}_${NUM_RUNS}times"
        RUN_LOG="${OUT_DIR}/run_logs"
        RESULTS_CSV="${OUT_DIR}/${SCENARIO}_runs_summary.csv"
        mkdir -p "${OUT_DIR}" "${RUN_LOG}"

        if [ $run -eq 1 ]; then
            echo "run_id,throughput_Mbps,plr_pct,cov_stability,jain_fairness" > "${RESULTS_CSV}"
        fi

        TCL_SCRIPT="${ALGO}Code_${QUEUE}.tcl"
        if [ ! -f "${TCL_SCRIPT}" ]; then
            echo "Error: Script ${TCL_SCRIPT} not found, skipping ${ALGO} for this run"
            continue
        fi

        if [ ! -f "${TCL_SCRIPT}.bak" ]; then
            cp "${TCL_SCRIPT}" "${TCL_SCRIPT}.bak"
            echo "Backed up original script as ${TCL_SCRIPT}.bak"
        fi

        sed -i "s/^set bw.*/set bw \"${BANDWIDTH}\"/" "${TCL_SCRIPT}"
        sed -i "s/DropTail/${QUEUE}/g" "${TCL_SCRIPT}"
        sed -i "s/^set seed.*/set seed ${SEED}/" "${TCL_SCRIPT}"

        echo "Running Simulation for ${ALGO}: ns ${TCL_SCRIPT}"
        SEED=${SEED} ns "${TCL_SCRIPT}" > "${RUN_LOG}/run_${run}_${ALGO}_sim.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "Warning: ${ALGO} Run ${run} simulation failed, log: ${RUN_LOG}/run_${run}_${ALGO}_sim.log"
            continue
        fi

        TRACE_FILE=$(find . -maxdepth 1 -type f -name "*${ALGO}*.tr" | sort -r | head -n 1)
        if [ -z "${TRACE_FILE}" ]; then
            echo "Warning: No trace file generated for ${ALGO} Run ${run}, skipping analysis"
            continue
        fi
        TRACE_DEST="${OUT_DIR}/${SCENARIO}_run${run}.tr"
        mv "${TRACE_FILE}" "${TRACE_DEST}"
        echo "Trace file for ${ALGO} saved to: ${TRACE_DEST}"

        echo "Analyzing ${ALGO} Run ${run} Results..."
        ANALYSIS_DIR="${OUT_DIR}/run${run}_analysis"
        mkdir -p "${ANALYSIS_DIR}"
        python3 analyser3.py "${TRACE_DEST}" "${ANALYSIS_DIR}" > "${RUN_LOG}/run_${run}_${ALGO}_analysis.log" 2>&1

        SUMMARY_CSV="${ANALYSIS_DIR}/algo_summary.csv"
        if [ -f "${SUMMARY_CSV}" ]; then
            line=$(grep "${ALGO}" "${SUMMARY_CSV}")
            if [ -n "$line" ]; then
                THROUGHPUT=$(echo "$line" | cut -d',' -f2)
                PLR=$(echo "$line" | cut -d',' -f3)
                COV=$(echo "$line" | cut -d',' -f4)
                JAIN=$(echo "$line" | cut -d',' -f5)
                echo "${run},${THROUGHPUT},${PLR},${COV},${JAIN}" >> "${RESULTS_CSV}"
                echo "${ALGO} Run ${run} Metrics: Throughput=${THROUGHPUT} Mb/s, PLR=${PLR}%, CoV=${COV}, Jain=${JAIN}"
            else
                echo "Warning: No metric row found for ${ALGO}"
            fi
        else
            echo "Warning: ${ALGO} Run ${run} analysis failed, ${SUMMARY_CSV} not found"
        fi
    done
done

# Restore all original TCL scripts
for ALGO in "${ALGOS[@]}"; do
    TCL_SCRIPT="${ALGO}Code_${QUEUE}.tcl"
    if [ -f "${TCL_SCRIPT}.bak" ]; then
        mv "${TCL_SCRIPT}.bak" "${TCL_SCRIPT}"
    fi
done
echo -e "\nRestored All Original TCL Scripts"

echo -e "\n=== Calculating Statistical Results and Generating Plots for Each Algorithm ==="
for ALGO in "${ALGOS[@]}"; do
    SCENARIO="${ALGO}_${QUEUE}"
    
    # Update the paths for the statistics and graph sections
    OUT_DIR="${MAIN_OUTPUT_DIR}/repeat_runs_${SCENARIO}_${NUM_RUNS}times"
    RESULTS_CSV="${OUT_DIR}/${SCENARIO}_runs_summary.csv"
    
    if [ -f "${RESULTS_CSV}" ]; then
        python3 - <<END
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os

# Set up Chinese font support
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# Set plot style
plt.style.use('default')

ALGO = "${ALGO}"
QUEUE = "${QUEUE}"
NUM_RUNS = int('${NUM_RUNS}')
RESULTS_CSV = '${RESULTS_CSV}'
OUT_DIR = '${OUT_DIR}'

# Read data
df = pd.read_csv(RESULTS_CSV)
valid_runs = len(df.dropna())
print(f"\n===== Statistical Results for {ALGO} =====")
print(f"Valid Runs: {valid_runs}/{NUM_RUNS}")

# Generate statistical results
summary_stats_path = f"{OUT_DIR}/summary_stats.csv"
if valid_runs >= 2:
    metrics = {
        "throughput_Mbps": "Throughput (Mb/s)",
        "plr_pct": "Packet Loss Rate (%)",
        "cov_stability": "Stability CoV (Lower is Better)",
        "jain_fairness": "Jain Fairness"
    }
    with open(summary_stats_path, "w") as f:
        f.write("metric,mean,ci_lower,ci_upper\n")
        for col, name in metrics.items():
            data = df[col].dropna()
            mean = data.mean().round(4)
            std = data.std()
            if std == 0:
                ci_lower = mean
                ci_upper = mean
            else:
                ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
                ci_lower = round(ci[0], 4)
                ci_upper = round(ci[1], 4)
            f.write(f"{col},{mean},{ci_lower},{ci_upper}\n")
            print(f"{name}:")
            print(f"  Mean: {mean}")
            print(f"  95% CI: [{ci_lower}, {ci_upper}]")
    print(f"Statistical Results Saved to: {summary_stats_path}")
else:
    print("Warning: Insufficient valid runs to calculate confidence intervals")

# ----------------------
# Generate Plot 1: Trend Plot of Metrics Over Multiple Runs
# ----------------------
if valid_runs >= 1:
    plt.figure(figsize=(10, 6))
    
    # Plot Throughput
    plt.plot(df['run_id'], df['throughput_Mbps'], 'b-', marker='o', label='Total Throughput (Mbps)')
    # Plot PLR
    plt.plot(df['run_id'], df['plr_pct'], 'y-', marker='s', label='Avg PLR (%)')
    # Plot Stability CoV
    plt.plot(df['run_id'], df['cov_stability'], 'k-', marker='^', label='Stability CoV')
    # Plot Jain's Fairness Index
    plt.plot(df['run_id'], df['jain_fairness'], 'orange', marker='d', label="Jain's Index")
    
    plt.title(f'{ALGO} + {QUEUE} run {NUM_RUNS} times with random seed')
    plt.xlabel('Run ID')
    plt.ylabel('Value')
    plt.xticks(df['run_id'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    trend_plot_path = f"{OUT_DIR}/{ALGO}_{QUEUE}_trend.png"
    plt.savefig(trend_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trend plot saved to: {trend_plot_path}")
else:
    print("Warning: Insufficient data to generate trend plot")

# ----------------------
# Generate Plot 2: 95% Confidence Interval Plot
# ----------------------
if valid_runs >= 2 and os.path.exists(summary_stats_path):
    stats_df = pd.read_csv(summary_stats_path)
    
    # Prepare plotting data
    metrics = stats_df['metric']
    means = stats_df['mean']
    ci_lower = stats_df['ci_lower']
    ci_upper = stats_df['ci_upper']
    
    # Calculate error range (upper CI - mean)
    errors = ci_upper - means
    
    plt.figure(figsize=(10, 6))
    
    # Plot confidence interval bounds
    bar_width = 0.35
    x = np.arange(len(metrics))
    
    # Plot lower CI
    plt.bar(x, ci_lower, bar_width, label='95% CI Lower', color='lightgray')
    # Plot upper CI (stacked on lower CI)
    plt.bar(x, ci_upper - ci_lower, bar_width, bottom=ci_lower, label='95% CI Upper', color='darkgray')
    # Plot mean line
    plt.plot(x, means, 'orange', marker='o', linewidth=2, label='Average')
    
    plt.title(f'95% Confidence Interval for {ALGO} + {QUEUE}')
    plt.xticks(x, [m.replace('_', ' ') for m in metrics])  # Beautify x-axis labels
    plt.ylabel('Value')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    ci_plot_path = f"{OUT_DIR}/{ALGO}_{QUEUE}_confidence_interval.png"
    plt.savefig(ci_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence interval plot saved to: {ci_plot_path}")
else:
    print("Warning: Insufficient data to generate confidence interval plot")
END
    else
        echo "Warning: No results CSV found for ${ALGO}, skipping statistical analysis and plotting"
    fi
done

echo -e "\n=== All Experiments and Plots Completed ==="
echo "All results are stored in the '${MAIN_OUTPUT_DIR}' directory."
