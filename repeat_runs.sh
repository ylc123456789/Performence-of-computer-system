#!/bin/bash

# Check and install required Python libraries
check_and_install_python_libs() {
    echo "=== Checking Python Dependencies ==="
    if ! command -v pip3 &> /dev/null; then
        echo "pip3 not found, installing python3-pip..."
        sudo apt-get update && sudo apt-get install -y python3-pip
    fi

    required_libs=("scipy" "pandas" "numpy")
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

if [ $# -ne 3 ]; then
    echo "Usage: $0 <algorithm> <queue_policy> <num_runs>"
    echo "Example: $0 cubic RED 5"
    exit 1
fi

ALGO="$1"
QUEUE="$2"
NUM_RUNS="$3"

VALID_ALGOS=("reno" "cubic" "yeah" "vegas")
VALID_QUEUES=("DropTail" "RED")
if ! [[ " ${VALID_ALGOS[@]} " =~ " ${ALGO} " ]]; then
    echo "Error: Unsupported algorithm! Available: ${VALID_ALGOS[*]}"
    exit 1
fi
if ! [[ " ${VALID_QUEUES[@]} " =~ " ${QUEUE} " ]]; then
    echo "Error: Unsupported queue policy! Available: ${VALID_QUEUES[*]}"
    exit 1
fi
if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [ "$NUM_RUNS" -lt 1 ]; then
    echo "Error: Number of runs must be a positive integer!"
    exit 1
fi

SCENARIO="${ALGO}_${QUEUE}"
BANDWIDTH="1000Mb"
OUT_DIR="repeat_runs_${SCENARIO}_${NUM_RUNS}times"
RUN_LOG="${OUT_DIR}/run_logs"
RESULTS_CSV="${OUT_DIR}/${SCENARIO}_runs_summary.csv"

mkdir -p "${OUT_DIR}" "${RUN_LOG}"

echo "run_id,throughput_Mbps,plr_pct,cov_stability,jain_fairness" > "${RESULTS_CSV}"

echo "=== Starting ${NUM_RUNS} Repeated Experiments (Scenario: ${ALGO}+${QUEUE}) ==="
for ((run=1; run<=NUM_RUNS; run++)); do
    echo -e "\n--- Run ${run}/${NUM_RUNS} ---"
    
    SEED=$(( $(date +%s) + run * 1000 ))
    echo "Random Seed: ${SEED}"
    
    JITTER=$(echo "scale=4; $RANDOM / 65535 * 1.0" | bc)
    echo "Start Time Jitter: ${JITTER}s"
    
    TCL_SCRIPT="${ALGO}Code_${QUEUE}.tcl"
    if [ ! -f "${TCL_SCRIPT}" ]; then
        echo "Error: Script ${TCL_SCRIPT} not found, aborting experiment"
        exit 1
    fi
    
    if [ ! -f "${TCL_SCRIPT}.bak" ]; then
        cp "${TCL_SCRIPT}" "${TCL_SCRIPT}.bak"
        echo "Backed up original script as ${TCL_SCRIPT}.bak"
    fi
    
    sed -i "s/^set bw.*/set bw \"${BANDWIDTH}\"/" "${TCL_SCRIPT}"
    sed -i "s/DropTail/${QUEUE}/g" "${TCL_SCRIPT}"
    sed -i "s/^set seed.*/set seed ${SEED}/" "${TCL_SCRIPT}"
    
    echo "Running Simulation: ns ${TCL_SCRIPT}"
    SEED=${SEED} ns "${TCL_SCRIPT}" > "${RUN_LOG}/run_${run}_sim.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "Warning: Run ${run} simulation failed, log: ${RUN_LOG}/run_${run}_sim.log"
        continue
    fi
    
    TRACE_FILE=$(find . -maxdepth 1 -type f -name "*${ALGO}*.tr" | sort -r | head -n 1)
    if [ -z "${TRACE_FILE}" ]; then
        echo "Warning: No trace file generated for Run ${run}, skipping analysis"
        continue
    fi
    TRACE_DEST="${OUT_DIR}/${SCENARIO}_run${run}.tr"
    mv "${TRACE_FILE}" "${TRACE_DEST}"
    echo "Trace file saved to: ${TRACE_DEST}"
    
    echo "Analyzing Run ${run} Results..."
    ANALYSIS_DIR="${OUT_DIR}/run${run}_analysis"
    mkdir -p "${ANALYSIS_DIR}"
    python3 analyser3.py "${TRACE_DEST}" "${ANALYSIS_DIR}" > "${RUN_LOG}/run_${run}_analysis.log" 2>&1
    
    SUMMARY_CSV="${ANALYSIS_DIR}/algo_summary.csv"
    if [ -f "${SUMMARY_CSV}" ]; then
        line=$(grep "${ALGO}" "${SUMMARY_CSV}")
        if [ -n "$line" ]; then
            THROUGHPUT=$(echo "$line" | cut -d',' -f2)
            PLR=$(echo "$line" | cut -d',' -f3)
            COV=$(echo "$line" | cut -d',' -f4)
            JAIN=$(echo "$line" | cut -d',' -f5)
            echo "${run},${THROUGHPUT},${PLR},${COV},${JAIN}" >> "${RESULTS_CSV}"
            echo "Run ${run} Metrics: Throughput=${THROUGHPUT} Mb/s, PLR=${PLR}%, CoV=${COV}, Jain=${JAIN}"
        else
            echo "Warning: No metric row found for ${ALGO}"
        fi
    else
        echo "Warning: Run ${run} analysis failed, ${SUMMARY_CSV} not found"
    fi
done

if [ -f "${TCL_SCRIPT}.bak" ]; then
    mv "${TCL_SCRIPT}.bak" "${TCL_SCRIPT}"
    echo -e "\nRestored Original TCL Script"
fi

echo -e "\n=== Calculating Statistical Results ==="
python3 - <<END
import pandas as pd
import scipy.stats as stats
import numpy as np

NUM_RUNS = int('${NUM_RUNS}')
RESULTS_CSV = '${RESULTS_CSV}'
OUT_DIR = '${OUT_DIR}'

df = pd.read_csv(RESULTS_CSV)
valid_runs = len(df.dropna())
print(f"Valid Runs: {valid_runs}/{NUM_RUNS}")

if valid_runs < 2:
    print("Warning: Insufficient valid runs to calculate confidence intervals")
else:
    metrics = {
        "throughput_Mbps": "Throughput (Mb/s)",
        "plr_pct": "Packet Loss Rate (%)",
        "cov_stability": "Stability CoV (Lower is Better)",
        "jain_fairness": "Jain Fairness"
    }
    with open(f"{OUT_DIR}/summary_stats.csv", "w") as f:
        f.write("metric,mean,ci_lower,ci_upper\n")
        print("\n===== Mean and 95% Confidence Intervals =====")
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
    print(f"\nStatistical Results Saved to: {OUT_DIR}/summary_stats.csv")

END

echo -e "\n=== All Experiments Completed ==="
echo "Result Directory: ${OUT_DIR}"
