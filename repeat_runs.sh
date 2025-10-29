#!/bin/bash

# 检查并安装必要的Python库
check_and_install_python_libs() {
    echo "=== 检查Python依赖库 ==="
    if ! command -v pip3 &> /dev/null; then
        echo "未找到pip3，尝试安装python3-pip..."
        sudo apt-get update && sudo apt-get install -y python3-pip
    fi

    required_libs=("scipy" "pandas" "numpy")
    for lib in "${required_libs[@]}"; do
        if ! python3 -c "import $lib" &> /dev/null; then
            echo "未找到$lib，正在安装..."
            pip3 install $lib --quiet
        else
            echo "$lib已安装"
        fi
    done
}

check_and_install_python_libs

if [ $# -ne 3 ]; then
    echo "使用方式: $0 <算法名称> <队列策略> <运行次数>"
    echo "示例: $0 cubic RED 5"
    exit 1
fi

ALGO="$1"
QUEUE="$2"
NUM_RUNS="$3"

VALID_ALGOS=("reno" "cubic" "yeah" "vegas")
VALID_QUEUES=("DropTail" "RED")
if ! [[ " ${VALID_ALGOS[@]} " =~ " ${ALGO} " ]]; then
    echo "错误：不支持的算法！可选算法：${VALID_ALGOS[*]}"
    exit 1
fi
if ! [[ " ${VALID_QUEUES[@]} " =~ " ${QUEUE} " ]]; then
    echo "错误：不支持的队列策略！可选策略：${VALID_QUEUES[*]}"
    exit 1
fi
if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [ "$NUM_RUNS" -lt 1 ]; then
    echo "错误：运行次数必须是正整数！"
    exit 1
fi

SCENARIO="${ALGO}_${QUEUE}"
BANDWIDTH="1000Mb"
OUT_DIR="repeat_runs_${SCENARIO}_${NUM_RUNS}times"
RUN_LOG="${OUT_DIR}/run_logs"
RESULTS_CSV="${OUT_DIR}/${SCENARIO}_runs_summary.csv"

mkdir -p "${OUT_DIR}" "${RUN_LOG}"

echo "run_id,throughput_Mbps,plr_pct,cov_stability,jain_fairness" > "${RESULTS_CSV}"

echo "=== 开始 ${NUM_RUNS} 次重复实验（场景：${ALGO}+${QUEUE}） ==="
for ((run=1; run<=NUM_RUNS; run++)); do
    echo -e "\n--- 第 ${run}/${NUM_RUNS} 次运行 ---"
    
    SEED=$(( $(date +%s) + run * 1000 ))  # 增加种子随机性
    echo "随机种子: ${SEED}"
    
    JITTER=$(echo "scale=4; $RANDOM / 65535 * 1.0" | bc)  # 扩大抖动范围
    echo "启动时间抖动: ${JITTER}秒"
    
    TCL_SCRIPT="${ALGO}Code.tcl"
    if [ ! -f "${TCL_SCRIPT}" ]; then
        echo "错误：未找到脚本 ${TCL_SCRIPT}，终止实验"
        exit 1
    fi
    
    if [ ! -f "${TCL_SCRIPT}.bak" ]; then
        cp "${TCL_SCRIPT}" "${TCL_SCRIPT}.bak"
        echo "已备份原始脚本为 ${TCL_SCRIPT}.bak"
    fi
    
    # 适配cubicCode.tcl的参数逻辑：通过环境变量传递带宽和队列
    sed -i "s/^set bw.*/set bw \"${BANDWIDTH}\"/" "${TCL_SCRIPT}"
    sed -i "s/DropTail/${QUEUE}/g" "${TCL_SCRIPT}"  # 全局替换队列策略
    # 增加流量随机化（假设TCL脚本支持seed参数）
    sed -i "s/^set seed.*/set seed ${SEED}/" "${TCL_SCRIPT}"
    
    # 运行仿真
    echo "运行仿真：ns ${TCL_SCRIPT}"
    SEED=${SEED} ns "${TCL_SCRIPT}" > "${RUN_LOG}/run_${run}_sim.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "警告：第 ${run} 次仿真失败，日志：${RUN_LOG}/run_${run}_sim.log"
        continue
    fi
    
    TRACE_FILE=$(find . -maxdepth 1 -type f -name "*${ALGO}*.tr" | sort -r | head -n 1)
    if [ -z "${TRACE_FILE}" ]; then
        echo "警告：第 ${run} 次未生成trace文件，跳过分析"
        continue
    fi
    TRACE_DEST="${OUT_DIR}/${SCENARIO}_run${run}.tr"
    mv "${TRACE_FILE}" "${TRACE_DEST}"
    echo "trace文件已保存至：${TRACE_DEST}"
    
    # 调整analyser3.py调用：明确指定trace文件路径
    echo "分析第 ${run} 次结果..."
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
            echo "第 ${run} 次指标：吞吐量=${THROUGHPUT} Mb/s，PLR=${PLR}%，CoV=${COV}，Jain=${JAIN}"
        else
            echo "警告：未找到${ALGO}对应的指标行"
        fi
    else
        echo "警告：第 ${run} 次分析失败，未找到 ${SUMMARY_CSV}"
    fi
done

if [ -f "${TCL_SCRIPT}.bak" ]; then
    mv "${TCL_SCRIPT}.bak" "${TCL_SCRIPT}"
    echo -e "\n已恢复原始TCL脚本"
fi

echo -e "\n=== 计算统计结果 ==="
python3 - <<END
import pandas as pd
import scipy.stats as stats
import numpy as np

NUM_RUNS = int('${NUM_RUNS}')
RESULTS_CSV = '${RESULTS_CSV}'
OUT_DIR = '${OUT_DIR}'

df = pd.read_csv(RESULTS_CSV)
valid_runs = len(df.dropna())
print(f"有效运行次数：{valid_runs}/{NUM_RUNS}")

if valid_runs < 2:
    print("警告：有效运行次数不足，无法计算置信区间")
else:
    metrics = {
        "throughput_Mbps": "吞吐量 (Mb/s)",
        "plr_pct": "丢包率 (%)",
        "cov_stability": "稳定性CoV (越低越好)",
        "jain_fairness": "Jain公平性"
    }
    with open(f"{OUT_DIR}/summary_stats.csv", "w") as f:
        f.write("metric,mean,ci_lower,ci_upper\n")
        print("\n===== 均值与95%置信区间 =====")
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
            print(f"  均值: {mean}")
            print(f"  95%置信区间: [{ci_lower}, {ci_upper}]")
    print(f"\n统计结果已保存至：{OUT_DIR}/summary_stats.csv")

END

echo -e "\n=== 所有实验完成 ==="
echo "结果目录：${OUT_DIR}"
