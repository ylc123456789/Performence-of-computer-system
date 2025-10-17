#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyser3.py (English-commented)

Two modes:

1) Part A (single set in current directory)
   python3 analyser3.py [artifacts_dir]
   - Finds *.tr for reno/cubic/yeah/vegas in the current directory.
   - If a trace is missing, tries to run the corresponding ns script
     (renoCode.tcl, cubicCode.tcl, yeahCode.tcl, vegasCode.tcl).
   - Outputs:
       artifacts/
         ├─ algo_summary.csv
         ├─ algo_compare.png              (goodput & PLR)
         ├─ fairness.png                  (Jain index)
         └─ stability_cov.png             (CoV, lower is better)

2) Part B (compare two directories, e.g., DropTail vs RED)
   python3 analyser3.py --compare runs_dt runs_red [out_dir]
   - Reads four traces for each algorithm from two folders.
   - Outputs side-by-side bar charts:
       out_dir/
         ├─ dt_vs_red_goodput_plr.png
         └─ dt_vs_red_fairness_stability.png
"""

import os
import csv
import math
import glob
import subprocess
from math import ceil
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ------------- Basic configuration -------------

# Payload size per acknowledged packet (bytes). Adjust if your trace uses a different size.
PKT_BYTES = 1000
BITS_PER_PKT = PKT_BYTES * 8

# Teacher-provided ns scripts (used only in Part A if traces are missing)
TCL_SCRIPTS = {
    "reno":  "renoCode.tcl",
    "cubic": "cubicCode.tcl",
    "yeah":  "yeahCode.tcl",
    "vegas": "vegasCode.tcl",
}

# Patterns to discover trace files (case-insensitive variants)
TR_GLOBS = {
    "reno":  ["*reno*.tr",  "*Reno*.tr"],
    "cubic": ["*cubic*.tr", "*Cubic*.tr"],
    "yeah":  ["*yeah*.tr",  "*Yeah*.tr"],
    "vegas": ["*vegas*.tr", "*Vegas*.tr"],
}

# ------------- Helpers: run ns, locate traces, read files -------------

def run_ns_if_needed(algo: str) -> bool:
    """Run the corresponding ns script if a trace is missing (Part A only)."""
    tcl = TCL_SCRIPTS.get(algo)
    if not tcl or not os.path.exists(tcl):
        print(f"[warn] {algo}: cannot find {tcl}; skip running ns.")
        return False
    print(f"[*] Running ns {tcl} to generate a trace ...")
    try:
        subprocess.check_call(["ns", tcl])
        return True
    except Exception as e:
        print(f"[err] ns {tcl} failed: {e}")
        return False

def find_trace_for(algo: str):
    """Try to find a trace by wildcard; if not found, run ns, then search again."""
    # First pass: search by glob in current directory
    for pat in TR_GLOBS.get(algo, []):
        hits = sorted(glob.glob(pat))
        if hits:
            # Pick the most recently modified one
            hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return hits[0]
    # If none found, try to run ns and search again
    if run_ns_if_needed(algo):
        for pat in TR_GLOBS.get(algo, []):
            hits = sorted(glob.glob(pat))
            if hits:
                hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return hits[0]
    return None

def split_file(filename: str):
    """Read a whitespace-tokenized trace file into a list of token lists."""
    lines = []
    with open(filename, "r") as f:
        for ln in f:
            parts = ln.split()
            if parts:
                lines.append(parts)
    return lines

# ------------- Parsing and metrics -------------

def parse_trace(lines):
    """
    Parse a single trace into per-second series and cumulative counters.

    Supported lines:
      - Custom ACK lines:  <time> <flowId> ... ack_ <cumulativeAckPackets>
      - Custom RTT lines:  <time> <flowId> ... rtt_ <rttValue>
      - ns2 drop lines:    first token == 'd', time at index 1,
                           flowId from the first char of the 4th token from the end (parts[-4][0])
    """
    max_t = 0.0
    # Per-second accumulators (later converted to per-second increments)
    sec_ack_cum = defaultdict(lambda: defaultdict(int))
    sec_loss_cnt = defaultdict(lambda: defaultdict(int))
    sec_rtt_vals = defaultdict(lambda: defaultdict(list))

    for parts in lines:
        if "ack_" in parts:
            t = float(parts[0]); flow = parts[1]; s = ceil(t)
            # Store the largest cumulative ACK value seen in this second
            sec_ack_cum[flow][s] = max(sec_ack_cum[flow].get(s, 0), int(parts[-1]))
            max_t = max(max_t, t)
        elif "rtt_" in parts:
            t = float(parts[0]); flow = parts[1]; s = ceil(t)
            # Collect all RTT samples for that second (we will average later)
            try:
                sec_rtt_vals[flow][s].append(float(parts[-1]))
            except:
                pass
            max_t = max(max_t, t)
        elif parts[0] == 'd':
            # Standard ns2 drop line
            try:
                t = float(parts[1]); s = ceil(t)
                flow = parts[-4][0]  # flow id convention used in the provided teacher code
                sec_loss_cnt[flow][s] = sec_loss_cnt[flow].get(s, 0) + 1
                max_t = max(max_t, t)
            except:
                pass

    T = int(math.ceil(max_t)) if max_t > 0 else 1
    flows = sorted(list(set(list(sec_ack_cum.keys()) + list(sec_loss_cnt.keys()))))

    # Convert cumulative ACKs to per-second ACK increments
    ack_series = {}
    cum_ack_pkts = {}
    for f in flows:
        arr = [0]*(T+1); prev = 0
        for s in range(0, T+1):
            val = sec_ack_cum[f].get(s, prev)
            arr[s] = max(0, val - prev)
            prev = val
        ack_series[f] = arr
        cum_ack_pkts[f] = sum(arr)

    # Per-second loss series and cumulative loss
    loss_series = {}
    cum_loss_pkts = {}
    for f in flows:
        arr = [0]*(T+1)
        for s in range(0, T+1):
            arr[s] = sec_loss_cnt[f].get(s, 0)
        loss_series[f] = arr
        cum_loss_pkts[f] = sum(arr)

    # Optional: average RTT per second (not used in main scoring)
    rtt_series = {}
    for f in flows:
        arr = [math.nan]*(T+1)
        for s in range(0, T+1):
            vals = sec_rtt_vals[f].get(s, [])
            if vals:
                arr[s] = float(sum(vals)/len(vals))
        rtt_series[f] = arr

    return {
        "T": T,
        "flows": flows,
        "ack_series": ack_series,
        "loss_series": loss_series,
        "rtt_series": rtt_series,
        "cum_ack_pkts": cum_ack_pkts,
        "cum_loss_pkts": cum_loss_pkts,
    }

def compute_metrics(parsed):
    """
    Compute required KPIs:
      - Per-second throughput (Mb/s)
      - Overall goodput (Mb/s) = total ACKed bits / duration
      - PLR (%) ≈ lost / (acked + lost)
      - Stability (CoV) over throughput time series
      - Fairness (Jain) using the last third of the experiment
    """
    T = parsed["T"]; flows = parsed["flows"]
    ack = parsed["ack_series"]; loss = parsed["loss_series"]

    # Per-second throughput for each flow (Mb/s)
    thrpt = {}
    for f in flows:
        per_sec_bits = np.array(ack[f], dtype=float) * BITS_PER_PKT
        thrpt[f] = per_sec_bits / 1e6

    # Overall goodput for each flow (Mb/s)
    overall = {}
    for f in flows:
        total_bits = float(parsed["cum_ack_pkts"][f]) * BITS_PER_PKT
        overall[f] = total_bits / max(1, T) / 1e6

    # Packet loss rate (%) approximation
    plr = {}
    for f in flows:
        a = float(parsed["cum_ack_pkts"][f])
        l = float(parsed["cum_loss_pkts"][f])
        denom = a + l if (a + l) > 0 else 1.0
        plr[f] = 100.0 * l / denom

    # Stability: coefficient of variation (std/mean) on throughput series
    cov = {}
    for f in flows:
        x = np.array(thrpt[f], dtype=float)
        mu = float(np.mean(x)) if x.size else 0.0
        sd = float(np.std(x)) if x.size else 0.0
        cov[f] = (sd/mu) if mu > 1e-9 else float("inf")

    # Fairness (Jain) on the last third of the time window
    s0 = int(T * (2.0/3.0))
    last_means = []
    for f in flows:
        seg = np.array(thrpt[f][s0:], dtype=float)
        last_means.append(float(np.mean(seg)) if seg.size else 0.0)
    x = np.array(last_means, dtype=float)
    jain = float(((np.sum(x))**2) / (x.size * np.sum(x**2))) if x.size > 0 and np.sum(x**2) > 0 else 1.0

    return {
        "overall_goodput_Mbps": overall,
        "plr_pct": plr,
        "cov": cov,
        "thrpt_series": thrpt,
        "fairness_jain_last_third": jain,
    }

# ------------- Plots and CSV -------------

def plot_algo_compare(results, out_dir):
    """Part A: bar charts for total goodput and average PLR across flavours."""
    algos = [a for a in ["reno","cubic","yeah","vegas"] if a in results]
    goodputs = [sum(results[a]["overall_goodput_Mbps"].values()) for a in algos]
    plrs = [float(sum(results[a]["plr_pct"].values())/len(results[a]["plr_pct"])) for a in algos]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].bar(algos, goodputs)
    axs[0].set_title("Overall Goodput"); axs[0].set_ylabel("Mb/s")
    axs[1].bar(algos, plrs)
    axs[1].set_title("Average PLR");     axs[1].set_ylabel("%")
    fig.suptitle("TCP Flavours Comparison")
    fig.tight_layout()
    p = os.path.join(out_dir, "algo_compare.png")
    fig.savefig(p, dpi=160); plt.close()
    print(f"[ok] wrote {p}")

def plot_fairness(results, out_dir):
    """Part A: bar chart of Jain's fairness index (last third)."""
    algos = [a for a in ["reno","cubic","yeah","vegas"] if a in results]
    vals  = [results[a]["fairness_jain_last_third"] for a in algos]
    plt.figure(figsize=(5,3))
    plt.bar(algos, vals)
    plt.ylim(0,1.05)
    plt.ylabel("Jain's Fairness (last 1/3)")
    plt.title("Fairness by TCP flavour")
    p = os.path.join(out_dir, "fairness.png")
    plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
    print(f"[ok] wrote {p}")

def plot_stability(results, out_dir):
    """Part A: bar chart of CoV (lower is better)."""
    algos = [a for a in ["reno","cubic","yeah","vegas"] if a in results]
    covs  = []
    for a in algos:
        cv = list(results[a]["cov"].values())
        covs.append(float(sum(cv)/len(cv)) if cv else float("nan"))
    plt.figure(figsize=(5,3))
    plt.bar(algos, covs)
    plt.ylabel("CoV of throughput (lower = more stable)")
    plt.title("Stability by TCP flavour")
    p = os.path.join(out_dir, "stability_cov.png")
    plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
    print(f"[ok] wrote {p}")

def write_summary_csv(results, out_dir):
    """Part A: CSV with total goodput, avg PLR, avg CoV, and Jain per flavour."""
    path = os.path.join(out_dir, "algo_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "overall_goodput_Mbps(sum_flows)",
                    "plr_pct(avg_flows)", "stability_CoV(avg_flows)",
                    "Jain_last_third"])
        for algo in ["reno","cubic","yeah","vegas"]:
            if algo not in results: 
                continue
            met = results[algo]
            gp = sum(met["overall_goodput_Mbps"].values())
            cov_vals = list(met["cov"].values())
            cov_avg = float(sum(cov_vals)/len(cov_vals)) if cov_vals else float("nan")
            plr_vals = list(met["plr_pct"].values())
            plr_avg = float(sum(plr_vals)/len(plr_vals)) if plr_vals else float("nan")
            jain = met["fairness_jain_last_third"]
            w.writerow([algo, f"{gp:.3f}", f"{plr_avg:.3f}", f"{cov_avg:.3f}", f"{jain:.4f}"])
    print(f"[ok] wrote {path}")

# ------------- Part A driver -------------

def main(out_dir="artifacts"):
    """Find or generate traces in the current folder, compute metrics, and export charts/CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # Locate or generate traces
    algo_tr = {}
    for algo in ["reno","cubic","yeah","vegas"]:
        tr = find_trace_for(algo)
        if tr:
            print(f"[ok] {algo}: use trace {tr}")
            algo_tr[algo] = tr
        else:
            print(f"[warn] {algo}: no trace found; skip.")

    # Compute metrics per flavour
    results = {}
    for algo, trfile in algo_tr.items():
        parsed = parse_trace(split_file(trfile))
        results[algo] = compute_metrics(parsed)

    # Export
    write_summary_csv(results, out_dir)
    plot_algo_compare(results, out_dir)
    plot_fairness(results, out_dir)
    plot_stability(results, out_dir)

# ------------- Part B (compare two folders) -------------

def load_results_from_dir(dirpath: str):
    """Read four algorithm traces from a folder and compute their metrics."""
    res = {}
    patterns = {
        "reno":  ["*reno*.tr",  "*Reno*.tr"],
        "cubic": ["*cubic*.tr", "*Cubic*.tr"],
        "yeah":  ["*yeah*.tr",  "*Yeah*.tr"],
        "vegas": ["*vegas*.tr", "*Vegas*.tr"],
    }
    for algo, pats in patterns.items():
        cand = []
        for p in pats:
            cand += glob.glob(os.path.join(dirpath, p))
        if not cand:
            print(f"[warn] {dirpath}: missing {algo} trace")
            continue
        cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        trfile = cand[0]
        parsed = parse_trace(split_file(trfile))
        res[algo] = compute_metrics(parsed)
    return res

def compare_two(results_A, results_B, labelA, labelB, out_dir):
    """Produce two side-by-side charts: (1) goodput & PLR, (2) fairness & stability."""
    algos = [a for a in ["reno","cubic","yeah","vegas"]
             if (a in results_A) and (a in results_B)]
    if not algos:
        print("[warn] nothing to compare")
        return

    # Chart 1: goodput & PLR
    gp_A = [sum(results_A[a]["overall_goodput_Mbps"].values()) for a in algos]
    gp_B = [sum(results_B[a]["overall_goodput_Mbps"].values()) for a in algos]
    plr_A= [float(sum(results_A[a]["plr_pct"].values())/len(results_A[a]["plr_pct"])) for a in algos]
    plr_B= [float(sum(results_B[a]["plr_pct"].values())/len(results_B[a]["plr_pct"])) for a in algos]

    x = np.arange(len(algos)); w = 0.35
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].bar(x-w/2, gp_A, width=w, label=labelA)
    axs[0].bar(x+w/2, gp_B, width=w, label=labelB)
    axs[0].set_xticks(x); axs[0].set_xticklabels(algos)
    axs[0].set_ylabel("Mb/s"); axs[0].set_title("Overall Goodput"); axs[0].legend()

    axs[1].bar(x-w/2, plr_A, width=w, label=labelA)
    axs[1].bar(x+w/2, plr_B, width=w, label=labelB)
    axs[1].set_xticks(x); axs[1].set_xticklabels(algos)
    axs[1].set_ylabel("%"); axs[1].set_title("Average PLR"); axs[1].legend()

    fig.tight_layout()
    p1 = os.path.join(out_dir, "dt_vs_red_goodput_plr.png")
    fig.savefig(p1, dpi=160); plt.close(); print(f"[ok] wrote {p1}")

    # Chart 2: fairness & stability (CoV)
    fair_A = [results_A[a]["fairness_jain_last_third"] for a in algos]
    fair_B = [results_B[a]["fairness_jain_last_third"] for a in algos]
    cov_A  = [float(sum(results_A[a]["cov"].values())/len(results_A[a]["cov"])) for a in algos]
    cov_B  = [float(sum(results_B[a]["cov"].values())/len(results_B[a]["cov"])) for a in algos]

    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].bar(x-w/2, fair_A, width=w, label=labelA)
    axs[0].bar(x+w/2, fair_B, width=w, label=labelB)
    axs[0].set_xticks(x); axs[0].set_xticklabels(algos)
    axs[0].set_ylim(0,1.05); axs[0].set_title("Jain's Fairness"); axs[0].legend()

    axs[1].bar(x-w/2, cov_A, width=w, label=labelA)
    axs[1].bar(x+w/2, cov_B, width=w, label=labelB)
    axs[1].set_xticks(x); axs[1].set_xticklabels(algos)
    axs[1].set_title("Stability (CoV, lower = better)"); axs[1].legend()

    fig.tight_layout()
    p2 = os.path.join(out_dir, "dt_vs_red_fairness_stability.png")
    fig.savefig(p2, dpi=160); plt.close(); print(f"[ok] wrote {p2}")

# ------------- Entry point -------------

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    # Part A (default): python3 analyser3.py [artifacts_dir]
    if len(args) == 0 or (len(args) == 1 and args[0] != "--compare"):
        out = args[0] if len(args) == 1 else "artifacts"
        main(out)

    # Part B (compare): python3 analyser3.py --compare runs_dt runs_red [out_dir]
    elif len(args) >= 1 and args[0] == "--compare":
        dt_dir  = args[1] if len(args) > 1 else "runs_dt"
        red_dir = args[2] if len(args) > 2 else "runs_red"
        out_dir = args[3] if len(args) > 3 else "artifacts_dt_vs_red"
        os.makedirs(out_dir, exist_ok=True)
        A = load_results_from_dir(dt_dir)
        B = load_results_from_dir(red_dir)
        compare_two(A, B, "DropTail", "RED", out_dir)

    else:
        print("Usage:\n  python3 analyser3.py [artifacts_dir]\n  python3 analyser3.py --compare runs_dt runs_red [out_dir]")



