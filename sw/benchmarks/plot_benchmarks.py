import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# ============================
# INPUT UTENTE
# ============================
kernel_name = input("Kernel name: ").strip()
naive_kernel = f"{kernel_name}_naive"

# ============================
# PATH
# ============================
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")

kernel_dir = os.path.join(results_dir, kernel_name)
csv_file = os.path.join(kernel_dir, f"{kernel_name}_results.csv")

if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"CSV del kernel '{kernel_name}' non trovato: {csv_file}")

naive_dir = os.path.join(results_dir, naive_kernel)
naive_csv = os.path.join(naive_dir, f"{naive_kernel}_results.csv")
has_naive = os.path.isfile(naive_csv)

# ============================
# LEGGI CSV (kernel principale)
# ============================
data = defaultdict(list)

with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cycles = float(row["cycles"])
        flops_alg = float(row["flops_alg_per_cycle"])
        flops_sust = float(row["flops_sust_per_cycle"])

        if cycles == 0.0 or flops_alg == 0.0 or flops_sust == 0.0:
            continue

        N = row["elements"]
        data[N].append({
            "cycles": cycles,
            "flops_alg": flops_alg,
            "flops_sust": flops_sust,
        })

sorted_N = sorted(data.keys(), key=lambda x: float(x))

# ============================
# LEGGI CSV (naive)
# ============================
data_naive = defaultdict(list)

if has_naive:
    with open(naive_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cycles = float(row["cycles"])
            flops_alg = float(row["flops_alg_per_cycle"])

            if cycles == 0.0 or flops_alg == 0.0:
                continue

            N = row["elements"]
            data_naive[N].append({
                "cycles": cycles,
                "flops_alg": flops_alg,
            })

# ============================
# UTILITY
# ============================
def mean_dict(data_dict, key):
    return {
        N: np.mean([p[key] for p in data_dict[N]])
        for N in data_dict
    }

mean_flops_alg = mean_dict(data, "flops_alg")
mean_cycles = mean_dict(data, "cycles")

mean_flops_naive = (
    mean_dict(data_naive, "flops_alg") if has_naive else {}
)
mean_cycles_naive = (
    mean_dict(data_naive, "cycles") if has_naive else {}
)

# ============================
# PLOT 1: Cycles vs N (log, mean + minmax)
# ============================
plt.figure(figsize=(8,5))

# --- kernel ottimizzato ---
cycles_mean = []
cycles_min = []
cycles_max = []

for N in sorted_N:
    vals = [p["cycles"] for p in data[N]]
    cycles_mean.append(np.mean(vals))
    cycles_min.append(np.min(vals))
    cycles_max.append(np.max(vals))

plt.plot(sorted_N, cycles_mean, "o-", label=f"{kernel_name} mean")
plt.fill_between(
    sorted_N, cycles_min, cycles_max,
    alpha=0.2
)

# --- kernel naive ---
if has_naive:
    common_N_cycles = [N for N in sorted_N if N in data_naive]

    naive_mean = []
    naive_min = []
    naive_max = []

    for N in common_N_cycles:
        vals = [p["cycles"] for p in data_naive[N]]
        naive_mean.append(np.mean(vals))
        naive_min.append(np.min(vals))
        naive_max.append(np.max(vals))

    plt.plot(
        common_N_cycles,
        naive_mean,
        "s--",
        color="tab:green",
        label=f"{naive_kernel} mean"
    )
    plt.fill_between(
        common_N_cycles,
        naive_min,
        naive_max,
        alpha=0.2,
        color="tab:green"
    )

plt.xlabel("N")
plt.ylabel("Cycles (core mean)")
plt.title(f"{kernel_name}  Cycles vs N (log scale)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()

plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_cycles_vs_N_log.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================
# PLOT 2: FLOPs_alg + naive + speedup
# ============================
fig, ax1 = plt.subplots(figsize=(8,5))

# FLOPs kernel ottimizzato
ax1.plot(
    sorted_N,
    [mean_flops_alg[N] for N in sorted_N],
    "o-",
    label=kernel_name
)

# FLOPs naive
common_N = [N for N in sorted_N if N in mean_flops_naive]

if has_naive and common_N:
    ax1.plot(
        common_N,
        [mean_flops_naive[N] for N in common_N],
        "s--",
        color="tab:green",
        label=naive_kernel
    )

ax1.set_xlabel("N")
ax1.set_ylabel("FLOPs_alg / cycle")
ax1.grid(True, which="both", ls="--", lw=0.5)

# Speedup
if has_naive and common_N:
    ax2 = ax1.twinx()
    speedup = [
        mean_flops_alg[N] / mean_flops_naive[N]
        for N in common_N
    ]

    ax2.plot(
        common_N,
        speedup,
        "r-o",
        label="Speedup vs naive"
    )
    ax2.set_ylabel("Speedup vs naive")

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2)
else:
    ax1.legend()

plt.title(f"{kernel_name} vs {naive_kernel}")
plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_flops_alg_and_speedup_dual.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Tutti i plot generati in: {kernel_dir}")
