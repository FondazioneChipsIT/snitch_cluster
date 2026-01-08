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

        elements = row["elements"]
        data[elements].append({
            "cycles": cycles,
            "flops_alg": flops_alg,
            "flops_sust": flops_sust,
        })

sorted_N = sorted(data.keys(), key=lambda x: float(x))

# ============================
# LEGGI CSV (naive, se esiste)
# ============================
data_naive = defaultdict(list)

if has_naive:
    with open(naive_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flops_alg = float(row["flops_alg_per_cycle"])
            if flops_alg == 0.0:
                continue
            elements = row["elements"]
            data_naive[elements].append(flops_alg)

# ============================
# UTILITY: media per N
# ============================
def mean_dict(data_dict, key):
    out = {}
    for N in data_dict:
        out[N] = np.mean([p[key] for p in data_dict[N]])
    return out

mean_flops_alg = mean_dict(data, "flops_alg")
mean_flops_sust = mean_dict(data, "flops_sust")
mean_cycles = mean_dict(data, "cycles")

mean_flops_naive = {N: np.mean(vals) for N, vals in data_naive.items()} if has_naive else None

# ============================
# PLOT COMBINATO FLOPs_alg + speedup
# ============================
fig, ax1 = plt.subplots(figsize=(8,5))

# FLOPs kernel ottimizzato
ax1.plot(sorted_N, [mean_flops_alg[N] for N in sorted_N], "o-", label=kernel_name)
ax1.set_xlabel("N")
ax1.set_ylabel("FLOPs_alg / cycle ", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid(True, which="both", ls="--", lw=0.5)

# FLOPs naive
if has_naive:
    ax1.plot(sorted_N, [mean_flops_naive[N] for N in sorted_N], "s--", color="tab:green", label=naive_kernel)

# Speedup asse y secondario
if has_naive:
    ax2 = ax1.twinx()
    speedup = []
    for N in sorted_N:
        if N in mean_flops_naive and mean_flops_naive[N] > 0:
            speedup.append(mean_flops_alg[N] / mean_flops_naive[N])
        else:
            speedup.append(np.nan)

    ax2.plot(sorted_N, speedup, "r-o", label="Speedup vs naive")  # linea rossa continua con cerchi
    ax2.set_ylabel("Speedup vs naive", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

# Titolo e legenda combinata
lines_1, labels_1 = ax1.get_legend_handles_labels()
if has_naive:
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2)
else:
    ax1.legend()

plt.title(f"{kernel_name} vs {naive_kernel}")
plt.savefig(os.path.join(kernel_dir, f"{kernel_name}_flops_alg_and_speedup_dual.png"),
            dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot FLOPs_alg/cycle + speedup con due scale generato nella cartella: {kernel_dir}")
