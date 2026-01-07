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

        # Escludi entry 0.0
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

mean_flops_naive = {
    N: np.mean(vals) for N, vals in data_naive.items()
} if has_naive else None

# ============================
# PLOT 1: Cycles vs N (log scale)
# ============================
cycles_means = []
min_cycles = []
max_cycles = []

for N in sorted_N:
    vals = [p["cycles"] for p in data[N]]
    cycles_means.append(np.mean(vals))
    min_cycles.append(np.min(vals))
    max_cycles.append(np.max(vals))

plt.figure(figsize=(8,5))
plt.plot(sorted_N, cycles_means, "o-", label="Mean")
plt.fill_between(sorted_N, min_cycles, max_cycles, alpha=0.2, label="Min-Max")
plt.xlabel("N (numero elementi)")
plt.ylabel("Cycles (core mean)")
plt.title(f"{kernel_name}  Cycles vs N (log scale)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.savefig(os.path.join(kernel_dir, f"{kernel_name}_cycles_vs_N_log.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================
# PLOT 2: FLOPs_alg/cycle vs N (lineare, senza barre)
# ============================
flops_alg_means = []
for N in sorted_N:
    vals = [p["flops_alg"] for p in data[N]]
    flops_alg_means.append(np.mean(vals))

plt.figure(figsize=(8,5))
plt.plot(sorted_N, flops_alg_means, "o-", label=kernel_name)
if has_naive:
    plt.plot(sorted_N, [mean_flops_naive[N] for N in sorted_N], "o--", label=naive_kernel)
plt.xlabel("N (numero elementi)")
plt.ylabel("FLOPs_alg / cycle")
plt.title(f"{kernel_name}  Algorithmic FLOPs/cycle vs N")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(kernel_dir, f"{kernel_name}_flops_alg_per_cycle_vs_N.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# ============================
# PLOT 3: FLOPs_sust/cycle vs N (lineare, senza barre)
# ============================
flops_sust_means = []
for N in sorted_N:
    vals = [p["flops_sust"] for p in data[N]]
    flops_sust_means.append(np.mean(vals))

plt.figure(figsize=(8,5))
plt.plot(sorted_N, flops_sust_means, "o-")
plt.xlabel("N (numero elementi)")
plt.ylabel("FLOPs_sust / cycle")
plt.title(f"{kernel_name}  Sustained FLOPs/cycle vs N")
plt.grid(True)
plt.savefig(os.path.join(kernel_dir, f"{kernel_name}_flops_sust_per_cycle_vs_N.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# ============================
# PLOT 4: SPEEDUP vs NAIVE
# ============================
if has_naive:
    speedup = []
    Ns_speedup = []
    for N in sorted_N:
        if N in mean_flops_naive and mean_flops_naive[N] > 0:
            speedup.append(mean_flops_alg[N] / mean_flops_naive[N])
            Ns_speedup.append(N)

    plt.figure(figsize=(8,5))
    plt.plot(Ns_speedup, speedup, "o-")
    plt.xlabel("N (numero elementi)")
    plt.ylabel("Speedup (FLOPs_alg/cycle)")
    plt.title(f"{kernel_name} vs {naive_kernel}  Speedup")
    plt.grid(True)
    plt.savefig(os.path.join(kernel_dir, f"{kernel_name}_speedup_vs_naive.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Speedup plot generato (vs {naive_kernel})")
else:
    print(f"Kernel naive '{naive_kernel}' non trovato  speedup non generato")

print(f"Tutti i plot generati nella cartella: {kernel_dir}")
