import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# ============================
# INPUT UTENTE
# ============================
kernel_name = input("Kernel name: ").strip()

# ============================
# PATH
# ============================
script_dir = os.path.dirname(os.path.abspath(__file__))
kernel_dir = os.path.join(script_dir, "results", kernel_name)
csv_file = os.path.join(kernel_dir, f"{kernel_name}_results.csv")

if not os.path.isfile(csv_file):
    raise FileNotFoundError(f"CSV del kernel '{kernel_name}' non trovato: {csv_file}")

# ============================
# LEGGI CSV
# ============================
data = defaultdict(list)  # key=num_elements, value=list di dict {cycles, flops_per_cycle}

with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        elements = row["elements"]
        data[elements].append({
            "cycles": float(row["cycles"]),
            "flops_per_cycle": float(row["flops_per_cycle"])
        })

# Ordina i num_elements numericamente
sorted_N = sorted(data.keys(), key=lambda x: float(x))

# ============================
# PLOT: Cycles vs N
# ============================
cycles_means = []
cycles_err_low = []
cycles_err_high = []

for N in sorted_N:
    cycles_list = [p["cycles"] for p in data[N]]
    mean_val = np.mean(cycles_list)
    min_val = np.min(cycles_list)
    max_val = np.max(cycles_list)
    cycles_means.append(mean_val)
    cycles_err_low.append(mean_val - min_val)
    cycles_err_high.append(max_val - mean_val)

plt.figure(figsize=(8,5))
plt.errorbar(
    sorted_N, cycles_means,
    yerr=[cycles_err_low, cycles_err_high],
    fmt='o-', capsize=5, label="Cycles"
)
plt.xlabel("N (numero elementi)")
plt.ylabel("Cycles (core mean)")
plt.title(f"{kernel_name}  Cycles vs N")
plt.grid(True)
plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_cycles_vs_N.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================
# PLOT: FLOPs/cycle vs N
# ============================
flops_means = []
flops_err_low = []
flops_err_high = []

for N in sorted_N:
    flops_list = [p["flops_per_cycle"] for p in data[N]]
    mean_val = np.mean(flops_list)
    min_val = np.min(flops_list)
    max_val = np.max(flops_list)
    flops_means.append(mean_val)
    flops_err_low.append(mean_val - min_val)
    flops_err_high.append(max_val - mean_val)

plt.figure(figsize=(8,5))
plt.errorbar(
    sorted_N, flops_means,
    yerr=[flops_err_low, flops_err_high],
    fmt='o-', capsize=5, label="FLOPs/cycle"
)
plt.xlabel("N (numero elementi)")
plt.ylabel("FLOPs/cycle (core mean)")
plt.title(f"{kernel_name}  FLOPs/cycle vs N")
plt.grid(True)
plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_flops_per_cycle_vs_N.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Plot generati nella cartella: {kernel_dir}")
