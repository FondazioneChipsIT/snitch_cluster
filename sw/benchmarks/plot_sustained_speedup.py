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
    raise FileNotFoundError(f"CSV del kernel '{kernel_name}' non trovato")

naive_dir = os.path.join(results_dir, naive_kernel)
naive_csv = os.path.join(naive_dir, f"{naive_kernel}_results.csv")
has_naive = os.path.isfile(naive_csv)

# ============================
# LETTURA CSV
# ============================
def read_csv_sust(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            flops_sust = float(r.get("flops_sust_per_cycle", 0.0))
            cycles = float(r["cycles"])
            flops_alg = float(r.get("flops_alg_per_cycle", 0.0))

            if flops_sust <= 0:
                continue  # filtro solo FLOPs_sust

            entry = {"flops_sust": flops_sust, "cycles": cycles, "flops_alg": flops_alg}
            data[r["elements"]].append(entry)
    return data

data_opt = read_csv_sust(csv_file)
data_naive = read_csv_sust(naive_csv) if has_naive else {}

sorted_N = sorted(data_opt.keys(), key=lambda x: float(x))

# ============================
# STATISTICHE min/mean/max
# ============================
def stats(data, key):
    out = {}
    for N in data:
        vals = [v[key] for v in data[N] if key in v]
        if vals:
            out[N] = {"mean": np.mean(vals), "min": np.min(vals), "max": np.max(vals)}
    return out

stats_sust_opt = stats(data_opt, "flops_sust")
stats_sust_naive = stats(data_naive, "flops_sust") if has_naive else {}

stats_cycles_opt = stats(data_opt, "cycles")
stats_cycles_naive = stats(data_naive, "cycles") if has_naive else {}

# ============================
# POSIZIONI ISTOGRAMMI
# ============================
bar_width = 0.35
group_spacing = 0.6
x_positions = []
labels = []

x = 0.0
for N in sorted_N:
    x_positions.append(x)
    labels.append(N)
    x += 1.0 + group_spacing
x_positions = np.array(x_positions)

COLOR_NAIVE = "tab:blue"
COLOR_OPT = "tab:orange"

# ============================
# PLOT 1: FLOPs_sust / cycle + speedup
# ============================
fig, ax1 = plt.subplots(figsize=(10,5))

# NAIVE a sinistra
if has_naive:
    naive_x, naive_means, naive_lo, naive_hi = [], [], [], []
    for i, N in enumerate(sorted_N):
        if N in stats_sust_naive:
            m = stats_sust_naive[N]["mean"]
            mn = stats_sust_naive[N]["min"]
            mx = stats_sust_naive[N]["max"]
            naive_x.append(x_positions[i] - bar_width/2)
            naive_means.append(m)
            naive_lo.append(max(m-mn,0.0))
            naive_hi.append(max(mx-m,0.0))

    if naive_x:
        ax1.bar(
            naive_x,
            naive_means,
            width=bar_width,
            color=COLOR_NAIVE,
            label=naive_kernel,
            yerr=[naive_lo, naive_hi],
            capsize=4
        )

# OPT a destra
opt_x, opt_means, opt_lo, opt_hi = [], [], [], []
for i, N in enumerate(sorted_N):
    if N in stats_sust_opt:
        m = stats_sust_opt[N]["mean"]
        mn = stats_sust_opt[N]["min"]
        mx = stats_sust_opt[N]["max"]
        opt_x.append(x_positions[i] + bar_width/2)
        opt_means.append(m)
        opt_lo.append(max(m-mn,0.0))
        opt_hi.append(max(mx-m,0.0))

if opt_x:
    ax1.bar(
        opt_x,
        opt_means,
        width=bar_width,
        color=COLOR_OPT,
        label=kernel_name,
        yerr=[opt_lo,opt_hi],
        capsize=4
    )

# ASSI E GRIGLIA
ax1.set_xlabel("N (elements)")
ax1.set_ylabel("FLOPs / cycle (sustained)")
ax1.set_xticks(x_positions)
ax1.set_xticklabels(labels)
ax1.grid(True, axis="y", ls="--", lw=0.5)

# SPEEDUP opt / naive
if has_naive:
    ax2 = ax1.twinx()
    xs, ys = [], []
    for i, N in enumerate(sorted_N):
        if N in stats_sust_naive and N in stats_sust_opt:
            xs.append(x_positions[i])
            ys.append(stats_sust_opt[N]["mean"]/stats_sust_naive[N]["mean"])
    if xs:
        ax2.plot(xs, ys, "ro-", label="Speedup opt / naive")
        ax2.set_ylabel("Speedup")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2)
    else:
        ax1.legend()
else:
    ax1.legend()

plt.title(f"{kernel_name}: FLOPs_sust / cycle")
plt.tight_layout()
plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_hist_flops_sust_speedup.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================
# PLOT 2: CYCLES (log scale)
# ============================
fig, ax = plt.subplots(figsize=(10,5))

# NAIVE a sinistra
if has_naive:
    ax.bar(
        x_positions - bar_width/2,
        [stats_cycles_naive[N]["mean"] if N in stats_cycles_naive else 0 for N in sorted_N],
        width=bar_width,
        color=COLOR_NAIVE,
        label=naive_kernel
    )

# OPT a destra
ax.bar(
    x_positions + bar_width/2,
    [stats_cycles_opt[N]["mean"] for N in sorted_N],
    width=bar_width,
    color=COLOR_OPT,
    label=kernel_name
)

ax.set_yscale("log")
ax.set_ylabel("Cycles")
ax.set_xlabel("N (elements)")
ax.set_xticks(x_positions)
ax.set_xticklabels(labels)
ax.grid(True, which="both", axis="y", ls="--", lw=0.5)
ax.legend()

plt.title(f"{kernel_name}: Cycles (log scale)")
plt.tight_layout()
plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_hist_cycles_log.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Plot FLOPs_sust + speedup e Cycles generati in: {kernel_dir}")
