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
def read_csv(path, has_sust):
    data = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cycles = float(r["cycles"])
            flops_alg = float(r["flops_alg_per_cycle"])

            if cycles == 0 or flops_alg == 0:
                continue

            entry = {
                "cycles": cycles,
                "flops_alg": flops_alg
            }

            if has_sust:
                entry["flops_sust"] = float(r["flops_sust_per_cycle"])

            data[r["elements"]].append(entry)
    return data

data_opt = read_csv(csv_file, has_sust=True)
data_naive = read_csv(naive_csv, has_sust=False) if has_naive else {}

sorted_N = sorted(data_opt.keys(), key=lambda x: float(x))

# ============================
# STATISTICHE
# ============================
def stats(data, key):
    out = {}
    for N in data:
        vals = [v[key] for v in data[N]]
        out[N] = {
            "mean": np.mean(vals),
            "min": np.min(vals),
            "max": np.max(vals)
        }
    return out

stats_flops_opt = stats(data_opt, "flops_alg")
stats_cycles_opt = stats(data_opt, "cycles")
stats_flops_sust_opt = stats(data_opt, "flops_sust")

stats_flops_naive = stats(data_naive, "flops_alg") if has_naive else {}
stats_cycles_naive = stats(data_naive, "cycles") if has_naive else {}

# ============================
# PREPARAZIONE ISTOGRAMMI
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

# ============================
# COLORI
# ============================
COLOR_OPT = "tab:orange"
COLOR_NAIVE = "tab:blue"

# ============================
# PLOT 1: FLOPs_alg / cycle + speedup + min/max
# ============================
fig, ax1 = plt.subplots(figsize=(10, 5))

# OPT
opt_means = [stats_flops_opt[N]["mean"] for N in sorted_N]
opt_err = [
    [
        stats_flops_opt[N]["mean"] - stats_flops_opt[N]["min"],
        stats_flops_opt[N]["max"] - stats_flops_opt[N]["mean"]
    ] for N in sorted_N
]

ax1.bar(
    x_positions - bar_width / 2,
    opt_means,
    width=bar_width,
    color=COLOR_OPT,
    label=kernel_name,
    yerr=np.array(opt_err).T,
    capsize=4
)

# NAIVE (solo dove esiste)
if has_naive:
    naive_means = []
    naive_err = []
    for N in sorted_N:
        if N in stats_flops_naive:
            naive_means.append(stats_flops_naive[N]["mean"])
            naive_err.append([
                stats_flops_naive[N]["mean"] - stats_flops_naive[N]["min"],
                stats_flops_naive[N]["max"] - stats_flops_naive[N]["mean"]
            ])
        else:
            naive_means.append(0)
            naive_err.append([0, 0])

    ax1.bar(
        x_positions + bar_width / 2,
        naive_means,
        width=bar_width,
        color=COLOR_NAIVE,
        label=naive_kernel,
        yerr=np.array(naive_err).T,
        capsize=4
    )

ax1.set_ylabel("FLOPs / cycle (alg)")
ax1.set_xlabel("N (elements)")
ax1.set_xticks(x_positions)
ax1.set_xticklabels(labels)
ax1.grid(True, axis="y", ls="--", lw=0.5)

# Speedup
if has_naive:
    ax2 = ax1.twinx()

    xs = []
    ys = []
    for i, N in enumerate(sorted_N):
        if N in stats_flops_naive:
            xs.append(x_positions[i])
            ys.append(
                stats_flops_opt[N]["mean"] /
                stats_flops_naive[N]["mean"]
            )

    ax2.plot(xs, ys, "ro-", label="Speedup opt / naive")
    ax2.set_ylabel("Speedup")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2)
else:
    ax1.legend()

plt.title(f"{kernel_name}: FLOPs_alg / cycle (minmax)")
plt.tight_layout()
plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_hist_flops_alg_speedup_minmax.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ============================
# PLOT 2: CYCLES (log scale)
# ============================
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    x_positions - bar_width / 2,
    [stats_cycles_opt[N]["mean"] for N in sorted_N],
    width=bar_width,
    color=COLOR_OPT,
    label=kernel_name
)

if has_naive:
    ax.bar(
        x_positions + bar_width / 2,
        [stats_cycles_naive[N]["mean"] if N in stats_cycles_naive else 0 for N in sorted_N],
        width=bar_width,
        color=COLOR_NAIVE,
        label=naive_kernel
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

# ============================
# PLOT 3: FLOPs_sust / cycle
# ============================
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    x_positions,
    [stats_flops_sust_opt[N]["mean"] for N in sorted_N],
    width=bar_width,
    color=COLOR_OPT,
    label=f"{kernel_name} sustained"
)

ax.set_ylabel("FLOPs / cycle (sustained)")
ax.set_xlabel("N (elements)")
ax.set_xticks(x_positions)
ax.set_xticklabels(labels)
ax.grid(True, axis="y", ls="--", lw=0.5)
ax.legend()

plt.title(f"{kernel_name}: FLOPs_sust / cycle")
plt.tight_layout()
plt.savefig(
    os.path.join(kernel_dir, f"{kernel_name}_hist_flops_sust.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print(f"Tutti i plot generati in: {kernel_dir}")
