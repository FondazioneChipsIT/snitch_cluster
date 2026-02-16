import os
import glob
import csv
import re

# ============================
# INPUT UTENTE
# ============================
kernel_name = input("Kernel name: ").strip()
num_elements = input("Number of elements: ").strip()

# ============================
# PERCORSI
# ============================
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Cartella dedicata al kernel
kernel_dir = os.path.join(results_dir, kernel_name)
os.makedirs(kernel_dir, exist_ok=True)

# CSV unico per tutto il benchmark
csv_file = os.path.join(kernel_dir, f"{kernel_name}_results.csv")

# Risali fino alla root di snitch_cluster
current_dir = script_dir
while os.path.basename(current_dir) != "snitch_cluster":
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        raise FileNotFoundError("snitch_cluster non trovato nella gerarchia")
    current_dir = parent

log_dir = os.path.join(current_dir, "logs")

# ============================
# FILE INPUT
# ============================
all_files = sorted(glob.glob(os.path.join(log_dir, "trace_hart_*.txt")))
files = [f for f in all_files if not f.endswith("trace_hart_00008.txt")]

if not files:
    raise FileNotFoundError("Nessun file trace_hart_*.txt trovato")

# ============================
# FILE OUTPUT PER NUM_ELEMENTS
# ============================
txt_file = os.path.join(kernel_dir, f"{num_elements}_performance.txt")
txt = open(txt_file, "w")

def log_print(msg):
    print(msg)
    txt.write(msg + "\n")

# ============================
# HEADER TXT
# ============================
log_print(f"Kernel: {kernel_name}")
log_print(f"Elements: {num_elements}")
log_print("")
log_print(
    f"{'Core':<5} {'Cycles':<10} {'IPC':<8} "
    f"{'FLOPs_alg/cyc':<15} {'FLOPs_sust/cyc':<15}"
)

# ============================
# CSV (append se esiste)
# ============================
csv_exists = os.path.isfile(csv_file)
csv_f = open(csv_file, "a", newline="")
csv_writer = csv.writer(csv_f)

if not csv_exists:
    csv_writer.writerow([
        "kernel", "elements", "core",
        "cycles", "ipc",
        "flops_alg_per_cycle",
        "flops_sust_per_cycle"
    ])

# ============================
# LISTE PER MEDIE
# ============================
cycles_list = []
ipc_list = []
flops_alg_list = []
flops_sust_list = []

# ============================
# Lista istruzioni FLOP RISC-V
# ============================
fpu_flop_2 = ["fmadd", "fmsub", "fnmadd", "fnmsub"]
fpu_flop_1 = [
    "fadd", "fsub", "fmul", "fdiv",
    "fsgnj", "fsgnjn", "fsgnjx", "fmin", "fmax"
]

all_fpu_instr = fpu_flop_1 + fpu_flop_2
pattern = re.compile(r"\b(" + "|".join(all_fpu_instr) + r")\b", re.IGNORECASE)

# ============================
# PARSING FILES
# ============================
for core_id, file_path in enumerate(files):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # ------------------------
    # Cicli e IPC
    # ------------------------
    content = "".join(lines)
    match = re.search(
        r"Performance metrics for section 1.*?\n((?:.*\n?)*)",
        content,
        re.S
    )
    if not match:
        continue

    section = match.group(1)
    cycles = int(re.search(r"^\s*cycles\s+(\d+)", section, re.M).group(1))
    ipc = float(re.search(r"^\s*total_ipc\s+([\d\.]+)", section, re.M).group(1))

    # ------------------------
    # CONTO FLOP
    # ------------------------
    flop_alg = 0
    flop_sust = 0

    for line in lines:
        m = pattern.search(line)
        if not m:
            continue

        instr = m.group(1).lower()

        # Sustained FLOPs: tutte le FP ops
        flop_sust += 2 if instr in fpu_flop_2 else 1

        # Algorithmic FLOPs: solo FMAs
        if instr in fpu_flop_2:
            flop_alg += 2

    flops_alg_per_cycle = flop_alg / cycles if cycles > 0 else 0.0
    flops_sust_per_cycle = flop_sust / cycles if cycles > 0 else 0.0

    # ------------------------
    # Salva dati
    # ------------------------
    cycles_list.append(cycles)
    ipc_list.append(ipc)
    flops_alg_list.append(flops_alg_per_cycle)
    flops_sust_list.append(flops_sust_per_cycle)

    log_print(
        f"{core_id:<5} {cycles:<10} {ipc:<8.3f} "
        f"{flops_alg_per_cycle:<15.4f} {flops_sust_per_cycle:<15.4f}"
    )

    csv_writer.writerow([
        kernel_name, num_elements, core_id,
        cycles, ipc,
        flops_alg_per_cycle,
        flops_sust_per_cycle
    ])

# ============================
# MEDIE CLUSTER
# ============================
log_print("\nCluster average:")
log_print(f"Cycles            : {sum(cycles_list)/len(cycles_list):.2f}")
log_print(f"IPC               : {sum(ipc_list)/len(ipc_list):.3f}")
log_print(f"FLOPs_alg/cycle   : {sum(flops_alg_list)/len(flops_alg_list):.4f}")
log_print(f"FLOPs_sust/cycle  : {sum(flops_sust_list)/len(flops_sust_list):.4f}")

txt.close()
csv_f.close()

print(f"\nOutput generato nella cartella {kernel_dir}")
print(f"CSV unico per il benchmark: {csv_file}")
