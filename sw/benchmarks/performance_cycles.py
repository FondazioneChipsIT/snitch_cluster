import re
import os
import glob

# Trova la cartella dello script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Risali fino alla root di snitch_cluster
current_dir = script_dir
while os.path.basename(current_dir) != "snitch_cluster":
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:
        raise FileNotFoundError("Cartella snitch_cluster non trovata nella gerarchia")
    current_dir = parent_dir

# Cartella logs relativa alla root snitch_cluster
log_dir = os.path.join(current_dir, "logs")

# Trova tutti i file trace_hart_*.txt
all_files = sorted(glob.glob(os.path.join(log_dir, "trace_hart_*.txt")))

# Escludi il core 8
files = [f for f in all_files if not f.endswith("trace_hart_00008.txt")]

if not files:
    raise FileNotFoundError(f"Nessun file trace_hart_*.txt valido trovato in {log_dir}")

# Header tabella
print(f"{'Core':<5} {'Cycles':<10} {'IPC':<10} {'FLOPs/cycle':<12}")

cycles_list = []
ipc_list = []
flops_list = []

for x, file_path in enumerate(files):
    try:
        with open(file_path, "r") as f:
            content = f.read()

            # Trova la sezione Performance metrics for section 1
            sections = re.findall(
                r"Performance metrics for section 1.*?\n((?:.*\S.*\n?)*)", content, re.S
            )

            if sections:
                section = sections[0]

                # Estrai cycles
                cycles_match = re.search(r"^\s*cycles\s+(\d+)", section, re.M)
                cycles = int(cycles_match.group(1)) if cycles_match else 0

                # Estrai total_ipc
                ipc_match = re.search(r"^\s*total_ipc\s+([\d\.]+)", section, re.M)
                ipc = float(ipc_match.group(1)) if ipc_match else 0.0

                # Estrai fpss_fpu_occupancy (stima FLOPs per ciclo)
                fpu_occ_match = re.search(r"^\s*fpss_fpu_occupancy\s+([\d\.]+)", section, re.M)
                flops = float(fpu_occ_match.group(1)) if fpu_occ_match else 0.0

                cycles_list.append(cycles)
                ipc_list.append(ipc)
                flops_list.append(flops)

                print(f"{x:<5} {cycles:<10} {ipc:<10.3f} {flops:<12.3f}")
            else:
                print(f"{x:<5} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
    except FileNotFoundError:
        print(f"{x:<5} File non trovato")

# Calcola medie cluster
if cycles_list:
    avg_cycles = sum(cycles_list) / len(cycles_list)
    avg_ipc = sum(ipc_list) / len(ipc_list)
    avg_flops = sum(flops_list) / len(flops_list)

    print("\nCluster average:")
    print(f"{'Cycles':<10}: {avg_cycles:.2f}")
    print(f"{'IPC':<10}: {avg_ipc:.3f}")
    print(f"{'FLOPs/cycle':<10}: {avg_flops:.3f}")
else:
    print("Nessun dato valido trovato")
