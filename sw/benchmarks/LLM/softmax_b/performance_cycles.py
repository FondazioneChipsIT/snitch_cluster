import re
import os
import glob

# Trova la cartella dello script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Risali fino alla root di snitch_cluster (supponiamo che ci sia 'snitch_cluster' nella path)
current_dir = script_dir
while os.path.basename(current_dir) != "snitch_cluster":
    parent_dir = os.path.dirname(current_dir)
    if parent_dir == current_dir:  # Arrivati alla root del filesystem senza trovare snitch_cluster
        raise FileNotFoundError("Cartella snitch_cluster non trovata nella gerarchia")
    current_dir = parent_dir

# Cartella logs relativa alla root snitch_cluster
log_dir = os.path.join(current_dir, "logs")

# Trova tutti i file trace_hart_*.txt e ordina
all_files = sorted(glob.glob(os.path.join(log_dir, "trace_hart_*.txt")))

# Escludi il core 8
files = [f for f in all_files if not f.endswith("trace_hart_00008.txt")]

if not files:
    raise FileNotFoundError(f"Nessun file trace_hart_*.txt valido trovato in {log_dir}")

cycles_per_core = []

for x, file_path in enumerate(files):
    try:
        with open(file_path, "r") as f:
            content = f.read()

            # Trova tutte le sezioni Performance metrics for section 1
            sections = re.findall(
                r"Performance metrics for section 1.*?\n((?:.*\S.*\n?)*)", content, re.S
            )

            if sections:
                section = sections[0]

                # Cerca la riga dei cycles
                cycles_match = re.search(r"^\s*cycles\s+(\d+)", section, re.M)
                if cycles_match:
                    cycles = int(cycles_match.group(1))
                    cycles_per_core.append(cycles)
                    print(f"{os.path.basename(file_path)}: {cycles} cycles")
                else:
                    print(f"{os.path.basename(file_path)}: 'cycles' non trovato nella sezione")
            else:
                print(f"{os.path.basename(file_path)}: sezione performance non trovata")
    except FileNotFoundError:
        print(f"{file_path} non trovato")

# Calcola e stampa la media
if cycles_per_core:
    avg_cycles = sum(cycles_per_core) / len(cycles_per_core)
    print(f"\nMedia cycles: {avg_cycles:.2f}")
else:
    print("Nessun dato di cycles trovato")
