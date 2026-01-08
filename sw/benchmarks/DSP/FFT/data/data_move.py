# gen_data_c.py
import re

# Input file .s
input_s = "sw/benchmarks/DSP/FFT/data/data.S"
# Output file .c
output_c = "sw/benchmarks/DSP/FFT/data/data.c"

# Dizionario per tenere i dati
sections = {}

with open(input_s, "r") as f:
    lines = f.readlines()

current_label = None
current_data = []

for line in lines:
    line = line.strip()
    if line.startswith(".global"):
        # Nuova etichetta
        if current_label:
            sections[current_label] = current_data
        current_label = line.split()[1]
        current_data = []
    elif line.startswith(".word"):
        # Rimuovo 0x e converti in valore esadecimale
        val = line.split()[1]
        current_data.append(val)
# Aggiungo l'ultima
if current_label:
    sections[current_label] = current_data

# Scrivo file C
with open(output_c, "w") as f:
    f.write("#include <stdint.h>\n\n")
    for label, data in sections.items():
        # Decido tipo: input_size -> uint32_t, gli altri double
        if "size" in label:
            f.write(f"uint32_t {label} = {int(data[0],16)};\n\n")
        else:
            f.write(f"double {label}[] = {{\n")
            for i, val in enumerate(data):
                f.write(f"{val}, ")
                if (i+1) % 4 == 0:
                    f.write("\n")
            f.write("\n};\n\n")
