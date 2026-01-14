#!/usr/bin/env python3

import numpy as np
import os

# ===============================
# CONFIGURAZIONE
# ===============================
N = 512
OUTPUT_FILE = "data.h"

script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, OUTPUT_FILE)

def rand_matrix(N, M, seed):
    np.random.seed(seed)
    return np.random.normal(size=(N, M)).astype(np.float64)

def twiddle(N):
    v = np.exp(-2j * np.pi * np.arange(N//2) / N)
    v = v.astype(np.complex128)
    return np.array((np.real(v), np.imag(v))).transpose()

def complex_mul(a, b):
    return np.array((
        a[:,0]*b[:,0] - a[:,1]*b[:,1],
        a[:,0]*b[:,1] + a[:,1]*b[:,0],
    )).transpose()

def fft(x, twiddle_func):
    if x.shape[0] == 1:
        return x
    E = fft(x[0::2, :], twiddle_func)
    O = fft(x[1::2, :], twiddle_func)
    tw = twiddle_func(x.shape[0])
    twO = complex_mul(tw, O)
    return np.concatenate((E + twO, E - twO), axis=0)

# ===============================
# GENERAZIONE DATI
# ===============================
x = rand_matrix(N, 2, seed=1)          # input reale + immag
y = fft(x, twiddle)                    # FFT custom

tw = twiddle(N)                        # twiddle N/2

# ===============================
# FLATTEN ARRAY (interleaved)
# ===============================
input_data = x.reshape(2*N)
twiddle_data = tw.reshape(N)
output_data = y.reshape(2*N)

# ===============================
# SCRITTURA data.h
# ===============================
with open(out_path, "w") as f:
    f.write("// Auto-generated FFT test data\n\n")
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write(f"static const uint32_t FFT_N = {N};\n\n")

    # INPUT
    f.write(f"double input[{2*N}] = {{\n")
    for i, v in enumerate(input_data):
        f.write(f"    {v:.17e}")
        f.write(",\n" if i < len(input_data)-1 else "\n")
    f.write("};\n\n")

    # TWIDDLE
    f.write(f"double twiddle[{N}] = {{\n")
    for i, v in enumerate(twiddle_data):
        f.write(f"    {v:.17e}")
        f.write(",\n" if i < len(twiddle_data)-1 else "\n")
    f.write("};\n\n")

    # OUTPUT FFT
    f.write(f"double output[{2*N}] = {{\n")
    for i, v in enumerate(output_data):
        f.write(f"    {v:.17e}")
        f.write(",\n" if i < len(output_data)-1 else "\n")
    f.write("};\n")

print(f"data.h generato correttamente in {out_path}")
