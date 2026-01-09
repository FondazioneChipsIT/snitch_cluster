#!/usr/bin/env python3

import numpy as np
import os

def rand_matrix(N, M, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=-10.0, high=10.0, size=(N, M)).astype(np.float64)

def twiddle(N):
    v = np.exp(-2j * np.pi * np.arange(N//2) / N)
    v = v.astype(np.complex128)
    return np.array((np.real(v), np.imag(v))).transpose()

def complex_mul(a, b):
    return np.array((
        a[:,0]*b[:,0] - a[:,1]*b[:,1],
        a[:,0]*b[:,1] + a[:,1]*b[:,0],
    )).transpose()

def fft(x, twiddle):
    if x.shape[0] == 1:
        return x
    else:
        E = fft(x[0::2,:], twiddle)
        O = fft(x[1::2,:], twiddle)
        tw = twiddle(x.shape[0])
        twO = complex_mul(tw, O)
        return np.concatenate((
            E + twO,
            E - twO
        ), axis=0)

def emit(out, name, array):
    print(f".global {name}", file=out)
    print(".align 3", file=out)
    print(f"{name}:", file=out)
    bs = array.tobytes()
    for i in range(0, len(bs), 4):
        s = ""
        for n in range(4):
            s += "%02x" % bs[i+3-n]
        print(f"    .word 0x{s}", file=out)

# =========================
# Main
# =========================

N = 128
x = rand_matrix(N, 2, 1)
y = fft(x, twiddle)

# Cartella dove si trova lo script
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "data.S")

with open(out_path, "w") as out:
    print('.section .l1,"aw",@progbits', file=out)
    emit(out, "input_size", np.array(N, dtype=np.uint32))
    emit(out, "input", x)
    emit(out, "input_twiddle", twiddle(N))
    emit(out, "output", y)
