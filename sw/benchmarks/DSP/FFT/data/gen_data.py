import numpy as np
import random
import os

# ==========================
# Configuration
# ==========================
N = 128          # number of complex FFT points
RANGE = 10.0    # Range of the complex numbers

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_file = os.path.join(SCRIPT_DIR, "data.h")

# ==========================
# Generate input data
# ==========================
x = np.array(
    [random.uniform(-RANGE, RANGE) + 1j * random.uniform(-RANGE, RANGE)
     for _ in range(N)],
    dtype=np.complex128
)

input_interleaved = np.empty(2 * N, dtype=np.float64)
input_interleaved[0::2] = x.real
input_interleaved[1::2] = x.imag

# ==========================
# Golden FFT (reference)
# ==========================
X = np.fft.fft(x)

golden_interleaved = np.empty(2 * N, dtype=np.float64)
golden_interleaved[0::2] = X.real
golden_interleaved[1::2] = X.imag

# ==========================
# Twiddle factors
# Flat layout:
# twiddle[2*k+0] = cos(2*pi*k/N)
# twiddle[2*k+1] = sin(2*pi*k/N)
# ==========================
twiddle_interleaved = np.empty(2 * N, dtype=np.float64)
for k in range(N):
    angle = -2.0 * np.pi * k / N
    twiddle_interleaved[2*k + 0] = np.cos(angle)
    twiddle_interleaved[2*k + 1] = np.sin(angle)

# ==========================
# Write single data.h
# ==========================
with open(out_file, "w") as f:
    f.write("#ifndef DATA_H\n")
    f.write("#define DATA_H\n\n")

    f.write(f"#define FFT_N {N}\n\n")

    # Input
    f.write("double input[FFT_N * 2] __attribute__((aligned(8))) = {\n")
    for v in input_interleaved:
        f.write(f"    {v:.17e},\n")
    f.write("};\n\n")

    # Twiddle (FLAT, no braces)
    f.write("double twiddle[FFT_N * 2] __attribute__((aligned(8))) = {\n")
    for v in twiddle_interleaved:
        f.write(f"    {v:.17e},\n")
    f.write("};\n\n")

    # Golden output
    f.write("double golden[FFT_N * 2] __attribute__((aligned(8))) = {\n")
    for v in golden_interleaved:
        f.write(f"    {v:.17e},\n")
    f.write("};\n\n")

    f.write("#endif // DATA_H\n")

print(f"Generated {out_file}")
print(f"FFT size: {N} complex points ({2*N} doubles per array)")
