import numpy as np
import os

# ── Parameters ────────────────────────────────────────────────────────────────
LEN        = 256
FILTER_LEN = 32    # number of taps actually used (must be <= LEN)

# ── Random input signal and filter coefficients ───────────────────────────────
rng = np.random.default_rng(42)
x = rng.random(LEN).astype(np.float32)
h = rng.random(LEN).astype(np.float32)

# ── Golden model: exact replica of fir_naive ─────────────────────────────────
# y[n] = sum_{k=0}^{taps-1} x[n-k] * h[k]
# taps = min(n+1, FILTER_LEN)
y = np.zeros(LEN, dtype=np.float32)
for n in range(LEN):
    taps = min(n + 1, FILTER_LEN)
    for k in range(taps):
        y[n] += x[n - k] * h[k]

# ── C array helper ────────────────────────────────────────────────────────────
def vec_to_c(name, arr, per_row=8):
    flat   = arr.flatten().astype(np.float32)
    length = len(flat)
    c_str  = f"float {name}[{length}] = {{\n"
    for i in range(0, length, per_row):
        chunk = ", ".join(f"{flat[k]:.6f}f" for k in range(i, min(i+per_row, length)))
        c_str += f"    {chunk},\n"
    c_str += "};\n"
    return c_str

# ── Write data.h ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "data.h")

with open(file_path, "w") as f:
    f.write("#ifndef DATA_H\n")
    f.write("#define DATA_H\n\n")

    f.write("/* FIR filter – causal convolution, left-border handled by reducing tap count */\n")
    f.write(f"#define LEN        {LEN}\n")
    f.write(f"#define FILTER_LEN {FILTER_LEN}\n\n")

    f.write("/* TCDM pointers – filled at runtime by DM core */\n")
    f.write("float *x;\n")
    f.write("float *y;\n")
    f.write("float *h;\n\n")

    f.write("/* Input signal (random) – copied to TCDM at runtime */\n")
    f.write(vec_to_c("x_data", x) + "\n")

    f.write("/* Filter coefficients (random) – copied to TCDM at runtime */\n")
    f.write(vec_to_c("h_data", h) + "\n")

    f.write("/* Golden output y[n] = sum_{k=0}^{taps-1} x[n-k]*h[k] */\n")
    f.write(vec_to_c("golden_y", y) + "\n")

    f.write("#endif /* DATA_H */\n")

print(f"data.h successfully generated at: {file_path}")