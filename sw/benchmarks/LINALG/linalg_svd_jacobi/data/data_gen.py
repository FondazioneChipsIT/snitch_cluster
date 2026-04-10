import numpy as np
import os

# ── Parameters ────────────────────────────────────────────────────────────────
M = 64   
N = 32

# ── Generate random SPD matrix ────────────────────────────────────────────────
# A = R^T * R with R random upper-triangular + large diagonal → guaranteed SPD
rng = np.random.default_rng(42)
R   = np.triu(rng.random((M, M)))
np.fill_diagonal(R, R.diagonal() + M)
mat = (R.T @ R).astype(np.float32)

# ── Golden model: Cholesky  mat = L * L^T ────────────────────────────────────
L_gold = np.linalg.cholesky(mat.astype(np.float64)).astype(np.float32)

# ── C array helper ────────────────────────────────────────────────────────────
def matrix_to_c_array(name, m):
    flat  = m.flatten()
    c_str = f"float {name}[M * M] = {{\n"
    for i in range(M):
        row = ", ".join(f"{flat[i * M + j]:.6f}f" for j in range(M))
        c_str += f"    {row},\n"
    c_str += "};\n"
    return c_str

# ── Write data.h ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "data.h")

with open(file_path, "w") as f:
    f.write("#ifndef DATA_H\n")
    f.write("#define DATA_H\n\n")

    f.write("#ifndef M\n")
    f.write(f"#define M {M}\n")
    f.write("#endif\n\n")

    f.write("#ifndef N\n")
    f.write(f"#define N {N}\n")
    f.write("#endif\n\n")

    f.write("/* TCDM pointers – filled at runtime by DM core */\n")
    f.write("float *mat;\n")
    f.write("float *mat_V;\n")
    f.write("float *vec_S;\n\n")
    f.write("float *local_max[8];\n\n")

    f.write("/* Input: random symmetric positive-definite matrix (row-major) */\n")
    f.write(matrix_to_c_array("mat_data", mat) + "\n")

    f.write("/* Golden output: lower-triangular L s.t. mat = L * L^T (row-major) */\n")
    f.write(matrix_to_c_array("golden_L", L_gold) + "\n")

    f.write("#endif /* DATA_H */\n")

print(f"data.h successfully generated at: {file_path}")