import numpy as np
import os

# -- Parameters ----------------------------------------------------------------
M = 16          # rows
N = 8         # cols  (M >= N assumed  tall/square matrix)
K = min(M, N)   # number of singular values

# -- Generate a random M×N matrix ----------------------------------------------
# Built as  A = U_rand * diag(sigma) * V_rand^T  so the condition number is
# controlled and the golden SVD is numerically clean.
rng = np.random.default_rng(42)

# Random orthogonal factors via QR decomposition
U_rand, _ = np.linalg.qr(rng.random((M, M)).astype(np.float64))
V_rand, _ = np.linalg.qr(rng.random((N, N)).astype(np.float64))

# Singular values: K values spread between 1 and M (well-conditioned)
sigma = np.sort(rng.uniform(1.0, float(M), K))[::-1]   # descending

# Build A = U * S * V^T  (only first K cols/rows of U/V are used)
A = (U_rand[:, :K] * sigma) @ V_rand[:, :K].T
mat = A.astype(np.float32)

# -- Golden model: full SVD  A = U * diag(S) * V^T ----------------------------
# numpy returns economy SVD by default with full_matrices=False:
#   U_gold : M×K,  s_gold : K,  Vt_gold : K×N
U_gold_f64, s_gold_f64, Vt_gold_f64 = np.linalg.svd(mat.astype(np.float64),
                                                      full_matrices=False)

# Store V  (N×K, column = right singular vector)  Jacobi typically gives V
V_gold = Vt_gold_f64.T.astype(np.float32)          # N×K
U_gold = U_gold_f64.astype(np.float32)              # M×K
s_gold = s_gold_f64.astype(np.float32)              # K

# -- C array helpers -----------------------------------------------------------
def mat_to_c(name: str, rows: int, cols: int, m: np.ndarray) -> str:
    """Emit  float name[rows * cols] = { ... };  in row-major order."""
    flat  = m.flatten()
    lines = [f"float {name}[{rows} * {cols}] = {{"]
    for i in range(rows):
        row = ", ".join(f"{flat[i * cols + j]:.6f}f" for j in range(cols))
        lines.append(f"    {row},")
    lines.append("};\n")
    return "\n".join(lines)

def vec_to_c(name: str, length: int, v: np.ndarray) -> str:
    """Emit  float name[length] = { ... };"""
    elems = ", ".join(f"{v[i]:.6f}f" for i in range(length))
    return f"float {name}[{length}] = {{\n    {elems}\n}};\n"

# -- Write data.h --------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "data.h")

with open(file_path, "w") as f:
    f.write("#ifndef DATA_H\n")
    f.write("#define DATA_H\n\n")

    # Dimension macros
    f.write("#ifndef M\n")
    f.write(f"#define M {M}\n")
    f.write("#endif\n\n")

    f.write("#ifndef N\n")
    f.write(f"#define N {N}\n")
    f.write("#endif\n\n")

    f.write("#ifndef K\n")
    f.write(f"#define K {K}   /* min(M,N)  number of singular values */\n")
    f.write("#endif\n\n")

    # Runtime TCDM pointers (filled by DM core)
    f.write("/* TCDM pointers  filled at runtime by DM core */\n")
    f.write("float *mat;      /* M×N input matrix            */\n")
    f.write("float *mat_U;    /* M×K left  singular vectors  */\n")
    f.write("float *mat_V;    /* N×K right singular vectors  */\n")
    f.write("float *vec_S;    /* K   singular values         */\n\n")
    f.write("float local_max[8];\n\n")

    # Input matrix
    f.write("/* Input: random M×N matrix (row-major) */\n")
    f.write(mat_to_c("mat_data", M, N, mat) + "\n")

    # Golden U  (M×K)
    f.write("/* Golden output: left singular vectors U, shape M×K (row-major) */\n")
    f.write(mat_to_c("golden_U", M, K, U_gold) + "\n")

    # Golden V  (N×K)
    f.write("/* Golden output: right singular vectors V, shape N×K (row-major) */\n")
    f.write(mat_to_c("golden_V", N, K, V_gold) + "\n")

    # Golden singular values
    f.write("/* Golden output: singular values (descending order), length K */\n")
    f.write(vec_to_c("golden_S", K, s_gold) + "\n")

    f.write("#endif /* DATA_H */\n")

print(f"data.h successfully generated at: {file_path}")

# -- Quick sanity check --------------------------------------------------------
recon  = (U_gold.astype(np.float64)
          * s_gold.astype(np.float64)
          ) @ V_gold.T.astype(np.float64)
err    = np.max(np.abs(recon - mat.astype(np.float64)))
print(f"Max reconstruction error |A - U·S·V?|8 = {err:.3e}  (should be < 1e-4)")