import numpy as np
import os
from sklearn.datasets import make_blobs

n_samples  = 100
n_features = 8
n_clusters = 8
seed       = 42
max_iter   = 5

X, _ = make_blobs(n_samples=n_samples, n_features=n_features,
                  centers=n_clusters, random_state=seed)
X = X.astype(np.float32)

rng = np.random.default_rng(seed=seed)
initial_centroids = rng.uniform(
    low=X.min(axis=0), high=X.max(axis=0),
    size=(n_clusters, n_features)
).astype(np.float32)

# Lloyd's algorithm — replicates C behavior exactly:
# float32 accumulation + division, no early stopping
centroids = initial_centroids.copy()
for it in range(max_iter):
    # Assignment: distanza euclidea in float32
    dists = np.zeros((n_samples, n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        diff = (X - centroids[k]).astype(np.float32)
        dists[:, k] = np.sum(diff * diff, axis=1).astype(np.float32)
    labels = np.argmin(dists, axis=1)

    # Update: sum float32 + divisione float32, come il C
    new_centroids = np.zeros_like(centroids)
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            s = X[mask].sum(axis=0).astype(np.float32)   # sum in float32
            new_centroids[k] = (s / np.float32(mask.sum())).astype(np.float32)
        else:
            new_centroids[k] = centroids[k]

    # NO early stopping — il C non ce l'ha
    centroids = new_centroids

golden_centroids = centroids

# ── Helpers ──────────────────────────────────────────────────────────────────
def float32_to_hex(v):
    """Converte float32 in hex literal C — round-trip perfetto garantito."""
    import struct
    bits = struct.unpack('<I', struct.pack('<f', float(v)))[0]
    return f"0x{bits:08X}"

def mat_to_c(name, arr):
    """Matrice float32 con hex literals per round-trip esatto."""
    flat = arr.flatten().astype(np.float32)
    rows, cols = arr.shape
    lines = [f"float {name}[{rows} * {cols}] __attribute__((aligned(4096))) = {{"]
    for i in range(rows):
        row = ", ".join(
            # __builtin_bit_cast oppure union trick in C — hex float è il più portabile
            f"{flat[i*cols+j]:.9f}f"
            for j in range(cols)
        )
        lines.append(f"    {row},")
    lines.append("};\n")
    return "\n".join(lines)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "../build/data.h")

with open(file_path, "w") as f:
    f.write(f"uint32_t n_samples  = {n_samples};\n\n")
    f.write(f"uint32_t n_features = {n_features};\n\n")
    f.write(f"uint32_t n_clusters = {n_clusters};\n\n")
    f.write(f"uint32_t n_iter     = {max_iter};\n\n")
    f.write(mat_to_c("centroids",        initial_centroids) + "\n")
    f.write(mat_to_c("samples",          X)                 + "\n")
    f.write(mat_to_c("golden_centroids", golden_centroids)  + "\n")

# ── Verifica round-trip ───────────────────────────────────────────────────────
print("Verifica round-trip float32:")
for name, arr in [("centroids", initial_centroids),
                  ("samples",   X),
                  ("golden",    golden_centroids)]:
    flat = arr.flatten().astype(np.float32)
    for v in flat:
        s = f"{v:.9f}"
        recovered = np.float32(float(s))
        if recovered != v:
            print(f"  WARN {name}: {v} → '{s}' → {recovered}")
            break
    else:
        print(f"  OK  {name}: tutti i valori sopravvivono al round-trip")

print(f"\ndata.h generato in: {file_path}")