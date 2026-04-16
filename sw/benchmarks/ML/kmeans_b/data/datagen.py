import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ── Parameters ────────────────────────────────────────────────────────────────
n_samples  = 100  
n_features = 8
n_clusters = 8
seed       = 1
max_iter   = 10

# ── Generate samples ──────────────────────────────────────────────────────────
X, _ = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=seed
)
X = X.astype(np.float32)  # Convert to float32 for C compatibility

# ── Initial centroids ─────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=seed)
initial_centroids = rng.uniform(
    low=X.min(axis=0), high=X.max(axis=0),
    size=(n_clusters, n_features)
)

# ── Golden model ──────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids,
                max_iter=max_iter, n_init=1)
kmeans.fit(X)
golden_centroids = kmeans.cluster_centers_   # float64
n_iter           = int(kmeans.n_iter_)

# ── C array helper (matches original format exactly) ─────────────────────────
def vec_to_c(name, arr, aligned=True):
    flat   = arr.flatten()
    length = len(flat)
    attr   = ' __attribute__ ((aligned (4096)))' if aligned else ''
    c_str  = f"float {name}[{length}]{attr} = {{\n"
    for v in flat:
        c_str += f"\t{repr(float(v))},\n"
    c_str += "};\n"
    return c_str

# ── Write data.h ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "../build/data.h")

with open(file_path, "w") as f:    
    f.write(f"uint32_t n_samples = {n_samples};\n\n")
    f.write(f"uint32_t n_features = {n_features};\n\n")
    f.write(f"uint32_t n_clusters = {n_clusters};\n\n")
    f.write(f"uint32_t n_iter = {n_iter};\n\n")

    f.write(vec_to_c("centroids", initial_centroids) + "\n")
    f.write(vec_to_c("samples", X) + "\n")
    f.write(vec_to_c("golden_centroids", golden_centroids) + "\n")

print(f"data.h successfully generated at: {file_path}")
print(f"n_iter = {n_iter}")