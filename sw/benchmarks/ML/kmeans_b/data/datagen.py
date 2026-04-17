import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

n_samples  = 100
n_features = 8
n_clusters = 8
seed       = 42
max_iter   = 20

X, _ = make_blobs(n_samples=n_samples, n_features=n_features,
                  centers=n_clusters, random_state=seed)
X = X.astype(np.float32)  # float32 come in C, evita drift float64 vs float32

rng = np.random.default_rng(seed=seed)
initial_centroids = rng.uniform(
    low=X.min(axis=0), high=X.max(axis=0),
    size=(n_clusters, n_features)
).astype(np.float32)

kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids,
                max_iter=max_iter, n_init=1)
kmeans.fit(X)
n_iter           = int(kmeans.n_iter_)
golden_centroids = kmeans.cluster_centers_.astype(np.float32)
# Nessun sort: il check in C usa nearest-centroid matching

def vec_to_c(name, arr, aligned=True):
    flat  = arr.flatten()
    attr  = ' __attribute__ ((aligned (4096)))' if aligned else ''
    lines = "\n".join(f"\t{v}f," for v in flat)
    return f"float {name}[{len(flat)}]{attr} = {{\n{lines}\n}};\n"

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "../build/data.h")

with open(file_path, "w") as f:
    f.write(f"uint32_t n_samples  = {n_samples};\n\n")
    f.write(f"uint32_t n_features = {n_features};\n\n")
    f.write(f"uint32_t n_clusters = {n_clusters};\n\n")
    f.write(f"uint32_t n_iter     = {max_iter};\n\n")
    f.write(vec_to_c("centroids",        initial_centroids) + "\n")
    f.write(vec_to_c("samples",          X)                 + "\n")
    f.write(vec_to_c("golden_centroids", golden_centroids)  + "\n")

print(f"data.h generato in: {file_path}  |  n_iter = {max_iter}  |  n_iter (sklearn) = {n_iter}")