#!/bin/python3
import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, TextIO, Tuple

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# repo root: kmeans_b (one level up from data/)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.helper_functions import (
    DKind,
    _to_dkind,
    dkind_name,
    _acc_dtype_main,
    acc_add,
    finalize_out_dtype,
    check_pulp_warnings,
    select_dtypes,
    check_vec_flag,
    check_cast,             # not used here but kept for parity
    matrix_init_like as matrix_init,
    str2bool,
    error_metric,
    _cast_apply
)

# Always relative to this script's directory
FOLDER_ADDR = Path(__file__).resolve().parent

# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class KMeansConfig:
    # flags
    mac: bool
    vec: bool
    cast: bool          # kept for parity; not used in KMeans math directly
    hw_mixed: bool
    mixed_vec: bool
    mantissa_bits: int

    # types
    dt_x: DKind         # points & centroids live in this dtype
    dt_out: DKind       # distances/output live in this dtype
    cast_to: Optional[DKind]  # None if no casting destination

# ----------------------------
# Distance kernel (point vs. all centroids)
# ----------------------------
def squared_distances_point_centroids(
    point: torch.Tensor, centroids: torch.Tensor, cfg: KMeansConfig
) -> torch.Tensor:
    """
    Return a K-length tensor of squared Euclidean distances
    between 'point' and each row in 'centroids'.

    Accumulator follows _acc_dtype_main(cfg).
    Vector path uses lane tree-reduction when cfg.mixed_vec is True.
    """
    K, F = centroids.shape

    # output buffer for distances (in OUT dtype, but accumulate in acc dtype)
    out_dtype = torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()
    dists = torch.zeros((K,), dtype=out_dtype, device=point.device)

    acc_dtype = _acc_dtype_main(cfg)

    if not cfg.vec:
        unroll = 2
        processed = (F // unroll) * unroll
        a = torch.zeros(
            unroll,
            dtype=(torch.float32 if cfg.dt_x is DKind.FP8_CUSTOM else cfg.dt_x.to_torch()),
            device=point.device,
        )
        b = torch.zeros(
            unroll,
            dtype=(torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()),
            device=point.device,
        )
        diff = torch.zeros(unroll, dtype=acc_dtype, device=point.device)
        # scalar path
        for i in range(K):

            acc = torch.zeros(1, dtype=acc_dtype, device=point.device)
            # main loop
            for j in range(0, processed, unroll):
                a = point[j : j + unroll]
                b = centroids[i, j : j + unroll]

                a = finalize_out_dtype(a, cfg)
                b = finalize_out_dtype(b, cfg)

                diff = a - b

                diff = finalize_out_dtype(diff, cfg)

                for k in range(unroll):
                    acc = acc_add(acc, diff[k], diff[k], cfg, leftover=False, in_main=True)
                    if not cfg.hw_mixed:
                        acc = finalize_out_dtype(acc, cfg)

            # leftovers
            if processed < F:
                for r in range(processed, F):
                    a0 = point[r]
                    b0 = centroids[i, r]
                    a0 = finalize_out_dtype(a0, cfg)
                    b0 = finalize_out_dtype(b0, cfg)
                    d = a0 - b0
                    d = finalize_out_dtype(d, cfg)
                    acc = acc_add(acc, d, d, cfg, leftover=False, in_main=True)
                    if not cfg.hw_mixed:
                        acc = finalize_out_dtype(acc, cfg)

            dists[i] = finalize_out_dtype(acc, cfg)
        return dists

    # -------------- vector path --------------
    vec_step = 4 if cfg.dt_x is DKind.FP8_CUSTOM else 2
    processed = (F // vec_step) * vec_step
    remainder = F - processed

    for i in range(K):
        temp = torch.zeros(vec_step, dtype=acc_dtype, device=point.device)
        sum_val = torch.zeros(1, dtype=acc_dtype, device=point.device)

        for j in range(0, processed, vec_step):
            a = point[j : j + vec_step]
            b = centroids[i, j : j + vec_step]

            a = finalize_out_dtype(a, cfg)
            b = finalize_out_dtype(b, cfg)

            d = a - b

            d = finalize_out_dtype(d, cfg)

            if cfg.mixed_vec:
                for k in range(vec_step):
                    t = torch.zeros(1, dtype=acc_dtype, device=point.device)
                    t = acc_add(t, d[k], d[k], cfg, leftover=False, in_main=True)
                    t = finalize_out_dtype(t, cfg)
                    temp[k] = t
                if vec_step == 2:
                    t = temp[0] + temp[1]
                else:
                    t01 = temp[0] + temp[1]
                    t23 = temp[2] + temp[3]
                    t = t01 + t23
                sum_val = sum_val + t
                sum_val = finalize_out_dtype(sum_val, cfg)
            else:
                for k in range(vec_step):
                    tk = temp[k].unsqueeze(0)
                    tk = acc_add(tk, d[k], d[k], cfg, leftover=False, in_main=True)
                    tk = finalize_out_dtype(tk, cfg)
                    temp[k] = tk.squeeze(0)

        # leftovers
        if remainder:
            for r in range(remainder):
                idx = processed + r
                a = point[idx]
                b = centroids[i, idx]
                a = finalize_out_dtype(a, cfg)
                b = finalize_out_dtype(b, cfg)
                diff = a - b
                diff = finalize_out_dtype(diff, cfg)
                if cfg.mixed_vec:
                    sum_val = acc_add(sum_val, diff, diff, cfg, leftover=True, in_main=False)
                else:
                    t0 = temp[0].unsqueeze(0)
                    t0 = acc_add(t0, diff, diff, cfg, leftover=True, in_main=False)
                    t0 = finalize_out_dtype(t0, cfg)
                    temp[0] = t0.squeeze(0)

        # finalize per-centroid
        if cfg.mixed_vec:
            acc = sum_val
            acc = finalize_out_dtype(acc, cfg)
        else:
            acc = torch.zeros(1, dtype=acc_dtype, device=point.device)
            for k in range(vec_step):
                acc = acc + temp[k]
                acc = finalize_out_dtype(acc, cfg)
        dists[i] = acc

    return dists

# ----------------------------
# KMeans (multicore emulation with cfg numerics)
# ----------------------------
def mean_func(inp: torch.Tensor, cfg: KMeansConfig) -> torch.Tensor:
    """
    Mean in declared OUT dtype; no MAC promotion here.
    """
    out_t = torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()

    mean = torch.zeros(inp.shape[1], dtype=out_t, device=inp.device)
    for i in range(inp.shape[1]):
        temp = torch.zeros(1, dtype=out_t, device=inp.device)
        for j in range(inp.shape[0]):
            val = inp[j, i].to(out_t)
            temp = temp + val
            temp = finalize_out_dtype(temp, cfg)
        temp = temp / float(inp.shape[0])
        temp = finalize_out_dtype(temp, cfg)
        mean[i] = temp
    return mean

def fit_multicore(
    X: torch.Tensor, K: int = 8, cfg: KMeansConfig = None, max_iterations: int = 100,
    threshold: float = 1e-4, num_cores: int = 4, exploration_flag: bool = False
) -> Tuple[torch.Tensor, np.ndarray, int]:
    """
    C-style multicore emulation.
    """

    N_OBJECTS, N_COORDS = X.shape

    out_t = torch.float32 if cfg.dt_out is DKind.FP8_CUSTOM else cfg.dt_out.to_torch()

    centroids = torch.zeros((K, N_COORDS), dtype=out_t, device=X.device)
    centroids.copy_(X[:K].to(out_t))

    membership = -1 * torch.ones(N_OBJECTS, dtype=torch.int32, device=X.device)
    delta = torch.ones(1, dtype=out_t, device=X.device)

    iteration = 0
    blocks = torch.tensor_split(torch.arange(N_OBJECTS, device=X.device), num_cores)
    cluster_blocks = torch.tensor_split(torch.arange(K, device=X.device), num_cores)

    def simulate_one_iteration(X, centroids, membership, num_cores):
        local_sum = torch.zeros((num_cores, K, N_COORDS), dtype=out_t, device=X.device)
        local_count = torch.zeros((num_cores, K), dtype=torch.int32, device=X.device)
        local_delta = torch.zeros(num_cores, dtype=out_t, device=X.device)

        for cid, block in enumerate(blocks):
            for i in block.tolist():
                point = X[i]
                closest = torch.argmin(squared_distances_point_centroids(point, centroids, cfg))

                if membership[i] != closest:
                    local_delta[cid] = local_delta[cid] + 1.0
                    local_delta[cid] = finalize_out_dtype(local_delta[cid], cfg)

                membership[i] = closest

                local_sum[cid, closest] = local_sum[cid, closest] + finalize_out_dtype(point, cfg)
                local_sum[cid, closest] = finalize_out_dtype(local_sum[cid, closest], cfg)
                local_count[cid, closest] += 1

        total_delta = torch.zeros(1, dtype=out_t, device=X.device)
        for cid in range(num_cores):
            total_delta = total_delta + local_delta[cid].item()
            total_delta = finalize_out_dtype(total_delta, cfg)

        newClusters = torch.zeros((K, N_COORDS), dtype=out_t, device=X.device)
        newClusterSize = torch.zeros(K, dtype=torch.int32, device=X.device)

        for cluster_block in cluster_blocks:
            for k in cluster_block.tolist():
                for cid in range(num_cores):
                    newClusterSize[k] += local_count[cid, k]
                    local_count[cid, k] = 0
                    for c in range(N_COORDS):
                        newClusters[k, c] = newClusters[k, c] + finalize_out_dtype(local_sum[cid, k, c], cfg)
                        newClusters[k, c] = finalize_out_dtype(newClusters[k, c], cfg)

                    newClusters[k] = finalize_out_dtype(newClusters[k], cfg)
                    local_sum[cid, k].zero_()
                    local_count[cid, k] = 0

        new_centroids = torch.zeros_like(centroids, dtype=out_t, device=X.device)
        for k in range(K):
            if newClusterSize[k] > 0:
                num = newClusters[k]
                den = torch.tensor(float(newClusterSize[k].item()), dtype=out_t, device=X.device)
                num = finalize_out_dtype(num, cfg)
                den = finalize_out_dtype(den, cfg)
                mk = num / den
                mk = finalize_out_dtype(mk, cfg)
                new_centroids[k] = mk
            else:
                mk = centroids[k]
                mk = finalize_out_dtype(mk, cfg)
                new_centroids[k] = mk

        denom = torch.tensor(float(N_OBJECTS), dtype=out_t, device=X.device)
        delta_frac = total_delta / finalize_out_dtype(denom, cfg)
        delta_frac = finalize_out_dtype(delta_frac, cfg)

        return new_centroids, delta_frac, membership

    while delta > threshold and iteration < max_iterations:
        centroids = finalize_out_dtype(centroids, cfg)

        prev = centroids.clone()
        centroids, delta, membership = simulate_one_iteration(
            X, centroids, membership, num_cores
        )
        print(f"Iteration {iteration + 1}/ {max_iterations}, delta={float(delta):.6f}")

        diff = centroids - prev
        iteration += 1
        if not diff.any():
            if exploration_flag == "false":
                print(f"Converged by centroid equality at iteration {iteration}")
            break

    y_pred = membership.cpu().numpy()
    return centroids, y_pred, iteration

# ----------------------------
# Data loading / scaling
# ----------------------------
def load_data(
    input_size: int,
    num_features: int,
    scaling_method: str = "normalize",
    scale_value: float = 0.0025,
    target_range: tuple = (0.0, 1.0),
) -> torch.Tensor:
    path = FOLDER_ADDR / "dataset" / "training.csv"
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    data = numeric.iloc[:input_size, :num_features].values.astype(np.float32)

    if scaling_method == "multiplicative":
        data *= float(scale_value)
    elif scaling_method == "normalize":
        a, b = target_range
        mn = np.min(data, axis=0, keepdims=True)
        mx = np.max(data, axis=0, keepdims=True)
        rng = mx - mn
        rng[rng == 0] = 1.0
        data = a + (data - mn) * (b - a) / rng
    elif scaling_method == "standardize":
        mu = np.mean(data, axis=0, keepdims=True)
        sd = np.std(data, axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        data = (data - mu) / sd

    return torch.tensor(data, dtype=torch.float32)


# ----------------------------
# I/O helpers for C headers
# ----------------------------
def write_matrix_flat(
    matrix_to_write: torch.Tensor,
    name: str,
    file_pointer,
) -> None:
    """Write a 2D tensor as a flat C float array (matches reference data.h format)."""
    sz0, sz1 = matrix_to_write.shape
    file_pointer.write(
        f"float {name}[{sz0} * {sz1}] __attribute__((aligned(4096))) = {{\n"
    )
    for i in range(sz0):
        row = ", ".join(f"{matrix_to_write[i, j].item():.9f}f" for j in range(sz1))
        file_pointer.write(f"    {row},\n")
    file_pointer.write("};\n\n")


def save_data_into_hfile(
    X: torch.Tensor, K: int, centers: torch.Tensor, iterations: int, threshold: float
) -> None:
    # --- data_def.h ---
    with open("data_def.h", "w") as f:
        f.write(
            f"#define N_CLUSTERS     {K}\n"
            f"#define N_OBJECTS      {X.shape[0]}\n"
            f"#define N_COORDS       {X.shape[1]}\n"
            f"#define MAX_ITERATIONS {iterations}\n\n"
        )
        # C variables (NOT macros)  avoids collision with function parameter names
        f.write(
            f"int32_t  n_samples  = {X.shape[0]};\n"
            f"uint32_t n_features = {X.shape[1]};\n"
            f"uint32_t n_clusters = {K};\n"
            f"uint32_t n_iter     = {iterations};\n"
            f"float    threshold  = {threshold}f;\n\n"
        )
        # initial centroids = first K rows of X
        write_matrix_flat(X[:K], "centroids", f)
        # full samples array
        write_matrix_flat(X, "samples", f)

    # --- out_ref.h ---
    with open("out_ref.h", "w") as g:
        g.write(
            '#ifndef __CHECKSUM_H__\n'
            '#define __CHECKSUM_H__\n'
            '#include "config.h"\n'
            '#include "pmsis.h"\n\n'
        )
        write_matrix_flat(centers, "golden_centroids", g)
        g.write("#endif\n")


# ----------------------------
# CLI / main
# ----------------------------
def get_initial_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--num_clusters", type=int, default=9)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--mac_flag", default=False)
    parser.add_argument("--vec_flag", default=False)
    parser.add_argument("--hwmixed_flag", default=False)
    parser.add_argument("--mantissa_bits", default=2)
    parser.add_argument("--exploration_flag", default=False)
    parser.add_argument("--scaling_method", default="normalize",
                        choices=["multiplicative", "normalize", "standardize"])
    parser.add_argument("--scale_value", type=float, default=0.0025)
    parser.add_argument("--target_range", default="-.25,0.25")
    parser.add_argument("--float_type", default="FP32")  # X and OUT
    args = parser.parse_args()

    bits = [s.strip() for s in args.float_type.split(",")]
    input_size = int(args.input_size)
    num_features = int(args.features)
    K = int(args.num_clusters)
    cores = int(args.cores)
    mac_flag = str2bool(args.mac_flag)
    vec_flag = str2bool(args.vec_flag)
    hwmixed_flag = str2bool(args.hwmixed_flag)
    mantissa_bits = int(args.mantissa_bits)
    exploration_flag = str2bool(args.exploration_flag)
    scaling_method = str(args.scaling_method)
    scale_value = float(args.scale_value)
    a, b = [float(v) for v in args.target_range.split(",")]
    target_range = (a, b)

    return (input_size, num_features, K, bits, mac_flag, vec_flag, hwmixed_flag,
            mantissa_bits, exploration_flag, scaling_method, scale_value, target_range, cores)

def main():
    (input_size, num_features, K, bits, mac_flag, vec_flag, hwmixed_flag,
    mantissa_bits, exploration_flag, scaling_method, scale_value, target_range, cores) = get_initial_config()

    if not exploration_flag:
        check_pulp_warnings(bits, mac_flag, hwmixed_flag, vec_flag)

    # dtypes & derived flags  expecting 2: [X, OUT]
    dts = select_dtypes(bits, 2)
    dt_x, dt_out = dts[0], dts[1]
    mixed_vec_flag = check_vec_flag(dts, vec_flag)

    max_iterations = 70
    threshold = 0.0001

    if not exploration_flag:
        if DKind.FP8_CUSTOM in dts:
            print(f"Running with {dkind_name(dt_x)}, {dkind_name(dt_out)}")
            print(f"and mantissa = {mantissa_bits} bits")
        else:
            print(f"Running with {dkind_name(dt_x)}, {dkind_name(dt_out)}")
        print(f"Mixed Vectorization Flag: {mixed_vec_flag}")

    # load data
    X_fp32 = load_data(input_size, num_features, scaling_method, scale_value, target_range)

    # FP32 reference (for error only)
    cfg_ref = KMeansConfig(
        mac=False, vec=False, cast=False, hw_mixed=False, mixed_vec=False,
        mantissa_bits=0,
        dt_x=DKind.FP32, dt_out=DKind.FP32, cast_to=None
    )
    if not exploration_flag:
        print("K-Means (FP32) ...")
    centers_ref, _, it_ref = fit_multicore(
        X_fp32, K, cfg_ref, max_iterations=max_iterations, threshold=threshold,
        num_cores=cores, exploration_flag=exploration_flag
    )

    # typed run
    X_typed = matrix_init(X_fp32.contiguous(), dt_x, mantissa_bits)

    cfg = KMeansConfig(
        mac=mac_flag,
        vec=vec_flag,
        cast=False,
        hw_mixed=hwmixed_flag,
        mixed_vec=mixed_vec_flag,
        mantissa_bits=mantissa_bits,
        dt_x=dt_x,
        dt_out=dt_out,
        cast_to=None,
    )

    if not exploration_flag:
        print(f"K-Means ({dkind_name(dt_x)} -> {dkind_name(dt_out)}) ...")

    centers_t, membership, it_used = fit_multicore(
        X_typed, K, cfg, max_iterations=it_ref, threshold=1e-4,
        num_cores=cores, exploration_flag=exploration_flag
    )

    # error metrics vs FP32 centers
    out_dir = os.path.join(
        os.getcwd(), "exploration", scaling_method,
        f"{'scale_'+str(scale_value) if scaling_method=='multiplicative' else (str(target_range[0])+'_'+str(target_range[1]) if scaling_method=='normalize' else 'minax_0_mean_0_std_1')}"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"error_metric__{input_size}_{dkind_name(dt_x)}_{dkind_name(dt_out)}_{mantissa_bits}.txt"
    ) if exploration_flag else None

    error_metric(centers_ref, centers_t, out_file)

    if not exploration_flag:
        save_data_into_hfile(X_typed, K, centers_t, it_used, 1e-4)
        print("############################## Done! ###################################")

if __name__ == "__main__":
    main()