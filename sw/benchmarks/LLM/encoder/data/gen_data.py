import numpy as np
import os

# ─────────────────────────────────────────────
#  GPT-2 124M encoder parameters
# ─────────────────────────────────────────────
B      = 2          # batch size
T      = 32         # sequence length
C      = 64        # embedding / channel dimension
V      = 512      # vocab size (50257 padded to nearest multiple of 64)
MAX_T  = T          # max positional embedding length (== T for this test)

# ─────────────────────────────────────────────
#  Set to False to embed the FULL wte[V*C].
#  Warning: full wte → ~500 MB header file.
#  When True only the V_USED rows actually
#  referenced by inp are stored; the C array
#  is indexed by a compact remapping table.
# ─────────────────────────────────────────────
COMPACT_WTE = True

np.random.seed(42)

# ── inputs ──────────────────────────────────
inp = np.random.randint(0, V, size=(B, T), dtype=np.int32)   # (B, T) token ids
wte = np.random.randn(V, C).astype(np.float32)                # (V,  C) token embeddings
wpe = np.random.randn(MAX_T, C).astype(np.float32)            # (T,  C) positional embeddings

# ── golden reference ────────────────────────
out = np.zeros((B, T, C), dtype=np.float32)
for b in range(B):
    for t in range(T):
        ix = inp[b, t]
        out[b, t, :] = wte[ix] + wpe[t]

# ── compact wte (optional) ──────────────────
unique_ids   = np.unique(inp)              # sorted unique token ids in inp
id_to_compact = {int(v): i for i, v in enumerate(unique_ids)}
V_USED       = len(unique_ids)
wte_compact  = wte[unique_ids]             # (V_USED, C)
inp_compact  = np.vectorize(id_to_compact.__getitem__)(inp).astype(np.int32)

# ─────────────────────────────────────────────
#  Helper: numpy array → C 1-D row-major literal
# ─────────────────────────────────────────────
def array_to_c_float(name: str, arr: np.ndarray, rows: int, cols: int) -> str:
    flat  = arr.flatten()
    lines = [f"float {name}[{rows}*{cols}] = {{"]
    for i in range(rows):
        row = ", ".join(f"{flat[i*cols + j]:.6f}f" for j in range(cols))
        lines.append(f"    {row},")
    lines.append("};\n")
    return "\n".join(lines)

def array_to_c_int(name: str, arr: np.ndarray, rows: int, cols: int) -> str:
    flat  = arr.flatten()
    lines = [f"int {name}[{rows}*{cols}] = {{"]
    for i in range(rows):
        row = ", ".join(str(flat[i*cols + j]) for j in range(cols))
        lines.append(f"    {row},")
    lines.append("};\n")
    return "\n".join(lines)

# ─────────────────────────────────────────────
#  Write data.h
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "data.h")

with open(file_path, "w") as f:

    f.write("#ifndef DATA_H\n")
    f.write("#define DATA_H\n\n")

    # ── dimension macros ──────────────────────
    f.write("/* GPT-2 124M encoder dimensions */\n")
    f.write(f"#define B      {B}\n")
    f.write(f"#define T      {T}\n")
    f.write(f"#define C      {C}\n")
    f.write(f"#define V      {V}\n")
    f.write(f"#define MAX_T  {MAX_T}\n")

    if COMPACT_WTE:
        f.write(f"#define V_USED {V_USED}  /* unique token ids in this batch */\n")

    f.write("\n")

    # ── TCDM pointer declarations ─────────────
    f.write("/* TCDM working buffers (to be allocated at runtime) */\n")
    f.write("float *wte_TCDM;\n")
    f.write("float *wpe_TCDM;\n")
    f.write("float *out_TCDM;\n\n")

    # ── token id input ────────────────────────
    f.write("/* inp[B*T] – token ids, row-major */\n")
    if COMPACT_WTE:
        # store the remapped ids (0-based into wte_compact)
        f.write("/* NOTE: ids are remapped to [0, V_USED) to index wte_compact */\n")
        f.write(array_to_c_int("inp", inp_compact, B, T))
    else:
        f.write(array_to_c_int("inp", inp, B, T))

    f.write("\n")

    # ── token embeddings ──────────────────────
    if COMPACT_WTE:
        f.write("/* wte_compact[V_USED*C] – only the token embeddings used in this batch */\n")
        f.write(array_to_c_float("wte_compact", wte_compact, V_USED, C))
    else:
        f.write("/* wte[V*C] – full token embedding table, row-major */\n")
        f.write(array_to_c_float("wte", wte, V, C))

    f.write("\n")

    # ── positional embeddings ─────────────────
    f.write("/* wpe[MAX_T*C] – positional embeddings, row-major */\n")
    f.write(array_to_c_float("wpe", wpe, MAX_T, C))
    f.write("\n")

    # ── golden output ─────────────────────────
    f.write("/* golden[B*T*C] – expected output of encoder_forward, row-major */\n")
    f.write(array_to_c_float("golden", out.reshape(B * T, C), B * T, C))

    f.write("#endif /* DATA_H */\n")

# ─────────────────────────────────────────────
#  Size report
# ─────────────────────────────────────────────
size_mb = os.path.getsize(file_path) / (1024 ** 2)

print(f"data.h written to : {file_path}")
print(f"File size          : {size_mb:.1f} MB")
print()
print("Array sizes:")
print(f"  inp              : {B}×{T}  = {B*T} int32")
if COMPACT_WTE:
    print(f"  wte_compact      : {V_USED}×{C} = {V_USED*C} float32  "
          f"  (full wte would be {V}×{C} = {V*C} floats)")
else:
    print(f"  wte              : {V}×{C} = {V*C} float32")
print(f"  wpe              : {MAX_T}×{C} = {MAX_T*C} float32")
print(f"  golden           : {B*T}×{C} = {B*T*C} float32")
print()
print(f"COMPACT_WTE = {COMPACT_WTE}  "
      f"({'compact ids remapped to [0,V_USED)' if COMPACT_WTE else 'full vocabulary stored'})")