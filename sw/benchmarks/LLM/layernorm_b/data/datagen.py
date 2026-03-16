import numpy as np
import os

# ── Parameters ────────────────────────────────────────────────────────────────
BATCH_SIZE  = 2
SEQ_LEN     = 16     # must be multiple of ncores (8)
EMBEDDINGS  = 32    # must be multiple of 8 (UNROLL * num_elems_per_vector)
EPS         = 1e-5  # float in C
CHECK_RESULTS = True

# ── Random input ──────────────────────────────────────────────────────────────
rng   = np.random.default_rng(42)
input_data = rng.random((BATCH_SIZE, SEQ_LEN, EMBEDDINGS)).astype(np.float32)

# ── Golden model: exact replica of layernorm_naive ────────────────────────────
# eps_f used in golden model
eps_f = float(EPS)

O_golden = np.zeros_like(input_data)
for b in range(BATCH_SIZE):
    for s in range(SEQ_LEN):
        row  = input_data[b, s, :]
        mean = np.sum(row) / EMBEDDINGS
        var  = np.sum((row - mean) ** 2) / EMBEDDINGS
        std  = np.sqrt(var + eps_f)
        O_golden[b, s, :] = (row - mean) / std

# ── C array helper ────────────────────────────────────────────────────────────
def matrix_to_c(name, arr, per_row=8):
    flat   = arr.flatten().astype(np.float32)
    length = len(flat)
    c_str  = f"float {name}[{length}] = {{\n"
    for i in range(0, length, per_row):
        chunk = ", ".join(f"{flat[k]:.8f}f" for k in range(i, min(i+per_row, length)))
        c_str += f"    {chunk},\n"
    c_str += "};\n"
    return c_str

# ── Write data.h ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(script_dir, "data.h")

total_elems = BATCH_SIZE * SEQ_LEN * EMBEDDINGS

with open(file_path, "w") as f:
    f.write("#ifndef DATA_H\n")
    f.write("#define DATA_H\n\n")
    f.write("#include \"snrt.h\"\n\n")

    f.write(f"uint32_t BATCH_SIZE =  {BATCH_SIZE};\n")
    f.write(f"uint32_t SEQ_LEN =     {SEQ_LEN};\n")
    f.write(f"uint32_t EMBEDDINGS =  {EMBEDDINGS};\n")
    f.write(f"/* EPS is float in the kernel */\n")
    f.write(f"float EPS =         {EPS}f;\n")
    f.write(f"/* Flag to check results*/\n")
    f.write(f"bool CHECK_RESULTS =  {1 if CHECK_RESULTS else 0};\n")

    f.write("/* TCDM pointers – filled at runtime by DM core */\n")
    f.write("float *ifmap_TCDM;\n")
    f.write("float *ofmap_TCDM;\n\n")

    f.write(f"/* Input tensor [BATCH_SIZE x SEQ_LEN x EMBEDDINGS] = [{total_elems}] – row-major */\n")
    f.write(matrix_to_c("input", input_data) + "\n")

    f.write(f"/* Golden output: layernorm per row, eps = {eps_f}f */\n")
    f.write(matrix_to_c("O_golden", O_golden) + "\n")

    f.write("#endif /* DATA_H */\n")

print(f"data.h successfully generated at: {file_path}")
print(f"Total elements: {total_elems}")
print(f"EPS as float: {eps_f}")