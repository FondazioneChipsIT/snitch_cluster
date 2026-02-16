import numpy as np
import os

# Matrix dimension (up to 24)
elems = 32

# Generate random matrices
mat_a = np.random.rand(elems, elems).astype(np.float32)
mat_b = np.random.rand(elems, elems).astype(np.float32)

# Golden model: product matrix
golden = np.matmul(mat_a, mat_b)

def matrix_to_c_array(name, mat):
    """Convert a numpy matrix into a C 1D row-major array string."""
    flat = mat.flatten()
    c_str = f"float {name}[elems*elems] = {{\n"
    for i in range(elems):
        row = ", ".join(f"{flat[i*elems + j]:.6f}f" for j in range(elems))
        c_str += f"    {row},\n"
    c_str += "};\n"
    return c_str

# File path in the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "data.h")

# Generate data.h
with open(file_path, "w") as f:
    f.write("#ifndef DATA_H\n")
    f.write("#define DATA_H\n\n")
    f.write("/* Square matrix dimension */\n")
    f.write("/* Max size: 24. 24*24*3(num of matrices) * 64 bits/element --> 110,592 bits, so 110KiB */\n")
    f.write("#ifndef elems\n")
    f.write(f"#define elems {elems}\n")
    f.write("#endif\n\n")
    f.write(f"float *mat_a_TCDM;\n")
    f.write(f"float *mat_b_TCDM;\n")
    f.write(f"float *dst_TCDM;\n")

    # Write matrices in row-major 1D form
    f.write("/* Matrices in row-major 1D order */\n")
    f.write(matrix_to_c_array("mat_a", mat_a) + "\n")
    f.write(matrix_to_c_array("mat_b", mat_b) + "\n")
    f.write(matrix_to_c_array("golden", golden) + "\n")
    
    f.write("#endif\n")

print(f"data.h successfully generated at: {file_path}")
