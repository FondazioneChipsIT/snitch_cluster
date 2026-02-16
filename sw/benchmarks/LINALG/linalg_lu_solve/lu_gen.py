import numpy as np
import os

def generate_matrix(elems):
    mat = np.zeros((elems, elems), dtype=float)
    for i in range(elems):
        for j in range(elems):
            mat[i, j] = ((i + j) % elems) + 1 + i
    return mat

def lu_decomposition(A):
    """ LU con pivoting parziale """
    A = A.copy()
    n = A.shape[0]
    piv = list(range(n))

    for k in range(n - 1):
        # pivoting parziale
        max_row = max(range(k, n), key=lambda i: abs(A[i, k]))
        if k != max_row:
            A[[k, max_row]] = A[[max_row, k]]
            piv[k], piv[max_row] = piv[max_row], piv[k]

        # eliminazione gaussiana
        for i in range(k + 1, n):
            A[i, k] /= A[k, k]
            for j in range(k + 1, n):
                A[i, j] -= A[i, k] * A[k, j]

    return A, piv

def write_header(filename, A, piv):
    n = A.shape[0]
    with open(filename, "w") as f:
        f.write("// Generated LU matrix data\n")
        f.write(f"#define N {n}\n\n")
        
        f.write("static float mat_LU[N*N] = {\n")
        # row-major flattening
        flat = A.flatten()
        for i, val in enumerate(flat):
            f.write(f"    {val:.10f},")
            if (i+1) % n == 0:
                f.write("\n")
        f.write("};\n\n")

        # salva vettore pivot
        f.write("static int pivots[N] = {")
        f.write(",".join(str(p) for p in piv))
        f.write("};\n")

def main():
    elems = 32  # <-- Modifica la dimensione della matrice qui
    A = generate_matrix(elems)
    LU, piv = lu_decomposition(A)
    
    # percorso assoluto rispetto allo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "data", "data_LU.h")

    write_header(output_path, LU, piv)
    print("LU matrix generated and saved to data_LU.h")

if __name__ == "__main__":
    main()
