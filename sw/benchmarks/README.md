# Benchmarks
These are benchmarks either already present in the kernels folder or newly written.
They have been copied and renamed in this folder for organizations sake.

The matmul kernel is more optimized than the one found in the PLAY_Codes to showcase how one can fully optimize the implementation on Snitch.

# Extracting performance metrics
Launch the simulation of the code, make the traces found in the log folder. Afterwards launch the script performance_cycles.py, it will print the cycles/IPC/FLOP/ for each core and the mean for the cluster. It will also save them in a .txt file for each N.

All the codes implemented in this folder will need this script to extract performance metrics.
This script can be used for each code that uses the function mccycle() only 2 times (so that section 0 is the boot, section 1 the hot loop and section 2 the end).
The function should be called before and after the section that we want to analyze. 

The script will save a .csv common file to all benchmarks, and a txt for each benchmark. The csv can be used to plot, using plot_benchmarks.py.

The plotting script will plot the optimized version against the naive version, if it exists. To use this functio, you must call the kernel that you analyze with a name, for example "matmul", for the opt version and then the same name plus "_naive", e.g. "matmul_naive". It will also add the speedup for the algebrical flops (excluding fadd and fsub, that are sometimes used to storeback results). To see all the flops analyze the sustained plot.
