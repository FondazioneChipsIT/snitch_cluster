# Benchmarks
These are benchmarks either already present in the kernels folder or newly written.
They have been copied and renamed in this folder for organizations sake.

The matmul kernel is more optimized than the one found in the PLAY_Codes to showcase how one can fully optimize the implementation on Snitch.

# Extracting and plotting performance metrics
Launch the simulation of the code, make the traces found in the log folder with

- `make traces SIM_DIR=. -j`

Afterwards launch the script performance_parsing.py. It will ask as an input the name that will be used to save the files (.txt and .csv), e.g. "matmul". 
It will also print on the terminal all the cores metrics and the mean ones.

All the codes implemented in this folder will need this script to extract performance metrics.
If you add or change a kernel, use the function

- `mcycle();`

Only before and after the section you want to analyze. This constraint is given by the fact that the script takes the cycles and other parameters in the last section of the traces, where there are the different sections of the code. The only way to divide the code is using this function. The script takes the metrics from section 1 (section 0 being the boot+preparation and section 2 the final one).

The script will save a .csv that is updated for each run of the kernel with the same name, and a .txt for each benchmark. The csv can be used to plot, using plot_benchmarks.py.

The plotting script will plot the optimized version against the naive version, if it exists. To use this feature, you must call the kernel that you analyze with performance_parsing.py with a name, for example "matmul", for the opt version, and then the same name plus "_naive", e.g. "matmul_naive". When calling the plotting script use as a name the opt version, e.g. matmul. It will also add the speedup for the algebrical flops (excluding fadd and fsub, that are sometimes used to storeback results). To see all the flops analyze the sustained plot, with the plot_sustained_speedup.py script.

To start, and understand better, check the general results in the results folder. There you will find different kernels, with many runs, naive and optimized versions and the plots.
