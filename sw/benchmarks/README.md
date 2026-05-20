# Benchmarks
These are benchmarks either already present in the kernels folder or newly written.
They have been copied and renamed in this folder for organizations sake.

The matmul kernel is more optimized than the one found in the PLAY_Codes to showcase how one can fully optimize the implementation on Snitch.

# Preparing your kernel
 
To best calculate every performance metric, you should write your code like this:


-   `snrt_cluster_hw_barrier();`
    `snrt_mcycle();`
    `Part of the code that you want to benchmark`
    `snrt_cluster_hw_barrier();`
    `snrt_mcycle();`

Thus all the cores will be synched before running the code.

# Extracting and plotting performance metrics
Launch the simulation of the code, make the traces found in the log folder with

- `make traces SIM_DIR=. -j`

Afterwards search the traces of the core zero for the two calls of mcycle, and take their PC.
Call the function pulp_cluster_flop_parsing, with the options:

- `--folder logs/`
- `--start #FIRST mcycle PC#`
- `--end #second mcycle PC#`

This will print on the terminal the cycles and flops for each core. 
If you want to put the result in a .txt, do:

- `python sw/benchmarks/pulp_cluster_flop_parsing.py --folder logs --start "${START_PC}" --end "${END_PC}" > "${OUTPUT_FILE}"`

# Automatic benchmarking

The file auto_benchmark.sh automatically does this procedure for each file in its list.
It compiles all the codes, takes one per one from the list and runs them, makes the traces, and saves the result of the parsing script in the folder result.
If needed, add and remove kernels from the list.
WARING: it will take a lot of time to run all the kernels as they are configured right now.
 