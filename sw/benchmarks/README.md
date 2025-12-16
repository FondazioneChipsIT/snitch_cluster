# Benchmarks
These are benchmarks either already present in the kernels folder or newly written.
They have been copied and renamed in this folder for organizations sake.

# Extracting performance metrics
Launch the simulation of the code, make the traces found in the log folder. Afterwards launch the script performance_cycles.py, it will print the cycles/IPC/FLOP for each core andthe mean cycles. 
All the codes implemented by PULP will need this script, while matmul_FP64/FIR will print the perf metrics when executing the code.
This script can be used for each code that uses the function mccycle() only 2 times (so that section 0 is the boot, section 1 the hot loop and section 2 the end).
The function should be called before and after the section that we want to analyze. 