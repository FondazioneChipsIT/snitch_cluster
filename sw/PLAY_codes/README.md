# PLAY CODES
This folder follows the structure of the repository found at https://github.com/FondazioneChipsIT/PLAY/tree/spatz_support, trying to compare Snitch to Spatz and Pulp Open. 
Not all codes in that folder have been implemented and the code structure is different (it follows the previouslyy found structure in the snitch cluster codes).
The data generation is primitive, and if you have some optimizations that could speed up the code they are always welcome, as my coding skills are intermidate but not as a real coder.
# CODING 
If you are new to coding with Snitch and SSRs i recommend starting from the vector codes, they all have the same main and almost same kernel.
Everything should be adequately commented.
The last and most complex codes take some time to fully understand the mess that I made (with good results).
# Extracting performance metrics
The codes automatically print the metrics.
If you want you can also use the python script found in the benchmarks folder.

Steps:
Launch the simulation of the code, make the traces found in the log folder. Afterwards launch the script performance_cycles.py, it will print the cycles/IPC/FLOP for each core andthe mean cycles. 
All the codes implemented by PULP will need this script, while matmul_FP64/FIR will print the perf metrics when executing the code.
This script can be used for each code that uses the function mccycle() only 2 times (so that section 0 is the boot, section 1 the hot loop and section 2 the end).
The function should be called before and after the section that we want to analyze. 

If needed, modify the python script to save the metrics in a file.

E.G. The AXPY code will print 1.6 flop/cycle, while the script 0.8. This is correct, as the script is general and considers a maximum of 1 FLOP/cycle, even though some
instructions (like FMADD) are considered to be 2 FLOPs. Still the result is correct but scaled to be from 0 to 1.