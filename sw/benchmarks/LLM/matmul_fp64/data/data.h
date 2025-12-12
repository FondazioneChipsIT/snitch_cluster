#ifndef DATA_H
#define DATA_H

/* Square matrix dimension */
/* Max size: 24. 24*24*3(num of matrices) * 64 bits/element --> 110,592 bits, so 110KiB*/
#ifndef elems
#define elems 32
#endif

/* TCDM pointers  */
double *mat_a; 
double *mat_b; 
double *dst; 

/* Cycle and perf metrics */
uint64_t start_cycle[16], end_cycle[16], total_cycles[16];
double flop_cycle[16];

#endif 